import inspect
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.captioning import caption_image
from app.config import settings
from app.prompt_builder import build_structured_prompt
from app.utils import clamp_int, ensure_directories
from app.video_utils import compute_target_size, save_frames_to_mp4

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    output_path: Path
    seed: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    used_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    effective_prompt_len: int
    negative_prompt_len: int
    duration_seconds: float


class HunyuanVideoPipelineManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._lock = threading.Lock()
        self.pipe: Any | None = None
        self._pipe_call_arg_names: set[str] = set()
        self.model_loaded = False
        self.load_error: str | None = None
        self._auto_max_input_side: int | None = None

        self._configure_runtime()
        self._load_pipeline()

    def _configure_runtime(self) -> None:
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            LOGGER.info("CUDA detected. Enabled TF32 matmul and cuDNN optimizations.")
        else:
            LOGGER.warning("CUDA not available. Falling back to CPU inference (slow).")

    def _load_pipeline(self) -> None:
        ensure_directories()
        start_time = time.perf_counter()
        LOGGER.info("Loading model pipeline. model_id=%s cache_dir=%s", settings.model_id, settings.models_dir)
        token = settings.hf_token or None
        if token is None:
            LOGGER.warning(
                "HF_TOKEN is not set. Model download/load may fail for gated repositories."
            )
        try:
            def _load_with(local_only: bool) -> Any:
                try:
                    from diffusers import (
                        HunyuanVideoImageToVideoPipeline,
                        HunyuanVideoTransformer3DModel,
                    )

                    transformer_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                        settings.model_id,
                        subfolder="transformer",
                        torch_dtype=transformer_dtype,
                        cache_dir=str(settings.models_dir),
                        token=token,
                        local_files_only=local_only,
                    )

                    return HunyuanVideoImageToVideoPipeline.from_pretrained(
                        settings.model_id,
                        transformer=transformer,
                        torch_dtype=self.dtype,
                        cache_dir=str(settings.models_dir),
                        token=token,
                        local_files_only=local_only,
                    )
                except ImportError:
                    from diffusers import DiffusionPipeline

                    return DiffusionPipeline.from_pretrained(
                        settings.model_id,
                        torch_dtype=self.dtype,
                        cache_dir=str(settings.models_dir),
                        trust_remote_code=True,
                        token=token,
                        local_files_only=local_only,
                    )

            try:
                self.pipe = _load_with(settings.local_files_only)
            except Exception as exc:
                if settings.local_files_only and settings.allow_remote_fallback:
                    LOGGER.warning(
                        "Local-only model load failed; retrying with remote fallback. error=%s",
                        exc,
                    )
                    self.pipe = _load_with(False)
                else:
                    raise
            try:
                self._pipe_call_arg_names = set(inspect.signature(self.pipe.__call__).parameters.keys())
            except Exception:  # pragma: no cover
                self._pipe_call_arg_names = set()
            using_cpu_offload = False
            if self.device == "cuda":
                if settings.enable_sequential_cpu_offload and hasattr(
                    self.pipe, "enable_sequential_cpu_offload"
                ):
                    self.pipe.enable_sequential_cpu_offload()
                    using_cpu_offload = True
                    LOGGER.info("Sequential CPU offload enabled.")
                elif settings.enable_model_cpu_offload and hasattr(
                    self.pipe, "enable_model_cpu_offload"
                ):
                    self.pipe.enable_model_cpu_offload()
                    using_cpu_offload = True
                    LOGGER.info("Model CPU offload enabled.")

            if not using_cpu_offload:
                self.pipe.to(self.device)

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing("auto")
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()

            if self.device == "cuda" and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                if settings.enable_xformers:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        LOGGER.info("xFormers memory efficient attention enabled.")
                    except Exception as exc:  # pragma: no cover
                        LOGGER.warning("xFormers unavailable: %s", exc)
                else:
                    LOGGER.info(
                        "xFormers memory efficient attention disabled (ENABLE_XFORMERS not set)."
                    )

            self.model_loaded = True
            LOGGER.info(
                "Loaded model pipeline successfully. model_id=%s elapsed=%.2fs",
                settings.model_id,
                time.perf_counter() - start_time,
            )
        except Exception as exc:
            self.model_loaded = False
            self.load_error = str(exc)
            LOGGER.exception("Failed to load model pipeline: %s", exc)

    def _resolve_max_input_side(self) -> int:
        if not settings.auto_max_input_side or settings.max_input_image_side_override:
            return settings.max_input_image_side

        if self._auto_max_input_side is not None:
            return self._auto_max_input_side

        max_side = settings.max_input_image_side
        if self.device == "cuda":
            try:
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                total_gb = props.total_memory / (1024**3)
                if total_gb >= 24:
                    max_side = 1536
                elif total_gb >= 16:
                    max_side = 1280
                else:
                    max_side = 1024
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Auto max input side detection failed: %s", exc)

        self._auto_max_input_side = max_side
        LOGGER.info("Auto max input side resolved to %s", max_side)
        return max_side

    def _build_effective_prompts(
        self,
        user_prompt: str,
        caption: str | None,
        subject: str | None,
        action: str | None,
        camera_motion: str | None,
        shot_type: str | None,
        lighting: str | None,
        mood: str | None,
        negative_prompt: str | None,
    ) -> tuple[str, str | None]:
        effective_prompt = build_structured_prompt(
            user_prompt=user_prompt,
            caption=caption,
            subject=subject,
            action=action,
            camera_motion=camera_motion,
            shot_type=shot_type,
            lighting=lighting,
            mood=mood,
            default_suffix=settings.default_prompt_suffix,
        )
        user_negative = (negative_prompt or "").strip()
        if user_negative:
            resolved_negative = user_negative
        else:
            resolved_negative = settings.default_negative_prompt.strip() or None
        return effective_prompt, resolved_negative

    def _truncate_prompt_for_tokenizers(self, prompt: str, tokenizers: list[Any]) -> str:
        truncated = prompt
        for tokenizer in tokenizers:
            if tokenizer is None:
                continue
            max_len = getattr(tokenizer, "model_max_length", None)
            if not isinstance(max_len, int) or max_len <= 0:
                continue
            try:
                encoded = tokenizer(
                    truncated,
                    truncation=True,
                    max_length=max_len,
                    return_overflowing_tokens=True,
                )
            except Exception:  # pragma: no cover
                continue
            input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if not input_ids:
                continue
            if hasattr(tokenizer, "decode"):
                decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
                if decoded and decoded != truncated:
                    LOGGER.info(
                        "Truncated prompt for %s to max_length=%s",
                        tokenizer.__class__.__name__,
                        max_len,
                    )
                    truncated = decoded
        return truncated

    @staticmethod
    def _truncate_prompt_words(prompt: str, max_words: int) -> str:
        if max_words <= 0:
            return prompt
        words = prompt.split()
        if len(words) <= max_words:
            return prompt
        return " ".join(words[:max_words]).strip()

    @staticmethod
    def _extract_frames(result: Any) -> list[Any]:
        frames = None

        if hasattr(result, "frames"):
            frames = result.frames
        elif isinstance(result, dict) and "frames" in result:
            frames = result["frames"]

        if frames is None:
            raise RuntimeError("Pipeline did not return frames.")

        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            return frames[0]
        if isinstance(frames, tuple):
            frames = list(frames)

        return list(frames)

    @staticmethod
    def _resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
        if max_side <= 0:
            max_side = max(image.size)
        width, height = image.size
        current_max_side = max(width, height)
        if current_max_side <= max_side:
            # HunyuanVideo requires width/height divisible by 16.
            if width % 16 == 0 and height % 16 == 0:
                return image
            new_width = max(16, (width // 16) * 16)
            new_height = max(16, (height // 16) * 16)
            if new_width == width and new_height == height:
                return image
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        scale = max_side / float(current_max_side)
        # Keep dimensions divisible by 16 for more stable latent shape handling.
        new_width = max(16, int((width * scale) // 16) * 16)
        new_height = max(16, int((height * scale) // 16) * 16)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _build_progress_bar(current_step: int, total_steps: int, width: int) -> str:
        safe_total = max(total_steps, 1)
        safe_width = max(width, 8)
        progress_ratio = min(max(current_step / safe_total, 0.0), 1.0)
        filled = int(progress_ratio * safe_width)
        bar = ("#" * filled) + ("-" * (safe_width - filled))
        return f"[{bar}] {current_step}/{safe_total} ({progress_ratio * 100:.1f}%)"

    def _make_step_callback(
        self,
        total_steps: int,
        resolution: tuple[int, int],
        num_frames: int,
        num_inference_steps: int,
    ):
        last_logged = {"step": 0}

        def _callback(*args):
            if len(args) != 4:
                return args[-1] if args else {}
            _, step_index, _timestep, callback_kwargs = args
            current_step = int(step_index) + 1
            log_every = max(1, settings.progress_log_every_steps)
            if (
                current_step == 1
                or current_step == total_steps
                or current_step % log_every == 0
            ) and current_step > last_logged["step"]:
                bar = self._build_progress_bar(current_step, total_steps, settings.progress_bar_width)
                LOGGER.info(
                    "Denoise progress %s | res=%sx%s frames=%s steps=%s",
                    bar,
                    resolution[0],
                    resolution[1],
                    num_frames,
                    num_inference_steps,
                )
                last_logged["step"] = current_step
            return callback_kwargs

        return _callback

    def generate_video(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        fps: int,
        seed: int | None,
        subject: str | None = None,
        action: str | None = None,
        camera_motion: str | None = None,
        shot_type: str | None = None,
        lighting: str | None = None,
        mood: str | None = None,
        negative_prompt: str | None = None,
        output_long_edge: int | None = None,
        enable_deflicker: bool | None = None,
    ) -> GenerationResult:
        if not self.model_loaded or self.pipe is None:
            raise RuntimeError(
                f"Model is not available. Load error: {self.load_error or 'unknown error'}"
            )

        safe_num_frames = clamp_int(num_frames, settings.min_num_frames, settings.max_num_frames)
        safe_steps = clamp_int(
            num_inference_steps,
            settings.min_inference_steps,
            settings.max_inference_steps,
        )
        safe_fps = clamp_int(fps, settings.min_fps, settings.max_fps)
        safe_guidance = max(settings.min_guidance_scale, min(settings.max_guidance_scale, guidance_scale))
        source_image = image.convert("RGB")
        caption = caption_image(source_image) if settings.enable_captioning else None
        effective_prompt, resolved_negative_prompt = self._build_effective_prompts(
            user_prompt=prompt,
            caption=caption,
            subject=subject,
            action=action,
            camera_motion=camera_motion,
            shot_type=shot_type,
            lighting=lighting,
            mood=mood,
            negative_prompt=negative_prompt,
        )
        tokenizers: list[Any] = []
        if self.pipe is not None:
            tokenizers = [
                getattr(self.pipe, "tokenizer", None),
                getattr(self.pipe, "tokenizer_2", None),
            ]
        if tokenizers:
            effective_prompt = self._truncate_prompt_for_tokenizers(effective_prompt, tokenizers)
            if resolved_negative_prompt:
                resolved_negative_prompt = self._truncate_prompt_for_tokenizers(
                    resolved_negative_prompt,
                    tokenizers,
                )
        word_truncated_prompt = self._truncate_prompt_words(
            effective_prompt,
            settings.max_prompt_words,
        )
        if word_truncated_prompt != effective_prompt:
            LOGGER.info(
                "Prompt truncated by word limit. before_words=%d after_words=%d limit=%d",
                len(effective_prompt.split()),
                len(word_truncated_prompt.split()),
                settings.max_prompt_words,
            )
            effective_prompt = word_truncated_prompt
        if resolved_negative_prompt:
            word_truncated_negative = self._truncate_prompt_words(
                resolved_negative_prompt,
                settings.max_negative_prompt_words,
            )
            if word_truncated_negative != resolved_negative_prompt:
                LOGGER.info(
                    "Negative prompt truncated by word limit. before_words=%d after_words=%d limit=%d",
                    len(resolved_negative_prompt.split()),
                    len(word_truncated_negative.split()),
                    settings.max_negative_prompt_words,
                )
                resolved_negative_prompt = word_truncated_negative
        LOGGER.info(
            "Prompt enhancement applied. user_prompt_len=%d effective_prompt_len=%d negative_prompt_len=%d",
            len(prompt),
            len(effective_prompt),
            len(resolved_negative_prompt or ""),
        )

        original_size = source_image.size
        LOGGER.info(
            "Source image received. size=%sx%s aspect_ratio=%.6f",
            original_size[0],
            original_size[1],
            original_size[0] / max(original_size[1], 1),
        )

        used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        result = None
        used_attempt_frames = safe_num_frames
        used_attempt_steps = safe_steps
        used_attempt_resolution = source_image.size
        generation_start = time.perf_counter()
        last_oom: BaseException | None = None
        attempted_profiles: list[str] = []

        original_max_side = max(original_size)
        max_input_side = self._resolve_max_input_side()
        highest_allowed_side = min(original_max_side, max(384, max_input_side))
        resolution_seed = [
            highest_allowed_side,
            2048,
            1536,
            1280,
            1024,
            896,
            768,
            640,
            512,
            384,
        ]
        resolution_candidates = []
        for side in resolution_seed:
            resolved_side = min(original_max_side, max(384, side))
            if resolved_side not in resolution_candidates:
                resolution_candidates.append(resolved_side)

        for resolution_index, max_side in enumerate(resolution_candidates, start=1):
            input_image = self._resize_to_max_side(source_image, max_side)
            if input_image.size != original_size:
                LOGGER.info(
                    "Resolution attempt %d/%d. original=%sx%s resized=%sx%s",
                    resolution_index,
                    len(resolution_candidates),
                    original_size[0],
                    original_size[1],
                    input_image.size[0],
                    input_image.size[1],
                )
            else:
                LOGGER.info(
                    "Resolution attempt %d/%d. using original size=%sx%s",
                    resolution_index,
                    len(resolution_candidates),
                    input_image.size[0],
                    input_image.size[1],
                )

            # Always honor user-requested frames/steps first, then downgrade only after OOM.
            frame_step_candidates = [
                (safe_num_frames, safe_steps),
                (min(safe_num_frames, 128), min(safe_steps, 24)),
                (min(safe_num_frames, 96), min(safe_steps, 22)),
                (min(safe_num_frames, 64), min(safe_steps, 18)),
                (
                    min(safe_num_frames, max(settings.oom_safe_num_frames, 48)),
                    min(safe_steps, max(settings.oom_safe_steps, 14)),
                ),
                (min(safe_num_frames, settings.oom_safe_num_frames), min(safe_steps, settings.oom_safe_steps)),
                (settings.min_num_frames, settings.min_inference_steps),
            ]

            attempts: list[tuple[int, int]] = []
            for candidate_frames, candidate_steps in frame_step_candidates:
                candidate = (
                    clamp_int(candidate_frames, settings.min_num_frames, settings.max_num_frames),
                    clamp_int(candidate_steps, settings.min_inference_steps, settings.max_inference_steps),
                )
                if candidate not in attempts:
                    attempts.append(candidate)

            for attempt_index, (attempt_frames, attempt_steps) in enumerate(attempts, start=1):
                attempted_profiles.append(
                    f"{input_image.size[0]}x{input_image.size[1]}:{attempt_frames}f/{attempt_steps}s"
                )
                LOGGER.info(
                    "Pipeline generation attempt %d/%d at %sx%s. seed=%s frames=%s steps=%s fps=%s guidance=%.2f",
                    attempt_index,
                    len(attempts),
                    input_image.size[0],
                    input_image.size[1],
                    used_seed,
                    attempt_frames,
                    attempt_steps,
                    safe_fps,
                    safe_guidance,
                )
                try:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    with self._lock:
                        with torch.inference_mode():
                            generator = torch.Generator(device=self.device).manual_seed(used_seed)
                            pipe_kwargs: dict[str, Any] = {
                                "image": input_image,
                                "prompt": effective_prompt,
                                "height": input_image.size[1],
                                "width": input_image.size[0],
                                "num_frames": attempt_frames,
                                "guidance_scale": safe_guidance,
                                "num_inference_steps": attempt_steps,
                                "generator": generator,
                                "callback_on_step_end": self._make_step_callback(
                                    total_steps=attempt_steps,
                                    resolution=input_image.size,
                                    num_frames=attempt_frames,
                                    num_inference_steps=attempt_steps,
                                ),
                                "callback_on_step_end_tensor_inputs": ["latents"],
                            }
                            if resolved_negative_prompt:
                                pipe_kwargs["negative_prompt"] = resolved_negative_prompt
                                pipe_kwargs["negative_prompt_2"] = resolved_negative_prompt
                            pipe_kwargs["prompt_2"] = effective_prompt
                            if settings.guidance_rescale is not None:
                                pipe_kwargs["guidance_rescale"] = settings.guidance_rescale
                            if settings.image_guidance_scale is not None:
                                pipe_kwargs["image_guidance_scale"] = settings.image_guidance_scale
                            if settings.strength is not None:
                                pipe_kwargs["strength"] = settings.strength

                            if self._pipe_call_arg_names:
                                pipe_kwargs = {
                                    key: value
                                    for key, value in pipe_kwargs.items()
                                    if key in self._pipe_call_arg_names
                                }

                            result = self.pipe(**pipe_kwargs)
                    used_attempt_frames = attempt_frames
                    used_attempt_steps = attempt_steps
                    used_attempt_resolution = input_image.size
                    break
                except torch.OutOfMemoryError as exc:
                    last_oom = exc
                    LOGGER.warning(
                        "OOM at %sx%s on attempt %d/%d (frames=%s, steps=%s). Retrying with lower settings.",
                        input_image.size[0],
                        input_image.size[1],
                        attempt_index,
                        len(attempts),
                        attempt_frames,
                        attempt_steps,
                    )
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:  # pragma: no cover
                            pass
                    continue
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" in message or "cuda out of memory" in message:
                        last_oom = exc
                        LOGGER.warning(
                            "Runtime OOM at %sx%s on attempt %d/%d (frames=%s, steps=%s). Retrying with lower settings.",
                            input_image.size[0],
                            input_image.size[1],
                            attempt_index,
                            len(attempts),
                            attempt_frames,
                            attempt_steps,
                        )
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            try:
                                torch.cuda.ipc_collect()
                            except Exception:  # pragma: no cover
                                pass
                        continue
                    raise

            if result is None and last_oom is not None:
                LOGGER.info(
                    "All frame/step profiles OOM at resolution %sx%s. Trying lower resolution.",
                    input_image.size[0],
                    input_image.size[1],
                )

            if result is not None:
                break

        if result is None:
            attempted = ", ".join(attempted_profiles)
            raise RuntimeError(
                "Generation failed due to GPU memory limits after retries. "
                f"Tried profiles: {attempted}. Last error: {last_oom}"
            )

        frames = self._extract_frames(result)
        if not frames:
            raise RuntimeError("No frames generated.")

        requested_long_edge = output_long_edge or settings.default_output_long_edge
        if requested_long_edge not in settings.output_long_edge_options:
            LOGGER.warning(
                "Requested output long edge %s not in allowed options %s. Using default %s.",
                requested_long_edge,
                settings.output_long_edge_options,
                settings.default_output_long_edge,
            )
            requested_long_edge = settings.default_output_long_edge

        output_width, output_height = compute_target_size(
            input_width=original_size[0],
            input_height=original_size[1],
            long_edge=requested_long_edge,
        )
        use_deflicker = settings.enable_deflicker if enable_deflicker is None else enable_deflicker

        output_path = save_frames_to_mp4(
            frames=frames,
            fps=safe_fps,
            target_width=output_width,
            target_height=output_height,
            enable_deflicker=use_deflicker,
            deflicker_window=settings.deflicker_window,
        )
        LOGGER.info(
            "Pipeline generation completed. output=%s resolution=%sx%s generated_frames=%d requested_frames=%d used_steps=%d elapsed=%.2fs",
            output_path,
            used_attempt_resolution[0],
            used_attempt_resolution[1],
            len(frames),
            used_attempt_frames,
            used_attempt_steps,
            time.perf_counter() - generation_start,
        )
        duration_seconds = len(frames) / float(safe_fps) if safe_fps > 0 else 0.0
        return GenerationResult(
            output_path=output_path,
            seed=used_seed,
            num_frames=used_attempt_frames,
            fps=safe_fps,
            num_inference_steps=used_attempt_steps,
            guidance_scale=safe_guidance,
            used_resolution=used_attempt_resolution,
            output_resolution=(output_width, output_height),
            effective_prompt_len=len(effective_prompt),
            negative_prompt_len=len(resolved_negative_prompt or ""),
            duration_seconds=duration_seconds,
        )


_PIPELINE_MANAGER = HunyuanVideoPipelineManager()


def get_pipeline_manager() -> HunyuanVideoPipelineManager:
    return _PIPELINE_MANAGER
