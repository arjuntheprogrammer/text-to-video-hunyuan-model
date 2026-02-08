import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.config import settings
from app.utils import clamp_int, ensure_directories
from app.video_utils import save_frames_to_mp4

LOGGER = logging.getLogger(__name__)


class HunyuanVideoPipelineManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._lock = threading.Lock()
        self.pipe: Any | None = None
        self.model_loaded = False
        self.load_error: str | None = None

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
                )

                self.pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
                    settings.model_id,
                    transformer=transformer,
                    torch_dtype=self.dtype,
                    cache_dir=str(settings.models_dir),
                    token=token,
                )
            except ImportError:
                from diffusers import DiffusionPipeline

                self.pipe = DiffusionPipeline.from_pretrained(
                    settings.model_id,
                    torch_dtype=self.dtype,
                    cache_dir=str(settings.models_dir),
                    trust_remote_code=True,
                    token=token,
                )
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
            return image
        width, height = image.size
        current_max_side = max(width, height)
        if current_max_side <= max_side:
            return image

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
    ) -> tuple[Path, int, int, int]:
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
        original_size = source_image.size

        used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        result = None
        used_attempt_frames = safe_num_frames
        used_attempt_steps = safe_steps
        used_attempt_resolution = source_image.size
        generation_start = time.perf_counter()
        last_oom: BaseException | None = None
        attempted_profiles: list[str] = []

        original_max_side = max(original_size)
        if original_max_side > settings.max_input_image_side:
            resolution_seed = [640, 512, 384]
        else:
            resolution_seed = [settings.max_input_image_side, 768, 640, 512, 384]
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

            conservative_mode = (
                input_image.size != original_size
                or safe_num_frames > settings.oom_safe_num_frames
                or safe_steps > settings.oom_safe_steps
            )
            if conservative_mode:
                preferred_frames = min(safe_num_frames, settings.oom_safe_num_frames)
                preferred_steps = min(safe_steps, settings.oom_safe_steps)
                if input_image.size != original_size:
                    preferred_frames = min(preferred_frames, 16)
                    preferred_steps = min(preferred_steps, 10)
                # Use stricter profile as resolution gets smaller to avoid repeated OOM cascades.
                if max(input_image.size) <= 640:
                    preferred_frames = min(preferred_frames, 24)
                    preferred_steps = min(preferred_steps, 10)
                if max(input_image.size) <= 512:
                    preferred_frames = min(preferred_frames, settings.min_num_frames)
                    preferred_steps = min(preferred_steps, settings.min_inference_steps)
                frame_step_candidates = [
                    (preferred_frames, preferred_steps),
                    (min(preferred_frames, 24), min(preferred_steps, 10)),
                    (settings.min_num_frames, settings.min_inference_steps),
                ]
            else:
                frame_step_candidates = [
                    (safe_num_frames, safe_steps),
                    (min(safe_num_frames, 128), min(safe_steps, 24)),
                    (min(safe_num_frames, 96), min(safe_steps, 22)),
                    (min(safe_num_frames, 64), min(safe_steps, 18)),
                    (min(safe_num_frames, 48), min(safe_steps, 14)),
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
                            result = self.pipe(
                                image=input_image,
                                prompt=prompt,
                                num_frames=attempt_frames,
                                guidance_scale=safe_guidance,
                                num_inference_steps=attempt_steps,
                                generator=generator,
                                callback_on_step_end=self._make_step_callback(
                                    total_steps=attempt_steps,
                                    resolution=input_image.size,
                                    num_frames=attempt_frames,
                                    num_inference_steps=attempt_steps,
                                ),
                                callback_on_step_end_tensor_inputs=["latents"],
                            )
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

        output_path = save_frames_to_mp4(frames=frames, fps=safe_fps)
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
        return output_path, used_seed, used_attempt_frames, safe_fps


_PIPELINE_MANAGER = HunyuanVideoPipelineManager()


def get_pipeline_manager() -> HunyuanVideoPipelineManager:
    return _PIPELINE_MANAGER
