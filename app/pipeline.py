import logging
import random
import threading
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
                )

                self.pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
                    settings.model_id,
                    transformer=transformer,
                    torch_dtype=self.dtype,
                    cache_dir=str(settings.models_dir),
                )
            except ImportError:
                from diffusers import DiffusionPipeline

                self.pipe = DiffusionPipeline.from_pretrained(
                    settings.model_id,
                    torch_dtype=self.dtype,
                    cache_dir=str(settings.models_dir),
                    trust_remote_code=True,
                )
            self.pipe.to(self.device)

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing("auto")
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()

            if self.device == "cuda" and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    LOGGER.info("xFormers memory efficient attention enabled.")
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("xFormers unavailable: %s", exc)

            self.model_loaded = True
            LOGGER.info("Loaded model: %s", settings.model_id)
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

        used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=self.device).manual_seed(used_seed)

        with self._lock:
            with torch.inference_mode():
                result = self.pipe(
                    image=image.convert("RGB"),
                    prompt=prompt,
                    num_frames=safe_num_frames,
                    guidance_scale=safe_guidance,
                    num_inference_steps=safe_steps,
                    generator=generator,
                )

        frames = self._extract_frames(result)
        if not frames:
            raise RuntimeError("No frames generated.")

        output_path = save_frames_to_mp4(frames=frames, fps=safe_fps)
        return output_path, used_seed, safe_num_frames, safe_fps


_PIPELINE_MANAGER = HunyuanVideoPipelineManager()


def get_pipeline_manager() -> HunyuanVideoPipelineManager:
    return _PIPELINE_MANAGER
