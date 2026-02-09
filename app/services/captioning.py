from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
from PIL import Image

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


class Captioner:
    def __init__(self) -> None:
        requested_device = settings.caption_device.strip().lower()
        if not requested_device:
            requested_device = "cpu"
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device != "cpu" and not torch.cuda.is_available():
            LOGGER.warning("Caption device %s unavailable. Falling back to CPU.", requested_device)
            requested_device = "cpu"
        self.device = requested_device
        self.model_id = settings.caption_model_id
        self.max_tokens = settings.caption_max_tokens
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        LOGGER.info(
            "Loading captioning model. model_id=%s device=%s", self.model_id, self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(settings.models_dir),
            token=settings.hf_token or None,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(settings.models_dir),
            token=settings.hf_token or None,
        )
        self.model.to(self.device)
        self.model.eval()

    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()


_captioner: Optional[Captioner] = None
_captioner_lock = threading.Lock()


def get_captioner() -> Optional[Captioner]:
    if not settings.enable_captioning:
        return None
    global _captioner
    if _captioner is None:
        with _captioner_lock:
            if _captioner is None:
                try:
                    _captioner = Captioner()
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Captioner load failed: %s", exc)
                    _captioner = None
    return _captioner


def caption_image(image: Image.Image) -> str | None:
    captioner = get_captioner()
    if captioner is None:
        return None
    try:
        return captioner.caption(image)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Captioning failed: %s", exc)
        return None
