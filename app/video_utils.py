import logging
from pathlib import Path
from typing import Iterable

import imageio
import numpy as np
from PIL import Image

from app.config import settings
from app.utils import build_timestamped_filename, ensure_directories

LOGGER = logging.getLogger(__name__)


def _normalize_frame(frame: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.array(frame.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3:
            raise ValueError("Invalid frame shape. Expected HxW or HxWxC.")
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise ValueError("Invalid frame channels. Expected 3-channel RGB frames.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def save_frames_to_mp4(
    frames: Iterable[np.ndarray | Image.Image],
    fps: int,
    output_path: Path | None = None,
) -> Path:
    ensure_directories()

    if output_path is None:
        output_path = settings.outputs_dir / build_timestamped_filename(prefix="hunyuan_i2v")

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames to encode.")
    LOGGER.info("Encoding video to mp4. output=%s fps=%s frames=%d", output_path, fps, len(frame_list))

    normalized_frames = [_normalize_frame(frame) for frame in frame_list]

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
    )
    try:
        for frame in normalized_frames:
            writer.append_data(frame)
    finally:
        writer.close()

    LOGGER.info("Video encode complete. output=%s", output_path)
    return output_path
