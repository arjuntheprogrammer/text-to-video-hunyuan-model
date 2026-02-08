import logging
import math
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


def _crop_frame_to_aspect_ratio(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return frame

    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return frame

    # Build an exact rational aspect target and center-crop to that ratio.
    ratio_gcd = math.gcd(target_width, target_height)
    unit_width = target_width // ratio_gcd
    unit_height = target_height // ratio_gcd
    max_multiplier = min(frame_width // unit_width, frame_height // unit_height)
    if max_multiplier <= 0:
        return frame

    # Keep encoded mp4 dimensions even for yuv420p compatibility.
    while max_multiplier > 1 and (
        ((unit_width * max_multiplier) % 2 != 0) or ((unit_height * max_multiplier) % 2 != 0)
    ):
        max_multiplier -= 1

    crop_width = unit_width * max_multiplier
    crop_height = unit_height * max_multiplier
    if crop_width <= 0 or crop_height <= 0:
        return frame

    offset_x = max((frame_width - crop_width) // 2, 0)
    offset_y = max((frame_height - crop_height) // 2, 0)
    return frame[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width]


def save_frames_to_mp4(
    frames: Iterable[np.ndarray | Image.Image],
    fps: int,
    target_aspect_width: int | None = None,
    target_aspect_height: int | None = None,
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
    if target_aspect_width and target_aspect_height:
        original_shape = normalized_frames[0].shape
        normalized_frames = [
            _crop_frame_to_aspect_ratio(
                frame=frame,
                target_width=target_aspect_width,
                target_height=target_aspect_height,
            )
            for frame in normalized_frames
        ]
        updated_shape = normalized_frames[0].shape
        if updated_shape != original_shape:
            LOGGER.info(
                "Adjusted frames to match input aspect ratio. target=%sx%s before=%sx%s after=%sx%s",
                target_aspect_width,
                target_aspect_height,
                original_shape[1],
                original_shape[0],
                updated_shape[1],
                updated_shape[0],
            )

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        macro_block_size=1,
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
