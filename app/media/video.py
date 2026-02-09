import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import imageio
import numpy as np
from PIL import Image

from app.core.config import settings
from app.utils.common import build_timestamped_filename, ensure_directories

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


def compute_target_size(input_width: int, input_height: int, long_edge: int) -> tuple[int, int]:
    if input_width <= 0 or input_height <= 0 or long_edge <= 0:
        return max(2, input_width), max(2, input_height)

    current_long_edge = max(input_width, input_height)
    if current_long_edge <= 0:
        return max(2, input_width), max(2, input_height)

    scale = long_edge / float(current_long_edge)
    target_width = max(2, int(round(input_width * scale)))
    target_height = max(2, int(round(input_height * scale)))

    if target_width % 2 != 0:
        target_width -= 1
    if target_height % 2 != 0:
        target_height -= 1

    target_width = max(2, target_width)
    target_height = max(2, target_height)
    return target_width, target_height


def resize_frames_to_target(
    frames: Sequence[np.ndarray],
    target_width: int,
    target_height: int,
) -> list[np.ndarray]:
    if not frames:
        return []
    if target_width <= 0 or target_height <= 0:
        return list(frames)

    sample = frames[0]
    if sample.shape[1] == target_width and sample.shape[0] == target_height:
        return list(frames)

    resized_frames: list[np.ndarray] = []
    for frame in frames:
        resized = np.array(
            Image.fromarray(frame).resize((target_width, target_height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        resized_frames.append(resized)
    return resized_frames


def deflicker_frames(frames: Sequence[np.ndarray], window: int = 3) -> list[np.ndarray]:
    if not frames:
        return []
    if window <= 1 or len(frames) <= 1:
        return list(frames)

    half = window // 2
    output: list[np.ndarray] = []
    for idx in range(len(frames)):
        start = max(0, idx - half)
        end = min(len(frames), idx + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        output.append(median_frame)
    return output


def _fit_frame_to_aspect_ratio(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return frame

    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return frame

    # Build an exact rational aspect target and create a matching canvas.
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

    canvas_width = unit_width * max_multiplier
    canvas_height = unit_height * max_multiplier
    if canvas_width <= 0 or canvas_height <= 0:
        return frame

    scale = min(canvas_width / frame_width, canvas_height / frame_height)
    resized_width = max(2, int(round(frame_width * scale)))
    resized_height = max(2, int(round(frame_height * scale)))
    resized_width = min(canvas_width, resized_width - (resized_width % 2))
    resized_height = min(canvas_height, resized_height - (resized_height % 2))
    if resized_width <= 0 or resized_height <= 0:
        return frame

    resized = np.array(
        Image.fromarray(frame).resize((resized_width, resized_height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset_x = max((canvas_width - resized_width) // 2, 0)
    offset_y = max((canvas_height - resized_height) // 2, 0)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def save_frames_to_mp4(
    frames: Iterable[np.ndarray | Image.Image],
    fps: int,
    target_aspect_width: int | None = None,
    target_aspect_height: int | None = None,
    output_path: Path | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    enable_deflicker: bool = False,
    deflicker_window: int = 3,
) -> Path:
    ensure_directories()

    if output_path is None:
        output_path = settings.outputs_dir / build_timestamped_filename(prefix="hunyuan_i2v")

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("No frames to encode.")
    LOGGER.info("Encoding video to mp4. output=%s fps=%s frames=%d", output_path, fps, len(frame_list))

    normalized_frames = [_normalize_frame(frame) for frame in frame_list]
    if enable_deflicker:
        LOGGER.info("Applying deflicker. window=%s frames=%d", deflicker_window, len(normalized_frames))
        normalized_frames = deflicker_frames(normalized_frames, window=deflicker_window)
    if target_width and target_height:
        LOGGER.info(
            "Resizing frames to target. target=%sx%s",
            target_width,
            target_height,
        )
        normalized_frames = resize_frames_to_target(
            normalized_frames,
            target_width=target_width,
            target_height=target_height,
        )
    if target_aspect_width and target_aspect_height:
        original_shape = normalized_frames[0].shape
        normalized_frames = [
            _fit_frame_to_aspect_ratio(
                frame=frame,
                target_width=target_aspect_width,
                target_height=target_aspect_height,
            )
            for frame in normalized_frames
        ]
        updated_shape = normalized_frames[0].shape
        if updated_shape != original_shape:
            LOGGER.info(
                "Adjusted frames to match input aspect ratio (fit+pad). target=%sx%s before=%sx%s after=%sx%s",
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
