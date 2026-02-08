import datetime as dt
import re
from pathlib import Path

from app.config import settings


def ensure_directories() -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)


def build_timestamped_filename(prefix: str = "video", suffix: str = ".mp4") -> str:
    now = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{now}{suffix}"


def sanitize_filename(filename: str) -> str:
    if not filename:
        return ""
    if "/" in filename or "\\" in filename:
        return ""
    if ".." in filename:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        return ""
    return filename


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def ensure_output_path(filename: str) -> Path:
    return settings.outputs_dir / filename
