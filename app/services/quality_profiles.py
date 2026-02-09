from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class QualityProfile:
    name: str
    max_side: int
    max_frames: int
    max_steps: int


def _build_profile(name: str) -> QualityProfile:
    return QualityProfile(
        name=name,
        max_side=settings.max_input_side_by_profile.get(name, 1024),
        max_frames=settings.max_frames_by_profile.get(name, settings.max_num_frames),
        max_steps=settings.max_steps_by_profile.get(name, settings.max_inference_steps),
    )


def get_quality_profile(name: str | None = None) -> QualityProfile:
    profile_name = (name or settings.quality_profile or "balanced").strip().lower()
    if profile_name not in {"low", "balanced", "high"}:
        profile_name = "balanced"
    return _build_profile(profile_name)
