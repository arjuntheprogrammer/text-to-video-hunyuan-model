from __future__ import annotations

from typing import Dict, List

PROMPT_FIELD_OPTIONS: Dict[str, List[str]] = {
    "subject": [
        "person",
        "product",
        "food",
        "fashion",
        "animal",
        "vehicle",
        "architecture",
        "landscape",
        "cityscape",
        "gadget",
    ],
    "action": [
        "walking",
        "turning head",
        "smiling",
        "hand gesture",
        "hair movement",
        "pouring",
        "rotating",
        "hovering",
        "panning reveal",
        "still",
    ],
    "camera_motion": [
        "static",
        "slow pan",
        "tilt",
        "dolly in",
        "dolly out",
        "orbit",
        "handheld",
        "zoom in",
        "zoom out",
    ],
    "shot_type": [
        "close-up",
        "medium",
        "wide",
        "macro",
        "overhead",
        "low angle",
        "high angle",
    ],
    "lighting": [
        "soft daylight",
        "golden hour",
        "studio softbox",
        "neon",
        "backlit",
        "overcast",
        "candlelight",
    ],
    "mood": [
        "cinematic",
        "calm",
        "energetic",
        "moody",
        "dreamy",
        "documentary",
        "romantic",
        "dramatic",
    ],
}


def _add_field(lines: List[str], label: str, value: str | None) -> None:
    if value is None:
        return
    cleaned = value.strip()
    if not cleaned:
        return
    lines.append(f"{label}: {cleaned}")


def build_structured_prompt(
    user_prompt: str,
    caption: str | None = None,
    subject: str | None = None,
    action: str | None = None,
    camera_motion: str | None = None,
    shot_type: str | None = None,
    lighting: str | None = None,
    mood: str | None = None,
    default_suffix: str | None = None,
) -> str:
    parts: List[str] = []

    if caption:
        caption_clean = caption.strip()
        if caption_clean:
            parts.append(f"Image description: {caption_clean}")

    field_lines: List[str] = []
    _add_field(field_lines, "Subject", subject)
    _add_field(field_lines, "Action", action)
    _add_field(field_lines, "Camera", camera_motion)
    _add_field(field_lines, "Shot", shot_type)
    _add_field(field_lines, "Lighting", lighting)
    _add_field(field_lines, "Mood", mood)

    if field_lines:
        parts.append("Creative direction:\n" + "\n".join(field_lines))

    user_prompt_clean = (user_prompt or "").strip()
    if user_prompt_clean:
        parts.append(user_prompt_clean)

    prompt_body = "\n\n".join([part for part in parts if part.strip()]).strip()

    suffix = (default_suffix or "").strip()
    if suffix and suffix.lower() not in prompt_body.lower():
        if prompt_body:
            prompt_body = f"{prompt_body}\n\n{suffix}"
        else:
            prompt_body = suffix

    return prompt_body
