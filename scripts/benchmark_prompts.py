import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from app.core.config import settings
from app.services.pipeline_manager import get_pipeline_manager
from app.utils.common import ensure_directories


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "case"


def _ensure_image(path: Path) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (512, 512), color=(127, 127, 127))
    image.save(path)
    return image


def _load_cases(payload: dict) -> list[dict]:
    if "cases" in payload:
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt benchmarks with the Hunyuan I2V pipeline.")
    parser.add_argument("--json", default="scripts/benchmark_prompts.json", help="Path to benchmark JSON.")
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="Optional samples dir (e.g., tests/samples). Uses subfolders with img.* and prompt.txt.",
    )
    parser.add_argument("--image", default=None, help="Override image path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for all cases.")
    args = parser.parse_args()

    ensure_directories()
    payload = {}
    cases = []
    defaults = {}
    image_path: Path | None = None

    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples dir not found: {samples_dir}")
        for sample_dir in sorted([p for p in samples_dir.iterdir() if p.is_dir()]):
            prompt_file = sample_dir / "prompt.txt"
            if not prompt_file.exists():
                continue
            prompt_text = prompt_file.read_text().strip()
            if not prompt_text:
                continue
            image_file = None
            for ext in ("jpg", "jpeg", "png", "webp"):
                candidate = sample_dir / f"img.{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            if image_file is None:
                continue
            cases.append(
                {
                    "name": sample_dir.name,
                    "prompt": prompt_text,
                    "image_path": str(image_file),
                }
            )
        if not cases:
            raise ValueError(f"No valid samples found in {samples_dir}")
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            raise FileNotFoundError(f"Benchmark JSON not found: {json_path}")

        payload = json.loads(json_path.read_text())
        cases = _load_cases(payload)
        if not cases:
            raise ValueError("No cases found in benchmark JSON.")

        defaults = payload.get("defaults", {}) if isinstance(payload, dict) else {}
        image_path = Path(args.image or payload.get("image_path", "setup/test_input_512.png"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else settings.outputs_dir / "benchmarks" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    manager = get_pipeline_manager()
    if not manager.model_loaded:
        raise RuntimeError(f"Model unavailable: {manager.load_error}")
    results = []

    for idx, case in enumerate(cases, start=1):
        prompt = case.get("prompt", "")
        if not prompt:
            continue

        if "image_path" in case:
            image = _ensure_image(Path(case["image_path"]))
        else:
            if image_path is None:
                image_path = Path(args.image or payload.get("image_path", "setup/test_input_512.png"))
            image = _ensure_image(image_path)

        seed = args.seed if args.seed is not None else case.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        result = manager.generate_video(
            image=image,
            prompt=prompt,
            num_frames=int(case.get("num_frames", defaults.get("num_frames", settings.default_num_frames))),
            guidance_scale=float(
                case.get("guidance_scale", defaults.get("guidance_scale", settings.default_guidance_scale))
            ),
            num_inference_steps=int(case.get("steps", defaults.get("steps", settings.default_num_inference_steps))),
            fps=int(case.get("fps", defaults.get("fps", settings.default_fps))),
            seed=seed,
            duration_seconds=case.get("duration_seconds", defaults.get("duration_seconds")),
            quality_profile=case.get("quality_profile", defaults.get("quality_profile", settings.quality_profile)),
            subject=case.get("subject", ""),
            action=case.get("action", ""),
            camera_motion=case.get("camera_motion", ""),
            shot_type=case.get("shot_type", ""),
            lighting=case.get("lighting", ""),
            mood=case.get("mood", ""),
            negative_prompt=case.get("negative_prompt", ""),
            output_long_edge=int(
                case.get("output_long_edge", defaults.get("output_long_edge", settings.default_output_long_edge))
            ),
            enable_deflicker=bool(
                case.get("enable_deflicker", defaults.get("enable_deflicker", settings.enable_deflicker))
            ),
            enable_sharpen=bool(
                case.get("enable_sharpen", defaults.get("enable_sharpen", settings.enable_sharpen))
            ),
        )

        name = _slugify(case.get("name", f"case_{idx}"))
        output_name = f"{name}_{idx}.mp4"
        final_path = output_root / output_name
        shutil.move(str(result.output_path), final_path)

        results.append(
            {
                "name": case.get("name", name),
                "prompt": prompt,
                "seed": result.seed,
                "num_frames": result.num_frames,
                "fps": result.fps,
                "steps": result.num_inference_steps,
                "guidance_scale": result.guidance_scale,
                "used_resolution": result.used_resolution,
                "output_resolution": result.output_resolution,
                "duration_seconds": result.duration_seconds,
                "output_path": str(final_path),
            }
        )

    metadata_path = output_root / "benchmark_results.json"
    metadata_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Saved benchmark results to {metadata_path}")


if __name__ == "__main__":
    main()
