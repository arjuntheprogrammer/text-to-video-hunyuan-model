import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        def _bool_env(name: str, default: str) -> bool:
            return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

        def _optional_int_env(name: str) -> int | None:
            raw = os.getenv(name)
            if raw is None:
                return None
            value = raw.strip().lower()
            if not value or value == "auto":
                return None
            return int(value)

        def _optional_float_env(name: str) -> float | None:
            raw = os.getenv(name)
            if raw is None:
                return None
            value = raw.strip().lower()
            if not value:
                return None
            return float(value)

        def _parse_profile_map(name: str, default_map: dict[str, int]) -> dict[str, int]:
            raw = os.getenv(name, "").strip()
            if not raw:
                return dict(default_map)
            parsed: dict[str, int] = {}
            for chunk in raw.split(","):
                item = chunk.strip()
                if not item or ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not key or not value:
                    continue
                try:
                    parsed[key] = int(value)
                except ValueError:
                    continue
            merged = dict(default_map)
            merged.update(parsed)
            return merged

        self.base_dir = Path(__file__).resolve().parents[1]
        self.model_id = os.getenv("MODEL_ID", "hunyuanvideo-community/HunyuanVideo-I2V")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        self.models_dir = Path(os.getenv("HF_HOME", self.base_dir / "models"))
        self.outputs_dir = Path(os.getenv("OUTPUT_DIR", self.base_dir / "outputs"))
        self.logs_dir = Path(os.getenv("LOG_DIR", self.base_dir / "logs"))
        self.app_log_file = Path(os.getenv("APP_LOG", self.logs_dir / "hunyuan_app.log"))

        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.gradio_host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        self.gradio_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

        self.default_fps = int(os.getenv("DEFAULT_FPS", "24"))
        self.default_duration_seconds = int(os.getenv("DEFAULT_DURATION_SECONDS", "6"))
        self.default_num_frames = int(
            os.getenv("DEFAULT_NUM_FRAMES", str(self.default_fps * self.default_duration_seconds))
        )

        self.min_num_frames = int(os.getenv("MIN_NUM_FRAMES", "16"))
        self.max_num_frames = int(os.getenv("MAX_NUM_FRAMES", "320"))

        self.default_guidance_scale = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5"))
        self.default_num_inference_steps = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "45"))

        self.min_fps = int(os.getenv("MIN_FPS", "8"))
        self.max_fps = int(os.getenv("MAX_FPS", "24"))

        self.min_guidance_scale = float(os.getenv("MIN_GUIDANCE_SCALE", "1.0"))
        self.max_guidance_scale = float(os.getenv("MAX_GUIDANCE_SCALE", "12.0"))

        self.min_inference_steps = int(os.getenv("MIN_INFERENCE_STEPS", "10"))
        self.max_inference_steps = int(os.getenv("MAX_INFERENCE_STEPS", "80"))

        self.gradio_concurrency_limit = int(os.getenv("GRADIO_CONCURRENCY_LIMIT", "1"))
        self.gradio_max_queue_size = int(os.getenv("GRADIO_MAX_QUEUE_SIZE", "16"))
        self.progress_log_every_steps = int(os.getenv("PROGRESS_LOG_EVERY_STEPS", "1"))
        self.progress_bar_width = int(os.getenv("PROGRESS_BAR_WIDTH", "24"))
        self.auto_max_input_side = _bool_env("AUTO_MAX_INPUT_SIDE", "1")
        max_input_override = _optional_int_env("MAX_INPUT_IMAGE_SIDE")
        self.max_input_image_side_override = max_input_override is not None
        self.max_input_image_side = max_input_override or 1024
        self.oom_safe_num_frames = int(os.getenv("OOM_SAFE_NUM_FRAMES", "32"))
        self.oom_safe_steps = int(os.getenv("OOM_SAFE_STEPS", "12"))
        self.default_prompt_suffix = os.getenv(
            "DEFAULT_PROMPT_SUFFIX",
            (
                "Keep motion physically plausible and temporally consistent with realistic inertia and continuity. "
                "Preserve subject identity, body proportions, and scene geometry. "
                "Use natural lighting, realistic textures, stable exposure, and cinematic camera behavior. "
                "Keep a single continuous shot with no scene cuts."
            ),
        ).strip()
        self.default_negative_prompt = os.getenv(
            "DEFAULT_NEGATIVE_PROMPT",
            (
                "flicker, frame-to-frame inconsistency, random fade, ghosting, jitter, morphing face, "
                "deformed anatomy, warped limbs, duplicate body parts, abrupt camera jumps, unrealistic motion, "
                "cartoon look, over-smoothing, over-sharpening, artifacts, text, watermark, logo, scene cuts"
            ),
        ).strip()
        self.enable_sequential_cpu_offload = _bool_env("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "0")
        self.enable_model_cpu_offload = _bool_env("ENABLE_MODEL_CPU_OFFLOAD", "1")
        self.enable_xformers = _bool_env("ENABLE_XFORMERS", "0")
        self.local_files_only = _bool_env("LOCAL_FILES_ONLY", "1")
        self.allow_remote_fallback = _bool_env("ALLOW_REMOTE_FALLBACK", "1")
        self.enable_deflicker = _bool_env("ENABLE_DEFLICKER", "1")
        self.deflicker_window = int(os.getenv("DEFLICKER_WINDOW", "3"))
        if self.deflicker_window < 1:
            self.deflicker_window = 1
        if self.deflicker_window % 2 == 0:
            self.deflicker_window += 1
        self.max_prompt_words = int(os.getenv("MAX_PROMPT_WORDS", "60"))
        self.max_negative_prompt_words = int(os.getenv("MAX_NEGATIVE_PROMPT_WORDS", "60"))
        if self.max_prompt_words < 1:
            self.max_prompt_words = 1
        if self.max_negative_prompt_words < 1:
            self.max_negative_prompt_words = 1

        self.quality_profile = os.getenv("QUALITY_PROFILE", "balanced").strip().lower()
        self.max_frames_by_profile = _parse_profile_map(
            "MAX_FRAMES_BY_PROFILE",
            {"low": 160, "balanced": 320, "high": 320},
        )
        self.max_steps_by_profile = _parse_profile_map(
            "MAX_STEPS_BY_PROFILE",
            {"low": 20, "balanced": 28, "high": 32},
        )
        self.max_input_side_by_profile = _parse_profile_map(
            "MAX_INPUT_SIDE_BY_PROFILE",
            {"low": 768, "balanced": 1024, "high": 1280},
        )

        self.default_output_long_edge = int(os.getenv("DEFAULT_OUTPUT_LONG_EDGE", "1080"))
        self.output_long_edge_options = [
            int(value.strip())
            for value in os.getenv("OUTPUT_LONG_EDGE_OPTIONS", "720,1080,1440").split(",")
            if value.strip()
        ]
        if self.default_output_long_edge not in self.output_long_edge_options:
            self.output_long_edge_options.append(self.default_output_long_edge)
        self.output_long_edge_options = sorted(set(self.output_long_edge_options))

        self.guidance_rescale = _optional_float_env("GUIDANCE_RESCALE")
        self.image_guidance_scale = _optional_float_env("IMAGE_GUIDANCE_SCALE")
        self.strength = _optional_float_env("STRENGTH")

        self.enable_captioning = _bool_env("ENABLE_CAPTIONING", "1")
        self.caption_model_id = os.getenv("CAPTION_MODEL_ID", "Salesforce/blip2-opt-2.7b")
        self.caption_device = os.getenv("CAPTION_DEVICE", "cpu")
        self.caption_max_tokens = int(os.getenv("CAPTION_MAX_TOKENS", "40"))


settings = Settings()
