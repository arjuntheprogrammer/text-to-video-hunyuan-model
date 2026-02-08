import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
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

        self.default_fps = int(os.getenv("DEFAULT_FPS", "16"))
        self.default_duration_seconds = int(os.getenv("DEFAULT_DURATION_SECONDS", "10"))
        self.default_num_frames = int(
            os.getenv("DEFAULT_NUM_FRAMES", str(self.default_fps * self.default_duration_seconds))
        )

        self.min_num_frames = int(os.getenv("MIN_NUM_FRAMES", "16"))
        self.max_num_frames = int(os.getenv("MAX_NUM_FRAMES", "320"))

        self.default_guidance_scale = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "6.0"))
        self.default_num_inference_steps = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "30"))

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
        self.max_input_image_side = int(os.getenv("MAX_INPUT_IMAGE_SIDE", "1024"))
        self.oom_safe_num_frames = int(os.getenv("OOM_SAFE_NUM_FRAMES", "32"))
        self.oom_safe_steps = int(os.getenv("OOM_SAFE_STEPS", "12"))
        self.default_prompt_suffix = os.getenv(
            "DEFAULT_PROMPT_SUFFIX",
            (
                "Keep motion physically plausible and temporally consistent with realistic inertia and continuity. "
                "Preserve subject identity, body proportions, and scene geometry. "
                "Use natural lighting, realistic textures, stable exposure, and cinematic camera behavior."
            ),
        ).strip()
        self.default_negative_prompt = os.getenv(
            "DEFAULT_NEGATIVE_PROMPT",
            (
                "flicker, frame-to-frame inconsistency, random fade, ghosting, jitter, morphing face, "
                "deformed anatomy, warped limbs, duplicate body parts, abrupt camera jumps, unrealistic motion, "
                "cartoon look, over-smoothing, over-sharpening, artifacts, text, watermark, logo"
            ),
        ).strip()
        self.enable_sequential_cpu_offload = os.getenv("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_model_cpu_offload = os.getenv("ENABLE_MODEL_CPU_OFFLOAD", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_xformers = os.getenv("ENABLE_XFORMERS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


settings = Settings()
