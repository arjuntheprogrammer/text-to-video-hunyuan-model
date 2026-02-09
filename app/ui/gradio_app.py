import logging
from pathlib import Path

import gradio as gr
from PIL import Image

from app.core.config import settings
from app.services.pipeline_manager import get_pipeline_manager
from app.services.prompt_builder import PROMPT_FIELD_OPTIONS
from app.utils.common import ensure_directories

LOGGER = logging.getLogger(__name__)


def _compute_frames(duration_seconds: float, fps: int) -> int:
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = float(settings.default_duration_seconds)
    try:
        fps_value = int(fps)
    except (TypeError, ValueError):
        fps_value = settings.default_fps
    fps_value = max(settings.min_fps, min(settings.max_fps, fps_value))
    frames = max(settings.min_num_frames, int(round(duration * fps_value)))
    return frames


def _generate_from_ui(
    image: Image.Image,
    prompt: str,
    subject: str,
    action: str,
    camera_motion: str,
    shot_type: str,
    lighting: str,
    mood: str,
    negative_prompt: str,
    duration_seconds: float,
    fps: int,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    output_long_edge: int,
    enable_deflicker: bool,
    enable_sharpen: bool,
    quality_profile: str,
    seed: int,
) -> tuple[str | None, str]:
    LOGGER.info(
        "Gradio generation request received. prompt_len=%d num_frames=%s steps=%s fps=%s guidance_scale=%s seed=%s output_long_edge=%s deflicker=%s sharpen=%s profile=%s duration=%s",
        len(prompt or ""),
        num_frames,
        steps,
        fps,
        guidance_scale,
        seed,
        output_long_edge,
        enable_deflicker,
        enable_sharpen,
        quality_profile,
        duration_seconds,
    )
    manager = get_pipeline_manager()

    if image is None:
        return None, "Please upload an input image."
    if not prompt or len(prompt.strip()) < 3:
        return None, "Prompt must be at least 3 characters."
    if not manager.model_loaded:
        return None, f"Model unavailable: {manager.load_error}"

    use_seed = seed if seed >= 0 else None

    try:
        result = manager.generate_video(
            image=image,
            prompt=prompt.strip(),
            num_frames=int(num_frames),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            fps=int(fps),
            seed=use_seed,
            duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
            quality_profile=quality_profile,
            image_source="gradio_upload",
            subject=subject,
            action=action,
            camera_motion=camera_motion,
            shot_type=shot_type,
            lighting=lighting,
            mood=mood,
            negative_prompt=negative_prompt,
            output_long_edge=int(output_long_edge),
            enable_deflicker=bool(enable_deflicker),
            enable_sharpen=bool(enable_sharpen),
        )
    except Exception as exc:
        LOGGER.exception("Gradio generation failed")
        return None, f"Generation failed: {exc}"

    LOGGER.info(
        "Gradio generation completed. output=%s seed=%s frames=%s fps=%s",
        Path(result.output_path).name,
        result.seed,
        result.num_frames,
        result.fps,
    )
    return (
        str(result.output_path),
        f"Generated {Path(result.output_path).name} (seed={result.seed}, frames={result.num_frames}, fps={result.fps})",
    )


def build_gradio_app() -> gr.Blocks:
    ensure_directories()
    profile_default = (
        settings.quality_profile if settings.quality_profile in {"low", "balanced", "high"} else "balanced"
    )

    with gr.Blocks(title="HunyuanVideo-I2V") as demo:
        gr.Markdown(
            "# HunyuanVideo-I2V\n"
            "Generate videos from a starting image and text prompt.\n"
            "Input image aspect ratio is preserved in output."
        )
        gr.Markdown("Tip: Use a single action, continuous shot, no scene cuts.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Starting Image (aspect preserved)")
                prompt_input = gr.Textbox(
                    label="Prompt (describe the motion, action, and scene)",
                    lines=4,
                    placeholder="Describe motion, animation, and scene changes...",
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (things to avoid)",
                    value=settings.default_negative_prompt,
                    lines=3,
                )
                subject_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["subject"],
                    label="Subject (main focus)",
                    allow_custom_value=True,
                    value="person",
                )
                action_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["action"],
                    label="Action (what happens)",
                    allow_custom_value=True,
                    value="still",
                )
                camera_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["camera_motion"],
                    label="Camera Motion (camera movement)",
                    allow_custom_value=True,
                    value="static",
                )
                shot_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["shot_type"],
                    label="Shot Type (framing)",
                    allow_custom_value=True,
                    value="medium",
                )
                lighting_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["lighting"],
                    label="Lighting (scene light)",
                    allow_custom_value=True,
                    value="soft daylight",
                )
                mood_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["mood"],
                    label="Mood (style/energy)",
                    allow_custom_value=True,
                    value="cinematic",
                )
                duration_input = gr.Number(
                    minimum=1,
                    value=settings.default_duration_seconds,
                    precision=2,
                    label="Duration (seconds)",
                )
                fps_input = gr.Slider(
                    minimum=settings.min_fps,
                    maximum=settings.max_fps,
                    value=settings.default_fps,
                    step=1,
                    label="FPS (smoothness)",
                )
                frames_input = gr.Number(
                    minimum=settings.min_num_frames,
                    value=settings.default_num_frames,
                    precision=0,
                    interactive=False,
                    label="Frames (auto = duration × FPS)",
                )
                with gr.Accordion("Advanced Settings", open=False):
                    steps_input = gr.Slider(
                        minimum=settings.min_inference_steps,
                        maximum=settings.max_inference_steps,
                        value=settings.default_num_inference_steps,
                        step=1,
                        label="Inference Steps (quality vs speed)",
                    )
                    guidance_input = gr.Slider(
                        minimum=settings.min_guidance_scale,
                        maximum=settings.max_guidance_scale,
                        value=settings.default_guidance_scale,
                        step=0.1,
                        label="Guidance Scale (prompt strength)",
                    )
                    output_long_edge_input = gr.Dropdown(
                        choices=settings.output_long_edge_options,
                        value=settings.default_output_long_edge,
                        label="Output Long Edge (px, aspect preserved)",
                    )
                    deflicker_input = gr.Checkbox(
                        value=settings.enable_deflicker,
                        label="Enable Deflicker (reduce flicker, may soften motion)",
                    )
                    sharpen_input = gr.Checkbox(
                        value=settings.enable_sharpen,
                        label="Enable Sharpen (restore crispness)",
                    )
                    quality_profile_input = gr.Dropdown(
                        choices=["low", "balanced", "high"],
                        value=profile_default,
                        label="Quality Profile (caps for stability)",
                    )
                    seed_input = gr.Number(
                        label="Seed (-1 for random, fixed for repeatability)",
                        value=-1,
                        precision=0,
                    )
                run_button = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video", format="mp4")
                status_output = gr.Textbox(label="Status", interactive=False)

        run_button.click(
            fn=_generate_from_ui,
            inputs=[
                image_input,
                prompt_input,
                subject_input,
                action_input,
                camera_input,
                shot_input,
                lighting_input,
                mood_input,
                negative_prompt_input,
                duration_input,
                fps_input,
                frames_input,
                steps_input,
                guidance_input,
                output_long_edge_input,
                deflicker_input,
                sharpen_input,
                quality_profile_input,
                seed_input,
            ],
            outputs=[video_output, status_output],
        )

        duration_input.change(
            fn=_compute_frames,
            inputs=[duration_input, fps_input],
            outputs=frames_input,
        )
        fps_input.change(
            fn=_compute_frames,
            inputs=[duration_input, fps_input],
            outputs=frames_input,
        )

    demo.queue(
        max_size=settings.gradio_max_queue_size,
        default_concurrency_limit=settings.gradio_concurrency_limit,
    )
    return demo


def launch_gradio_app(demo: gr.Blocks | None = None, prevent_thread_lock: bool = False) -> gr.Blocks:
    app = demo or build_gradio_app()
    app.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=False,
        prevent_thread_lock=prevent_thread_lock,
        allowed_paths=[str(settings.outputs_dir)],
        show_error=True,
        quiet=True,
    )
    return app
