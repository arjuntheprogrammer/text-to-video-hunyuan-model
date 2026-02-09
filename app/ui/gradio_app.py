import logging
from pathlib import Path

import gradio as gr
from PIL import Image

from app.core.config import settings
from app.services.pipeline_manager import get_pipeline_manager
from app.services.prompt_builder import PROMPT_FIELD_OPTIONS
from app.utils.common import ensure_directories

LOGGER = logging.getLogger(__name__)


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
    num_frames: int,
    steps: int,
    guidance_scale: float,
    fps: int,
    output_long_edge: int,
    enable_deflicker: bool,
    seed: int,
) -> tuple[str | None, str]:
    LOGGER.info(
        "Gradio generation request received. prompt_len=%d num_frames=%s steps=%s fps=%s guidance_scale=%s seed=%s output_long_edge=%s deflicker=%s",
        len(prompt or ""),
        num_frames,
        steps,
        fps,
        guidance_scale,
        seed,
        output_long_edge,
        enable_deflicker,
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
            subject=subject,
            action=action,
            camera_motion=camera_motion,
            shot_type=shot_type,
            lighting=lighting,
            mood=mood,
            negative_prompt=negative_prompt,
            output_long_edge=int(output_long_edge),
            enable_deflicker=bool(enable_deflicker),
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

    with gr.Blocks(title="HunyuanVideo-I2V") as demo:
        gr.Markdown(
            "# HunyuanVideo-I2V\n"
            "Generate videos from a starting image and text prompt.\n"
            "Input image aspect ratio is preserved in output."
        )
        gr.Markdown("Tip: Use a single action, continuous shot, no scene cuts.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Starting Image")
                subject_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["subject"],
                    label="Subject",
                    allow_custom_value=True,
                    value="",
                )
                action_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["action"],
                    label="Action",
                    allow_custom_value=True,
                    value="",
                )
                camera_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["camera_motion"],
                    label="Camera Motion",
                    allow_custom_value=True,
                    value="",
                )
                shot_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["shot_type"],
                    label="Shot Type",
                    allow_custom_value=True,
                    value="",
                )
                lighting_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["lighting"],
                    label="Lighting",
                    allow_custom_value=True,
                    value="",
                )
                mood_input = gr.Dropdown(
                    choices=PROMPT_FIELD_OPTIONS["mood"],
                    label="Mood",
                    allow_custom_value=True,
                    value="",
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Describe motion, animation, and scene changes...",
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    value=settings.default_negative_prompt,
                    lines=3,
                )
                output_long_edge_input = gr.Dropdown(
                    choices=settings.output_long_edge_options,
                    value=settings.default_output_long_edge,
                    label="Output Long Edge (px)",
                )
                deflicker_input = gr.Checkbox(
                    value=settings.enable_deflicker,
                    label="Enable Deflicker",
                )
                frames_input = gr.Number(
                    minimum=settings.min_num_frames,
                    value=settings.default_num_frames,
                    precision=0,
                    label="Frames (duration ~= frames / FPS)",
                )
                steps_input = gr.Slider(
                    minimum=settings.min_inference_steps,
                    maximum=settings.max_inference_steps,
                    value=settings.default_num_inference_steps,
                    step=1,
                    label="Inference Steps",
                )
                guidance_input = gr.Slider(
                    minimum=settings.min_guidance_scale,
                    maximum=settings.max_guidance_scale,
                    value=settings.default_guidance_scale,
                    step=0.1,
                    label="Guidance Scale",
                )
                fps_input = gr.Slider(
                    minimum=settings.min_fps,
                    maximum=settings.max_fps,
                    value=settings.default_fps,
                    step=1,
                    label="FPS",
                )
                seed_input = gr.Number(
                    label="Seed (-1 for random)",
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
                frames_input,
                steps_input,
                guidance_input,
                fps_input,
                output_long_edge_input,
                deflicker_input,
                seed_input,
            ],
            outputs=[video_output, status_output],
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
