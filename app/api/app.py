import logging
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.services.pipeline_manager import get_pipeline_manager
from app.api.schemas import GenerateResponse, HealthResponse
from app.utils.common import ensure_directories, sanitize_filename

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="HunyuanVideo-I2V API",
    description="Image-to-video generation using hunyuanvideo-community/HunyuanVideo-I2V",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    ensure_directories()
    LOGGER.info("FastAPI startup complete.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    manager = get_pipeline_manager()
    status = "ok" if manager.model_loaded else "error"
    return HealthResponse(
        status=status,
        device=manager.device,
        model_loaded=manager.model_loaded,
        model_id=settings.model_id,
        error=manager.load_error,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=3, max_length=2000),
    subject: str = Form(default=""),
    action: str = Form(default=""),
    camera_motion: str = Form(default=""),
    shot_type: str = Form(default=""),
    lighting: str = Form(default=""),
    mood: str = Form(default=""),
    negative_prompt: str = Form(default=settings.default_negative_prompt),
    num_frames: int = Form(settings.default_num_frames),
    guidance_scale: float = Form(settings.default_guidance_scale),
    steps: int = Form(settings.default_num_inference_steps),
    fps: int = Form(settings.default_fps),
    duration_seconds: float | None = Form(default=None),
    quality_profile: str = Form(default=settings.quality_profile),
    output_long_edge: int = Form(settings.default_output_long_edge),
    enable_deflicker: bool = Form(settings.enable_deflicker),
    enable_sharpen: bool = Form(settings.enable_sharpen),
    seed: int | None = Form(default=None),
) -> GenerateResponse:
    LOGGER.info(
        "API /generate request received. filename=%s prompt_len=%d num_frames=%s steps=%s fps=%s guidance_scale=%s seed=%s output_long_edge=%s deflicker=%s sharpen=%s profile=%s duration=%s",
        image.filename,
        len(prompt),
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

    if not manager.model_loaded:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {manager.load_error}")

    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is missing.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    LOGGER.info(
        "Uploaded image received. filename=%s bytes=%d",
        image.filename,
        len(content),
    )

    try:
        pil_image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    try:
        result = manager.generate_video(
            image=pil_image,
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            fps=fps,
            seed=seed,
            duration_seconds=duration_seconds,
            quality_profile=quality_profile,
            image_source=f"upload:{image.filename}",
            subject=subject,
            action=action,
            camera_motion=camera_motion,
            shot_type=shot_type,
            lighting=lighting,
            mood=mood,
            negative_prompt=negative_prompt,
            output_long_edge=output_long_edge,
            enable_deflicker=enable_deflicker,
            enable_sharpen=enable_sharpen,
        )
    except ValueError as exc:
        LOGGER.warning("Generation rejected by validation: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Generation failed with unhandled exception")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    LOGGER.info(
        "API /generate completed. output=%s seed=%s frames=%s fps=%s",
        result.output_path.name,
        result.seed,
        result.num_frames,
        result.fps,
    )
    return GenerateResponse(
        filename=result.output_path.name,
        output_path=str(result.output_path),
        output_url=f"/outputs/{result.output_path.name}",
        fps=result.fps,
        num_frames=result.num_frames,
        seed=result.seed,
        used_steps=result.num_inference_steps,
        used_guidance_scale=result.guidance_scale,
        used_resolution_width=result.used_resolution[0],
        used_resolution_height=result.used_resolution[1],
        output_resolution_width=result.output_resolution[0],
        output_resolution_height=result.output_resolution[1],
        effective_prompt_len=result.effective_prompt_len,
        negative_prompt_len=result.negative_prompt_len,
        duration_seconds=result.duration_seconds,
    )


@app.get("/outputs/{filename}")
def get_output(filename: str) -> FileResponse:
    safe_name = sanitize_filename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = settings.outputs_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path=file_path, media_type="video/mp4", filename=file_path.name)
