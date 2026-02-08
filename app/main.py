import logging
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.pipeline import get_pipeline_manager
from app.schemas import GenerateResponse, HealthResponse
from app.utils import ensure_directories, sanitize_filename

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
    num_frames: int = Form(settings.default_num_frames),
    guidance_scale: float = Form(settings.default_guidance_scale),
    steps: int = Form(settings.default_num_inference_steps),
    fps: int = Form(settings.default_fps),
    seed: int | None = Form(default=None),
) -> GenerateResponse:
    LOGGER.info(
        "API /generate request received. filename=%s prompt_len=%d num_frames=%s steps=%s fps=%s guidance_scale=%s seed=%s",
        image.filename,
        len(prompt),
        num_frames,
        steps,
        fps,
        guidance_scale,
        seed,
    )
    manager = get_pipeline_manager()

    if not manager.model_loaded:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {manager.load_error}")

    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is missing.")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        pil_image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    try:
        output_path, used_seed, used_num_frames, used_fps = manager.generate_video(
            image=pil_image,
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            fps=fps,
            seed=seed,
        )
    except ValueError as exc:
        LOGGER.warning("Generation rejected by validation: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Generation failed with unhandled exception")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    LOGGER.info(
        "API /generate completed. output=%s seed=%s frames=%s fps=%s",
        output_path.name,
        used_seed,
        used_num_frames,
        used_fps,
    )
    return GenerateResponse(
        filename=output_path.name,
        output_path=str(output_path),
        output_url=f"/outputs/{output_path.name}",
        fps=used_fps,
        num_frames=used_num_frames,
        seed=used_seed,
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
