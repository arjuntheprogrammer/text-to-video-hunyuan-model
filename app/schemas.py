from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_id: str
    error: str | None = None


class GenerateResponse(BaseModel):
    status: str = Field(default="ok")
    filename: str
    output_path: str
    output_url: str
    fps: int
    num_frames: int
    seed: int
    used_steps: int
    used_guidance_scale: float
    used_resolution_width: int
    used_resolution_height: int
    output_resolution_width: int
    output_resolution_height: int
    effective_prompt_len: int
    negative_prompt_len: int
    duration_seconds: float
