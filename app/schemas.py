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
