# HunyuanVideo Production Starter Project

End-to-end image-to-video generation service powered by the open-source `hunyuanvideo-community/HunyuanVideo-I2V` model. The project includes:
- FastAPI REST API
- Public Gradio UI
- Dockerized deployment for Ubuntu 22.04 with NVIDIA GPU

## Requirements
- Ubuntu 22.04 host
- NVIDIA GPU (L40s recommended)
- NVIDIA driver installed
- NVIDIA Container Toolkit installed
- Docker Engine + Docker Compose v2

## NVIDIA Container Toolkit Setup
Install toolkit and restart Docker:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run

```bash
docker compose up --build
```

## URLs
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Gradio UI (public bind): `http://localhost:7860`

## API Example

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "image=@/absolute/path/input.png" \
  -F "prompt=cinematic camera dolly with gentle subject motion" \
  -F "num_frames=160" \
  -F "guidance_scale=6.0" \
  -F "steps=30" \
  -F "fps=16" \
  -F "seed=42"
```

Download generated file from the returned `output_url`:

```bash
curl -O "http://localhost:8000/outputs/<filename>.mp4"
```

## Persistence
- `./models` stores Hugging Face model cache and weights.
- `./outputs` stores generated MP4 files.

Both are mounted into the container and persist across restarts.

## L40s Performance Notes
- FP16 inference on CUDA.
- TF32 enabled for faster matrix operations.
- Attention/tiling optimizations enabled.
- Conservative defaults: 160 frames, 16 FPS, 30 steps.
- Max frame cap: 320 frames (~20 seconds at 16 FPS).

## Optional Model Pre-download

```bash
python scripts/download_model.py --cache-dir ./models
```
