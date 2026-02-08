# HunyuanVideo Production Starter Project

End-to-end image-to-video generation service powered by `hunyuanvideo-community/HunyuanVideo-I2V`.

Includes:
- FastAPI REST API
- Gradio UI
- Conda-native runtime (recommended for remote GPU containers)
- Optional Docker runtime (only when host Docker access is available)

## Runtime Modes

### 1) Conda-native (recommended)

Use this when you do not have host-level Docker access (common on rented GPU/container instances).

```bash
cd /home/ubuntu/text-to-video-hunyuan-model
chmod +x setup/setup.sh
HF_TOKEN=hf_xxx ./setup/setup.sh
```

If `.env` already has a valid `HF_TOKEN`, this also works:

```bash
cd /home/ubuntu/text-to-video-hunyuan-model
./setup/setup.sh
```

What it does:
- Installs missing OS packages
- Installs/configures Conda (`/opt/conda`)
- Creates env `hunyuanvideo`
- Installs Python dependencies
- Configures `.env` paths
- Starts `python run.py`
- Waits for `/health`

Optional full E2E test during setup:

```bash
RUN_GENERATE_TEST=1 ./setup/setup.sh
```

### 2) Docker (optional, host access required)

Only use this if you control the host daemon and NVIDIA container runtime.

```bash
cp .env_template .env
# set HF_TOKEN in .env
docker compose up --build
```

## Requirements

### Conda-native
- Linux x86_64
- NVIDIA GPU + working driver (`nvidia-smi`)
- Internet access for model/dependency download
- Hugging Face token with access to the model

### Docker mode
- Ubuntu host with Docker Engine + Compose v2
- NVIDIA Container Toolkit configured for Docker

## URLs
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Gradio UI: `http://localhost:7860`

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

Download generated file from `output_url`:

```bash
curl -O "http://localhost:8000/outputs/<filename>.mp4"
```

## Important Runtime Notes

- `transformers` is pinned to `<5.0.0` for model compatibility.
- xFormers attention is opt-in. Set `ENABLE_XFORMERS=1` in `.env` to enable.
- First startup may take a long time due to model download.

## Persistence
- `./models` stores model/cache files
- `./outputs` stores generated MP4 files

Both persist across restarts in the same filesystem.

## Setup Folder

See:
- `setup/setup.sh` for full automated setup
- `setup/setup.md` for usage, overrides, and troubleshooting
- `setup/vscode-extensions.txt` for auto-install extension list
