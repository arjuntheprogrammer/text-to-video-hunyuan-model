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

## Project Flow (Mermaid)

```mermaid
flowchart TB
    %% High-level project flow for HunyuanVideo-I2V
    classDef entry fill:#0b1f3b,stroke:#4fa3ff,stroke-width:1px,color:#e7f1ff
    classDef compute fill:#0f2f1f,stroke:#38c172,stroke-width:1px,color:#e6fff2
    classDef io fill:#3b1b0f,stroke:#ff9f43,stroke-width:1px,color:#fff2e6
    classDef storage fill:#2b1b3b,stroke:#b07cff,stroke-width:1px,color:#f3e6ff
    classDef util fill:#1f2b2b,stroke:#9bd1d1,stroke-width:1px,color:#ecffff

    subgraph Setup["Setup & Environment"]
        Env[".env / env vars"]:::util
        Conda["setup/setup.sh (conda-native)"]:::entry
        Docker["docker-compose.yml (optional)"]:::entry
        Conda --> Env
        Docker --> Env
    end

    subgraph Bootstrap["Runtime Bootstrap (run.py)"]
        Run["run.py"]:::entry
        Log["logging_utils.configure_logging"]:::util
        Dirs["utils.ensure_directories"]:::util
        Uvicorn["uvicorn server"]:::entry
        GradioLaunch["gradio_ui.launch_gradio_app"]:::entry
        Run --> Log
        Run --> Dirs
        Run --> Uvicorn
        Run --> GradioLaunch
    end

    subgraph API["FastAPI (app/main.py)"]
        Health["GET /health"]:::io
        Generate["POST /generate"]:::io
        Outputs["GET /outputs/{filename}"]:::io
        Health --> ManagerStatus["pipeline manager status"]:::compute
        Generate --> Validate["validate input + decode image"]:::util
        Outputs --> FileResp["FileResponse (mp4)"]:::io
    end

    subgraph UI["Gradio UI (app/gradio_ui.py)"]
        UIForm["Blocks UI + prompt fields"]:::io
        UIGen["_generate_from_ui"]:::io
        UIForm --> UIGen
    end

    subgraph Pipeline["Pipeline Core (app/pipeline.py)"]
        Manager["HunyuanVideoPipelineManager"]:::compute
        Init["configure runtime + load model"]:::compute
        Caption["captioning.caption_image (optional)"]:::compute
        Prompt["prompt_builder.build_structured_prompt"]:::compute
        Truncate["token + word truncation"]:::util
        OOM["OOM-safe retry ladder"]:::compute
        Diffusers["diffusers HunyuanVideo pipeline"]:::compute
        Frames["frames[]"]:::compute
        SaveMp4["video_utils.save_frames_to_mp4"]:::compute
        Manager --> Init
        Caption --> Prompt
        Prompt --> Truncate
        Truncate --> OOM
        OOM --> Diffusers
        Diffusers --> Frames
        Frames --> SaveMp4
    end

    subgraph Captioning["Captioning (app/captioning.py)"]
        BLIP["BLIP-2 model (optional)"]:::compute
    end

    subgraph VideoUtils["Video Encode (app/video_utils.py)"]
        Target["compute_target_size"]:::compute
        Normalize["normalize + deflicker"]:::compute
        Resize["resize/pad to target"]:::compute
        Encode["imageio -> mp4"]:::compute
        Target --> Normalize --> Resize --> Encode
    end

    subgraph Storage["Persistence"]
        Models["models/ (HF cache)"]:::storage
        OutputsDir["outputs/ (mp4 files)"]:::storage
        Logs["logs/ (hunyuan_app.log)"]:::storage
    end

    subgraph Scripts["Scripts & Tests"]
        Bench["scripts/benchmark_prompts.py"]:::util
        Samples["tests/samples/*"]:::util
        Tests["tests/test_*"]:::util
        Bench --> Samples
    end

    %% Cross-subgraph flow
    Env --> Run
    Uvicorn --> Health
    Uvicorn --> Generate
    GradioLaunch --> UIForm
    UIGen --> Manager
    Validate --> Manager
    Manager --> Caption
    Caption --> BLIP
    Init --> Models
    SaveMp4 --> Target
    Encode --> OutputsDir
    OutputsDir --> FileResp
    Log --> Logs
    Dirs --> Models
    Dirs --> OutputsDir
    Dirs --> Logs
    SaveMp4 --> Response["GenerateResponse (output_url + metadata)"]:::io
    Generate --> Response
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
  -F "subject=person" \
  -F "action=turning head" \
  -F "camera_motion=dolly in" \
  -F "shot_type=close-up" \
  -F "lighting=soft daylight" \
  -F "mood=cinematic" \
  -F "negative_prompt=flicker, jitter, deformed anatomy" \
  -F "num_frames=160" \
  -F "guidance_scale=6.0" \
  -F "steps=30" \
  -F "fps=16" \
  -F "output_long_edge=1080" \
  -F "enable_deflicker=true" \
  -F "seed=42"
```

Download generated file from `output_url`:

```bash
curl -O "http://localhost:8000/outputs/<filename>.mp4"
```

## Structured Prompt Fields (Gradio + API)

| Field | Options |
| --- | --- |
| subject | person, product, food, fashion, animal, vehicle, architecture, landscape, cityscape, gadget |
| action | walking, turning head, smiling, hand gesture, hair movement, pouring, rotating, hovering, panning reveal, still |
| camera_motion | static, slow pan, tilt, dolly in, dolly out, orbit, handheld, zoom in, zoom out |
| shot_type | close-up, medium, wide, macro, overhead, low angle, high angle |
| lighting | soft daylight, golden hour, studio softbox, neon, backlit, overcast, candlelight |
| mood | cinematic, calm, energetic, moody, dreamy, documentary, romantic, dramatic |

Output long-edge presets: `720`, `1080`, `1440` (aspect ratio derived from the input image).

## Important Runtime Notes

- `transformers` is pinned to `<5.0.0` for model compatibility.
- xFormers attention is opt-in. Set `ENABLE_XFORMERS=1` in `.env` to enable.
- Setup pins all model/cache paths to `./models` (`HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`).
- OOM safety defaults:
  - input images are auto-resized based on GPU VRAM when `AUTO_MAX_INPUT_SIDE=1`
  - set `MAX_INPUT_IMAGE_SIDE` to a number to override auto sizing
  - conservative fallback profile uses `OOM_SAFE_NUM_FRAMES=32` and `OOM_SAFE_STEPS=12`
  - retry order now starts with user-requested `frames/steps` at highest allowed resolution, and downgrades only after OOM
- Output video is resized to the selected long-edge preset while preserving input aspect ratio.
- Prompt enhancement defaults:
  - builds a structured prompt from dropdown fields + user text
  - optionally prepends a BLIP-2 caption for better identity anchoring
  - appends an internal realism suffix to the user prompt
  - applies a default negative prompt to reduce flicker/fade/morph artifacts (can be overridden)
  - enforces word limits to avoid CLIP token overflow (override via `MAX_PROMPT_WORDS`, `MAX_NEGATIVE_PROMPT_WORDS`)
- Deflicker post-processing is disabled by default; toggle via `ENABLE_DEFLICKER` or the API parameter.
- Optional sharpening can be enabled via `ENABLE_SHARPEN` to recover crisp motion.
- Model loading is local-only by default. If local files are missing, `ALLOW_REMOTE_FALLBACK=1` lets it download as a fallback.
- CPU offload defaults:
  - `ENABLE_SEQUENTIAL_CPU_OFFLOAD=0`
  - `ENABLE_MODEL_CPU_OFFLOAD=1`
- Gradio UI uses duration + FPS to auto-compute frames (`frames = duration × fps`); frame count is capped by quality profile at runtime.
- Progress logs in `logs/hunyuan_app.log`:
  - `PROGRESS_LOG_EVERY_STEPS=1`
  - `PROGRESS_BAR_WIDTH=24`
- First startup may take a long time due to model download.
- Quality profiles cap max frames/steps/input side for stability (`low`, `balanced`, `high`). Set `QUALITY_PROFILE` and override caps via `MAX_FRAMES_BY_PROFILE`, `MAX_STEPS_BY_PROFILE`, `MAX_INPUT_SIDE_BY_PROFILE`.

## Persistence
- `./models` stores model/cache files
- `./outputs` stores generated MP4 files
- `./logs` stores runtime logs (`hunyuan_app.log`, health snapshot, pid file)

Both persist across restarts in the same filesystem.
`hunyuan_app.log` is append-only by default in `setup/setup.sh`, so previous runs remain in the same file.

## Setup Folder

See:
- `setup/setup.sh` for full automated setup
- `setup/setup.md` for usage, overrides, and troubleshooting
- `setup/vscode-extensions.txt` for auto-install extension list

## Benchmark Script

Run the bundled prompt benchmark set:

```bash
python scripts/benchmark_prompts.py --json scripts/benchmark_prompts.json
```

Run against sample folders (each subfolder with `img.*` and `prompt.txt`):

```bash
python scripts/benchmark_prompts.py --samples-dir tests/samples
```

Outputs are written to `outputs/benchmarks/<timestamp>/` along with `benchmark_results.json`.
