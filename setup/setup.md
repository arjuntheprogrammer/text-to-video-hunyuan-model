# HunyuanVideo Conda Setup (No Docker)

This setup runs `text-to-video-hunyuan-model` directly on the remote GPU instance using Conda.

Docker is intentionally not used in this flow.

## Files

- Script: `setup/setup.sh`
- VS Code extension list: `setup/vscode-extensions.txt`
- Repo expected by default at: parent of this script (`<repo>/setup/setup.sh` -> repo root)

## What `setup.sh` does

The script is idempotent and non-interactive:

1. Checks and installs required OS dependencies if missing:
   - `wget`, `curl`, `ca-certificates`, `bzip2`, `zsh`, `ffmpeg`, `git`, `procps`, `libgl1`, `libglib2.0-0`
2. Sets global git identity (default):
   - `user.name=Arjun Gupta`
   - `user.email=arjuntheprogrammer@gmail.com`
3. Installs only missing VS Code extensions from `setup/vscode-extensions.txt` (if VS Code remote CLI is available).
4. Installs Miniconda to `/opt/conda` if not present.
5. Accepts Conda channel ToS non-interactively.
6. Initializes Conda for `bash` and `zsh`.
7. Creates Conda env `hunyuanvideo` if missing (`python=3.10`, `pip`, `ffmpeg`).
8. Installs Python dependencies from `requirements.txt`.
9. Ensures `.env` exists and has a valid `HF_TOKEN`.
10. Sets cache/runtime paths in `.env` to repo-local folders:
   - `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME` -> `<repo>/models`
   - `OUTPUT_DIR` -> `<repo>/outputs`
   - `LOG_DIR`, `APP_LOG` -> `<repo>/logs/...`
   - Progress log defaults:
     - `PROGRESS_LOG_EVERY_STEPS=1`
     - `PROGRESS_BAR_WIDTH=24`
   - OOM safety defaults:
     - `MAX_INPUT_IMAGE_SIDE=1024`
     - `OOM_SAFE_NUM_FRAMES=32`
     - `OOM_SAFE_STEPS=12`
     - `ENABLE_SEQUENTIAL_CPU_OFFLOAD=0`
     - `ENABLE_MODEL_CPU_OFFLOAD=1`
   - Removes deprecated `TRANSFORMERS_CACHE` entry if present
11. Starts app in background (`python run.py` in Conda env).
12. Waits for `http://127.0.0.1:8000/health` to become available.

Optional:

- If `RUN_GENERATE_TEST=1`, generates a 512x512 RGB PNG via `ffmpeg`, runs one real `/generate` request, and downloads output MP4.
  - Default test image path: `<repo>/setup/test_input_512.png` (reused if it already exists).
- `ENABLE_XFORMERS` defaults to disabled unless explicitly set (`1/true/yes/on`).

## Run

After cloning into `/home/ubuntu` (repo path `/home/ubuntu/text-to-video-hunyuan-model`), run:

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

## Useful environment overrides

```bash
REPO_DIR=/path/to/text-to-video-hunyuan-model \
ENV_NAME=hunyuanvideo \
CONDA_DIR=/opt/conda \
GIT_USER_NAME="Arjun Gupta" \
GIT_USER_EMAIL="arjuntheprogrammer@gmail.com" \
INSTALL_VSCODE_EXTENSIONS=1 \
VSCODE_EXTENSIONS_FILE=./setup/vscode-extensions.txt \
LOG_DIR=./logs \
APP_START_TIMEOUT_SECONDS=10800 \
RUN_GENERATE_TEST=0 \
TEST_IMAGE_PATH=./setup/test_input_512.png \
PROGRESS_LOG_EVERY_STEPS=1 \
PROGRESS_BAR_WIDTH=24 \
AUTO_MAX_INPUT_SIDE=1 \
MAX_INPUT_IMAGE_SIDE=auto \
OOM_SAFE_NUM_FRAMES=32 \
OOM_SAFE_STEPS=12 \
DEFAULT_FPS=24 \
DEFAULT_DURATION_SECONDS=6 \
DEFAULT_NUM_INFERENCE_STEPS=45 \
DEFAULT_GUIDANCE_SCALE=7.5 \
MAX_PROMPT_WORDS=60 \
MAX_NEGATIVE_PROMPT_WORDS=60 \
DEFAULT_OUTPUT_LONG_EDGE=1080 \
OUTPUT_LONG_EDGE_OPTIONS=720,1080,1440 \
ENABLE_DEFLICKER=1 \
DEFLICKER_WINDOW=3 \
ENABLE_CAPTIONING=1 \
CAPTION_MODEL_ID=Salesforce/blip2-opt-2.7b \
CAPTION_DEVICE=cpu \
CAPTION_MAX_TOKENS=40 \
ENABLE_SEQUENTIAL_CPU_OFFLOAD=0 \
ENABLE_MODEL_CPU_OFFLOAD=1 \
ENABLE_XFORMERS=0 \
./setup/setup.sh
```

To skip VS Code extension install:

```bash
INSTALL_VSCODE_EXTENSIONS=0 ./setup/setup.sh
```

## Output / logs

- Log directory: `<repo>/logs`
- App log: `<repo>/logs/hunyuan_app.log`
- App pid: `<repo>/logs/hunyuan_app.pid`
- Health snapshot: `<repo>/logs/hunyuan_health.json`
- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Gradio: `http://127.0.0.1:7860`

`setup.sh` appends to `hunyuan_app.log` on each restart (no truncation), so logs remain after process stop/kill.

## Runtime compatibility notes

- `requirements.txt` pins `transformers` to `<5.0.0` for model compatibility.
- xFormers memory-efficient attention is opt-in via `ENABLE_XFORMERS=1`.
- Auto input sizing is enabled via `AUTO_MAX_INPUT_SIDE=1`. Set `MAX_INPUT_IMAGE_SIDE` to a number to override.
- Output video is resized to the selected long-edge preset while preserving input aspect ratio.
- Deflicker post-processing is enabled by default; toggle via `ENABLE_DEFLICKER` or the API parameter.
- Gradio uses a free-form `Frames` input so duration is not UI-capped (`duration = frames / fps`).
- Prompt enhancement is enabled by default:
  - builds a structured prompt from dropdown fields + user text
  - optionally prepends a BLIP-2 caption for better identity anchoring
  - appends realism instructions to user prompt
  - applies negative prompt terms to reduce flicker/fade/morph artifacts
  - optional overrides: `DEFAULT_PROMPT_SUFFIX`, `DEFAULT_NEGATIVE_PROMPT`
- Generation retry behavior:
  - first attempt uses user-requested frames/steps at highest allowed resolution
  - frame/step and resolution are downgraded only after OOM

## Structured prompt dropdowns

| Field | Options |
| --- | --- |
| subject | person, product, food, fashion, animal, vehicle, architecture, landscape, cityscape, gadget |
| action | walking, turning head, smiling, hand gesture, hair movement, pouring, rotating, hovering, panning reveal, still |
| camera_motion | static, slow pan, tilt, dolly in, dolly out, orbit, handheld, zoom in, zoom out |
| shot_type | close-up, medium, wide, macro, overhead, low angle, high angle |
| lighting | soft daylight, golden hour, studio softbox, neon, backlit, overcast, candlelight |
| mood | cinematic, calm, energetic, moody, dreamy, documentary, romantic, dramatic |

Output long-edge presets: `720`, `1080`, `1440` (aspect ratio derived from the input image).

## Restart app manually

```bash
pkill -f "python run.py" || true
source /opt/conda/etc/profile.d/conda.sh
conda activate hunyuanvideo
cd /home/ubuntu/text-to-video-hunyuan-model
set -a; source .env; set +a
python run.py
```

## History-informed notes

From shell history on this instance, previous actions included:

- `git clone https://github.com/arjuntheprogrammer/text-to-video-hunyuan-model`
- `nvidia-smi`
- `ls /.dockerenv`
- multiple Docker/systemctl troubleshooting commands

Because this environment is containerized and Docker-in-Docker is restricted, this setup deliberately avoids Docker and uses Conda-native execution.

## One-time cache migration (if old cache exists in `/workspace/.hf_home`)

If you already downloaded the model earlier to `/workspace/.hf_home`, run this once to avoid re-downloading:

```bash
cd /home/ubuntu/text-to-video-hunyuan-model
pkill -f "python run.py" || true
find ./models -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete /workspace/.hf_home/ ./models/
else
  cp -a /workspace/.hf_home/. ./models/
fi
```
