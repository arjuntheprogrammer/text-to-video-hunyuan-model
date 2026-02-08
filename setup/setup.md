# HunyuanVideo Conda Setup (No Docker)

This setup is for running `text-to-video-hunyuan-model` directly on the remote GPU instance using Conda.

Docker is intentionally not used.

## Files

- Script: `setup/setup.sh`
- Repo expected by default at: parent of this script (`<repo>/setup/setup.sh` -> repo root)

## What `setup.sh` does

The script is idempotent and non-interactive:

1. Checks and installs required OS dependencies if missing:
   - `wget`, `curl`, `ca-certificates`, `bzip2`, `zsh`, `ffmpeg`, `git`, `procps`, `libgl1`, `libglib2.0-0`
2. Installs Miniconda to `/opt/conda` if not present.
3. Accepts Conda channel ToS non-interactively.
4. Initializes Conda for `bash` and `zsh`.
5. Creates Conda env `hunyuanvideo` if missing (`python=3.10`, `pip`, `ffmpeg`).
6. Installs Python dependencies from `requirements.txt`.
7. Ensures `.env` exists and has a valid `HF_TOKEN`.
8. Sets cache/runtime paths in `.env` to repo-local folders:
   - `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME` -> `<repo>/models`
   - `OUTPUT_DIR` -> `<repo>/outputs`
9. Starts app in background (`python run.py`).
10. Waits for `http://127.0.0.1:8000/health` to become available.

Optional:

- If `RUN_GENERATE_TEST=1`, runs one real `/generate` request and downloads output MP4.

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
APP_START_TIMEOUT_SECONDS=10800 \
RUN_GENERATE_TEST=0 \
./setup/setup.sh
```

## Output / logs

- App log: `/tmp/hunyuan_app.log`
- App pid: `/tmp/hunyuan_app.pid`
- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Gradio: `http://127.0.0.1:7860`

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
