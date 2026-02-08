#!/usr/bin/env bash
set -euo pipefail

# Basic structured logging helpers for readable script output.
log() {
  printf '[setup] %s\n' "$*"
}

err() {
  printf '[setup][error] %s\n' "$*" >&2
}

# Resolve script location and infer repo directory.
# Supports:
# 1) preferred layout: <repo>/setup/setup.sh
# 2) explicit REPO_DIR override
# 3) legacy fallback layout
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${REPO_DIR:-}" ]]; then
  REPO_DIR="${REPO_DIR}"
elif [[ -f "${PARENT_DIR}/requirements.txt" && -f "${PARENT_DIR}/run.py" ]]; then
  # Preferred layout: this script lives at <repo>/setup/setup.sh
  REPO_DIR="${PARENT_DIR}"
elif [[ -d "${PARENT_DIR}/text-to-video-hunyuan-model" ]]; then
  # Backward-compatible fallback
  REPO_DIR="${PARENT_DIR}/text-to-video-hunyuan-model"
else
  REPO_DIR="${PARENT_DIR}"
fi

# Configurable defaults (can be overridden via environment variables).
CONDA_DIR="${CONDA_DIR:-/opt/conda}"
CONDA_BIN="${CONDA_DIR}/bin/conda"
ENV_NAME="${ENV_NAME:-hunyuanvideo}"
GIT_USER_NAME="${GIT_USER_NAME:-Arjun Gupta}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-arjuntheprogrammer@gmail.com}"
INSTALL_VSCODE_EXTENSIONS="${INSTALL_VSCODE_EXTENSIONS:-1}"
VSCODE_EXTENSIONS_FILE="${VSCODE_EXTENSIONS_FILE:-${REPO_DIR}/setup/vscode-extensions.txt}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

APP_LOG="${APP_LOG:-${LOG_DIR}/hunyuan_app.log}"
APP_PID_FILE="${APP_PID_FILE:-${LOG_DIR}/hunyuan_app.pid}"
HEALTH_STATUS_FILE="${HEALTH_STATUS_FILE:-${LOG_DIR}/hunyuan_health.json}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
APP_START_TIMEOUT_SECONDS="${APP_START_TIMEOUT_SECONDS:-10800}" # 3 hours (first model download can be large)

# Validate prerequisites that must exist before continuing.
if [[ ! -d "${REPO_DIR}" ]]; then
  err "Repo directory not found: ${REPO_DIR}"
  err "Set REPO_DIR to the repo path and rerun."
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  err "apt-get is required on this host."
  exit 1
fi

mkdir -p "${LOG_DIR}"

# Determine whether elevated privileges are needed for package install.
SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    err "Please run as root or install sudo."
    exit 1
  fi
fi

# Check core OS tools and install only when missing.
need_install=0
for bin in wget curl zsh ffmpeg git; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    need_install=1
    break
  fi
done

if [[ "${need_install}" -eq 1 ]]; then
  log "Installing OS packages (missing binaries detected)."
  ${SUDO} apt-get update
  ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget curl ca-certificates bzip2 zsh ffmpeg git procps libgl1 libglib2.0-0
else
  log "OS packages already present; skipping apt install."
fi

# Ensure global git identity is configured for commits on fresh instances.
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
log "Git identity set: $(git config --global user.name) <$(git config --global user.email)>"

# Resolve a usable VS Code CLI (Remote-SSH/remote-cli or local code).
find_vscode_code_cli() {
  if command -v code >/dev/null 2>&1; then
    command -v code
    return 0
  fi

  local candidates=()
  local user_home="${HOME:-}"
  local sudo_home=""
  local login_home=""

  if [[ -n "${SUDO_USER:-}" ]]; then
    sudo_home="$(getent passwd "${SUDO_USER}" 2>/dev/null | cut -d: -f6 || true)"
  fi
  if command -v logname >/dev/null 2>&1; then
    login_home="$(getent passwd "$(logname 2>/dev/null || true)" 2>/dev/null | cut -d: -f6 || true)"
  fi

  for h in "${user_home}" "${sudo_home}" "${login_home}" "/home/ubuntu" "/root"; do
    [[ -z "${h}" ]] && continue
    if compgen -G "${h}/.vscode-server/cli/servers/Stable-*/server/bin/remote-cli/code" >/dev/null; then
      while IFS= read -r p; do
        candidates+=("${p}")
      done < <(ls -1dt "${h}"/.vscode-server/cli/servers/Stable-*/server/bin/remote-cli/code 2>/dev/null || true)
    fi
  done

  if [[ "${#candidates[@]}" -gt 0 ]]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  return 1
}

# Install VS Code extensions listed in setup/vscode-extensions.txt when requested.
if [[ "${INSTALL_VSCODE_EXTENSIONS}" == "1" ]]; then
  if [[ -f "${VSCODE_EXTENSIONS_FILE}" ]]; then
    if vscode_code_cli="$(find_vscode_code_cli)"; then
      log "Installing VS Code extensions via: ${vscode_code_cli}"
      while IFS= read -r ext || [[ -n "${ext}" ]]; do
        # Skip comments and empty lines.
        [[ -z "${ext}" ]] && continue
        [[ "${ext}" =~ ^# ]] && continue
        if "${vscode_code_cli}" --install-extension "${ext}" --force >/dev/null 2>&1; then
          log "VS Code extension ensured: ${ext}"
        else
          err "Failed to install VS Code extension: ${ext}"
        fi
      done < "${VSCODE_EXTENSIONS_FILE}"
    else
      log "VS Code CLI not found. Skipping extension install."
      log "Connect once with VS Code Remote-SSH and rerun to install extensions automatically."
    fi
  else
    log "No extension list file found at ${VSCODE_EXTENSIONS_FILE}; skipping VS Code extension install."
  fi
else
  log "INSTALL_VSCODE_EXTENSIONS=${INSTALL_VSCODE_EXTENSIONS}; skipping VS Code extension install."
fi

# Install Miniconda only if conda is not already available at target location.
if [[ ! -x "${CONDA_BIN}" ]]; then
  log "Installing Miniconda to ${CONDA_DIR}."
  tmp_installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${tmp_installer}"
  ${SUDO} bash "${tmp_installer}" -b -p "${CONDA_DIR}"
  rm -f "${tmp_installer}"
else
  log "Conda already installed at ${CONDA_BIN}; skipping Miniconda install."
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  err "Conda install failed: ${CONDA_BIN} not found."
  exit 1
fi

log "Conda version: $(${CONDA_BIN} --version)"

# Accept conda channel ToS non-interactively to avoid first-run blocking.
${CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
${CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# Initialize shell hooks so conda activation works for both bash and zsh sessions.
${CONDA_BIN} init bash >/dev/null 2>&1 || true
${CONDA_BIN} init zsh >/dev/null 2>&1 || true

# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Create the runtime env once; skip creation if it already exists.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Conda env '${ENV_NAME}' exists; skipping create."
else
  log "Creating conda env '${ENV_NAME}' (python=3.10, pip, ffmpeg)."
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.10 pip ffmpeg
fi

# Install/refresh Python dependencies required by the app.
log "Installing Python dependencies in env '${ENV_NAME}'."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${REPO_DIR}/requirements.txt"

# Load .env (or create it from template) and validate Hugging Face token.
ENV_FILE="${REPO_DIR}/.env"
ENV_TEMPLATE="${REPO_DIR}/.env_template"

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${ENV_TEMPLATE}" ]]; then
    log "Creating .env from template."
    cp "${ENV_TEMPLATE}" "${ENV_FILE}"
  else
    err ".env and .env_template are both missing."
    exit 1
  fi
fi

# Helper: insert or replace KEY=value entries in .env.
upsert_env_key() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -q "^${key}=" "${file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${file}"
  fi
}

# Helper: remove KEY=value entries from .env if present.
remove_env_key() {
  local file="$1"
  local key="$2"
  sed -i "/^${key}=/d" "${file}"
}

# Helper: read the latest value for a key from .env.
read_env_key() {
  local file="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "${file}" | tail -n 1
}

# Ensure HF token is available from .env or environment variable.
token_in_file="$(read_env_key "${ENV_FILE}" "HF_TOKEN")"
if [[ -z "${token_in_file}" || "${token_in_file}" == "your_huggingface_token_here" ]]; then
  if [[ -n "${HF_TOKEN:-}" && "${HF_TOKEN}" != "your_huggingface_token_here" ]]; then
    log "HF_TOKEN in .env is missing/placeholder; setting from environment."
    upsert_env_key "${ENV_FILE}" "HF_TOKEN" "${HF_TOKEN}"
  else
    err "HF_TOKEN is missing."
    err "Set HF_TOKEN in ${ENV_FILE}, or export HF_TOKEN before running this script."
    exit 1
  fi
fi

# Force cache/output paths to repo-local directories for reproducible setup.
mkdir -p "${REPO_DIR}/models" "${REPO_DIR}/outputs"
upsert_env_key "${ENV_FILE}" "HF_HOME" "${REPO_DIR}/models"
upsert_env_key "${ENV_FILE}" "HF_HUB_CACHE" "${REPO_DIR}/models"
# TRANSFORMERS_CACHE is deprecated in transformers v5; remove to avoid warnings.
remove_env_key "${ENV_FILE}" "TRANSFORMERS_CACHE"
upsert_env_key "${ENV_FILE}" "TORCH_HOME" "${REPO_DIR}/models"
upsert_env_key "${ENV_FILE}" "OUTPUT_DIR" "${REPO_DIR}/outputs"
upsert_env_key "${ENV_FILE}" "API_HOST" "0.0.0.0"
upsert_env_key "${ENV_FILE}" "API_PORT" "8000"
upsert_env_key "${ENV_FILE}" "GRADIO_SERVER_NAME" "0.0.0.0"
upsert_env_key "${ENV_FILE}" "GRADIO_SERVER_PORT" "7860"

# Print GPU summary if available (informational only).
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

# Reuse running app if PID file is valid; otherwise clear stale PID file.
if [[ -f "${APP_PID_FILE}" ]]; then
  existing_pid="$(cat "${APP_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && ps -p "${existing_pid}" >/dev/null 2>&1; then
    log "Existing app process found (pid=${existing_pid}); keeping it."
  else
    rm -f "${APP_PID_FILE}"
  fi
fi

# Start app in background when not already running.
if [[ ! -f "${APP_PID_FILE}" ]]; then
  if pgrep -f "python run.py" >/dev/null 2>&1; then
    log "Cleaning stale run.py process."
    pkill -f "python run.py" || true
    sleep 1
  fi

  log "Starting application in background."
  nohup bash -lc "
    set -euo pipefail
    source '${CONDA_DIR}/etc/profile.d/conda.sh'
    conda activate '${ENV_NAME}'
    cd '${REPO_DIR}'
    set -a
    source '${ENV_FILE}'
    set +a
    python run.py
  " >"${APP_LOG}" 2>&1 &
  echo $! > "${APP_PID_FILE}"
fi

app_pid="$(cat "${APP_PID_FILE}")"
log "App pid: ${app_pid}"
log "App log: ${APP_LOG}"

# Wait until health endpoint is ready or timeout/crash is detected.
start_ts="$(date +%s)"
while true; do
  if curl -fsS "${HEALTH_URL}" >"${HEALTH_STATUS_FILE}" 2>/dev/null; then
    log "Health endpoint is up: ${HEALTH_URL}"
    cat "${HEALTH_STATUS_FILE}"
    break
  fi

  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if [[ "${elapsed}" -ge "${APP_START_TIMEOUT_SECONDS}" ]]; then
    err "Timed out waiting for app health (${APP_START_TIMEOUT_SECONDS}s)."
    err "Last log lines:"
    tail -n 80 "${APP_LOG}" || true
    exit 1
  fi

  if ! ps -p "${app_pid}" >/dev/null 2>&1; then
    err "App process exited before becoming healthy."
    err "Last log lines:"
    tail -n 120 "${APP_LOG}" || true
    exit 1
  fi

  cache_dir="$(read_env_key "${ENV_FILE}" "HF_HOME")"
  if [[ -n "${cache_dir}" && -d "${cache_dir}" ]]; then
    cache_size="$(du -sh "${cache_dir}" 2>/dev/null | awk '{print $1}')"
    log "Waiting for startup... elapsed=${elapsed}s cache=${cache_size:-n/a}"
  else
    log "Waiting for startup... elapsed=${elapsed}s"
  fi

  sleep 10
done

# Optional: run one real generation request to verify end-to-end inference.
if [[ "${RUN_GENERATE_TEST:-0}" == "1" ]]; then
  log "RUN_GENERATE_TEST=1: running a real generate test."
  test_img="/tmp/hunyuan_test_input.png"
  test_mp4="/tmp/hunyuan_test_output.mp4"
  # Use ffmpeg to generate a valid RGB test image.
  ffmpeg -loglevel error -f lavfi -i color=c=blue:s=512x512:d=1 -frames:v 1 -y "${test_img}"

  gen_json="$(curl -fsS -X POST 'http://127.0.0.1:8000/generate' \
    -F "image=@${test_img}" \
    -F "prompt=simple camera motion" \
    -F "num_frames=16" \
    -F "steps=10" \
    -F "fps=8" \
    -F "guidance_scale=6.0" \
    -F "seed=42")"
  echo "${gen_json}" > /tmp/hunyuan_generate.json

  output_url="$(sed -n 's/.*"output_url":"\([^"]*\)".*/\1/p' /tmp/hunyuan_generate.json | head -n 1)"
  if [[ -z "${output_url}" ]]; then
    err "Generate call succeeded but output_url was not found."
    cat /tmp/hunyuan_generate.json
    exit 1
  fi

  curl -fsS "http://127.0.0.1:8000${output_url}" -o "${test_mp4}"
  if [[ ! -s "${test_mp4}" ]]; then
    err "Downloaded test video is empty: ${test_mp4}"
    exit 1
  fi
  log "Generate test passed. Output: ${test_mp4}"
fi

log "Setup complete."
log "API URL: http://127.0.0.1:8000"
log "Docs URL: http://127.0.0.1:8000/docs"
log "Gradio URL: http://127.0.0.1:7860"
