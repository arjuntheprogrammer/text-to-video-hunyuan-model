#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ubuntu/Development"
REPO_URL="https://github.com/arjuntheprogrammer/text-to-video-hunyuan-model"
REPO_DIR="${BASE_DIR}/text-to-video-hunyuan-model"

# Set your Hugging Face token here before running.
HF_TOKEN="YOUR_HUGGINGFACE_TOKEN_HERE"

mkdir -p "${BASE_DIR}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  printf '[base_setup] Repo already exists: %s\n' "${REPO_DIR}"
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Create or update .env with the provided HF token.
ENV_FILE="${REPO_DIR}/.env"
ENV_TEMPLATE="${REPO_DIR}/.env_template"

if [[ "${HF_TOKEN}" == "YOUR_HUGGINGFACE_TOKEN_HERE" || -z "${HF_TOKEN}" ]]; then
  printf '[base_setup][error] HF_TOKEN placeholder not updated. Edit HF_TOKEN in this script.\n' >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${ENV_TEMPLATE}" ]]; then
    cp "${ENV_TEMPLATE}" "${ENV_FILE}"
  else
    printf 'HF_TOKEN=%s\n' "${HF_TOKEN}" > "${ENV_FILE}"
  fi
fi

if grep -q "^HF_TOKEN=" "${ENV_FILE}"; then
  sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_TOKEN}|" "${ENV_FILE}"
else
  printf 'HF_TOKEN=%s\n' "${HF_TOKEN}" >> "${ENV_FILE}"
fi

bash "${REPO_DIR}/setup/setup.sh"
