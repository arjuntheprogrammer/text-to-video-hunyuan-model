#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ubuntu/Development"
REPO_URL="https://github.com/arjuntheprogrammer/text-to-video-hunyuan-model"
REPO_DIR="${BASE_DIR}/text-to-video-hunyuan-model"

mkdir -p "${BASE_DIR}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  printf '[base_setup] Repo already exists: %s\n' "${REPO_DIR}"
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

bash "${REPO_DIR}/setup/setup.sh"
