#!/usr/bin/env bash
# Entrypoint for the AIC Training HF Space.
#
# Behavior:
#   1. Clone (or update) the GitHub repo into /workspace/aic-repo.
#   2. Install the pinned requirements + editable install of the package.
#   3. Launch JupyterLab on port 7860 with the JUPYTER_TOKEN secret.
set -euo pipefail

REPO_URL="${AIC_REPO_URL:-https://github.com/COolAlien35/AIC.git}"
REPO_DIR="/workspace/aic-repo"
JUPY_TOKEN="${JUPYTER_TOKEN:-aic}"

cd /workspace

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[start.sh] Cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
    echo "[start.sh] Repo already present; pulling latest"
    (cd "${REPO_DIR}" && git pull --ff-only origin main || true)
fi

echo "[start.sh] Installing requirements..."
python -m pip install --user --no-cache-dir -r "${REPO_DIR}/requirements.txt"
python -m pip install --user --no-cache-dir -e "${REPO_DIR}"

mkdir -p /workspace/.hf_home

echo "[start.sh] Starting JupyterLab on :7860"
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=7860 \
    --no-browser \
    --ServerApp.token="${JUPY_TOKEN}" \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.root_dir=/workspace \
    --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}'
