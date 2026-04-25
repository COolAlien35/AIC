#!/usr/bin/env bash
# Entrypoint for the AIC Training HF Space.
set -euo pipefail

REPO_URL="${AIC_REPO_URL:-https://github.com/COolAlien35/AIC.git}"
REPO_DIR="/workspace/aic-repo"
REQ_BRANCH="${AIC_REPO_BRANCH:-feat/4xl4-ddp-space}"
FALLBACK_BRANCH="main"

clone_branch () {
    local branch="$1"
    git clone --depth 1 --branch "${branch}" "${REPO_URL}" "${REPO_DIR}" 2>/dev/null
}

cd /workspace

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[start.sh] Cloning ${REPO_URL}@${REQ_BRANCH} -> ${REPO_DIR}"
    if ! clone_branch "${REQ_BRANCH}"; then
        echo "[start.sh] WARN: branch ${REQ_BRANCH} not found; falling back to ${FALLBACK_BRANCH}"
        rm -rf "${REPO_DIR}"
        clone_branch "${FALLBACK_BRANCH}"
    fi
else
    echo "[start.sh] Repo present; fetching ${REQ_BRANCH}"
    (cd "${REPO_DIR}" \
        && git fetch --depth 1 origin "${REQ_BRANCH}" 2>/dev/null \
        && git checkout "${REQ_BRANCH}" \
        && git reset --hard "origin/${REQ_BRANCH}") \
        || (echo "[start.sh] WARN: ${REQ_BRANCH} fetch failed; keeping current" \
            && cd "${REPO_DIR}" && (git status || true))
fi

echo "[start.sh] Installing requirements..."
python -m pip install --user --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --user --no-cache-dir \
  "trl==0.14.0" "transformers==4.46.3" "datasets==3.1.0" \
  "accelerate==1.1.1" "peft==0.13.2" "bitsandbytes==0.46.1" \
  "tokenizers>=0.20,<0.21" "huggingface_hub>=0.26,<1.0"
python -m pip install --user --no-cache-dir -r "${REPO_DIR}/requirements.txt"

mkdir -p /workspace/.hf_home

python - <<'PY'
from trl import GRPOConfig, GRPOTrainer  # noqa: F401
import trl, bitsandbytes, torch
print(
    f"[start.sh] trl={trl.__version__} bnb={bitsandbytes.__version__} "
    f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"ngpu={torch.cuda.device_count()}"
)
PY

echo "[start.sh] Starting JupyterLab on :7860"
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=7860 \
  --no-browser \
  --ServerApp.token='' \
  --IdentityProvider.token='' \
  --ServerApp.root_dir=/workspace \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy":"frame-ancestors *","X-Frame-Options":"ALLOWALL"}}'
