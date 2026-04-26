#!/usr/bin/env bash
# Deploy the AIC static results dashboard as a Hugging Face Space (Docker SDK).
#
# This MUST NOT modify the canonical judge env Space (KINGKK007/aic-training).
# It creates/pushes a separate Space that serves `dashboard/site/` as static files.
#
# Usage:
#   export HF_TOKEN=hf_xxx                      # write token
#   export HF_USER=KINGKK007                    # optional
#   export SPACE_NAME=aic-results-dashboard     # optional
#   bash scripts/deploy_hf_dashboard_space.sh
#
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${HF_USER:-KINGKK007}"
SPACE_NAME="${SPACE_NAME:-aic-results-dashboard}"
SPACE_FULL="${HF_USER}/${SPACE_NAME}"
SPACE_HOST="$(printf '%s-%s.hf.space' "$HF_USER" "$SPACE_NAME" | tr '[:upper:]' '[:lower:]')"
SPACE_URL="https://${SPACE_HOST}"

WORK_DIR="${WORK_DIR:-/tmp/hf-aic-results-dashboard}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[err] HF_TOKEN env var must be set (write-scope token from huggingface.co/settings/tokens)"
  exit 1
fi

echo "============================================================"
echo " AIC | HF Dashboard Space deploy"
echo " space  : ${SPACE_FULL}"
echo " url    : ${SPACE_URL}"
echo " work   : ${WORK_DIR}"
echo "============================================================"

echo
echo "[1/3] creating Space (no-op if it already exists) ..."
python - <<PY
from huggingface_hub import HfApi
import os, sys
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    api.create_repo(
        repo_id="${SPACE_FULL}",
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    print("[ok] Space exists or was created: ${SPACE_FULL}")
except Exception as exc:
    print(f"[err] create_repo failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "[2/3] preparing working tree at ${WORK_DIR} ..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git init -q
git remote add origin "https://USER:${HF_TOKEN}@huggingface.co/spaces/${SPACE_FULL}"

# Copy the minimal payload for a static dashboard Space.
mkdir -p hf_dashboard_space
cp "${REPO_ROOT}/hf_dashboard_space/Dockerfile" hf_dashboard_space/Dockerfile
cp "${REPO_ROOT}/hf_dashboard_space/nginx.conf" hf_dashboard_space/nginx.conf
cp "${REPO_ROOT}/hf_dashboard_space/README.md" hf_dashboard_space/README.md
cp -R "${REPO_ROOT}/dashboard/site" dashboard

# Space root requirements:
# - Dockerfile at repo root
# - README.md at repo root
cp hf_dashboard_space/Dockerfile ./Dockerfile
cp hf_dashboard_space/README.md ./README.md
cp hf_dashboard_space/nginx.conf ./nginx.conf

# Copy static files into repo root for nginx COPY line stability
rm -rf ./dashboard/site/__pycache__ 2>/dev/null || true

echo
echo "[3/3] committing and pushing to Space repo ..."
git config user.email "deploy-bot@aic.local"
git config user.name "AIC Deploy Bot"
git add -A
git commit -q -m "deploy AIC results dashboard (static nginx on :7860)" || true
git branch -M main
git push -u origin main --force

echo
echo "============================================================"
echo " done. Space URL: ${SPACE_URL}"
echo "============================================================"

