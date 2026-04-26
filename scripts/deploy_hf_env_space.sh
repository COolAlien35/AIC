#!/usr/bin/env bash
# Deploy the AIC OpenEnv environment as a public Hugging Face Space.
#
# This is the URL judges pull to evaluate the environment. Keep
# `KINGKK007/aic-incident-command-center` as the optional Gradio demo; this
# new Space (`KINGKK007/aic-openenv-env`) is the canonical OpenEnv server.
#
# Usage:
#     export HF_TOKEN=hf_xxx                          # write token
#     bash scripts/deploy_hf_env_space.sh             # creates + pushes
#     bash scripts/deploy_hf_env_space.sh push-only   # skip create
#
# After it runs, smoke-test:
#     curl https://kingkk007-aic-openenv-env.hf.space/health
#
# Logs go to results/hf_space_smoke.log so the run is auditable.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${HF_USER:-KINGKK007}"
SPACE_NAME="${SPACE_NAME:-aic-openenv-env}"
SPACE_FULL="${HF_USER}/${SPACE_NAME}"
SPACE_HOST="${HF_USER,,}-${SPACE_NAME,,}.hf.space"   # lowercased
SPACE_HOST="$(echo "$SPACE_HOST" | tr '[:upper:]' '[:lower:]')"
SPACE_URL="https://${SPACE_HOST}"

WORK_DIR="${WORK_DIR:-/tmp/hf-aic-openenv-env}"
SMOKE_LOG="${REPO_ROOT}/results/hf_space_smoke.log"
mkdir -p "$(dirname "$SMOKE_LOG")"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[err] HF_TOKEN env var must be set (write-scope token from huggingface.co/settings/tokens)"
    exit 1
fi

CMD="${1:-deploy}"

echo "============================================================"
echo " AIC | HF Space deploy"
echo " space  : ${SPACE_FULL}"
echo " url    : ${SPACE_URL}"
echo " work   : ${WORK_DIR}"
echo "============================================================"

if [ "$CMD" = "deploy" ]; then
    echo
    echo "[1/4] creating Space (no-op if it already exists) ..."
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
fi

echo
echo "[2/4] preparing working tree at ${WORK_DIR} ..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git init -q
git remote add origin "https://USER:${HF_TOKEN}@huggingface.co/spaces/${SPACE_FULL}"

# Copy only the files we need into the Space repo. Keep the image small.
cp -R "${REPO_ROOT}/aic" ./aic
cp -R "${REPO_ROOT}/server" ./server
cp "${REPO_ROOT}/openenv.yaml" ./openenv.yaml
cp "${REPO_ROOT}/pyproject.toml" ./pyproject.toml
cp "${REPO_ROOT}/hf_env_space/README.md" ./README.md
cp "${REPO_ROOT}/hf_env_space/Dockerfile" ./Dockerfile
cp "${REPO_ROOT}/hf_env_space/requirements-runtime.txt" ./requirements-runtime.txt

# Strip __pycache__ and .pyc to keep image lean
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

echo
echo "[3/4] committing and pushing to Space repo ..."
git config user.email "deploy-bot@aic.local"
git config user.name "AIC Deploy Bot"
git add -A
git commit -q -m "deploy AIC OpenEnv environment service (FastAPI on :7860)" || true

# HF Spaces use main as the default branch.
git branch -M main
git push -u origin main --force

echo
echo "[4/4] smoke-testing live endpoint ..."
{
    echo "# AIC HF Space smoke-test log"
    echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# space: ${SPACE_FULL}"
    echo "# url:   ${SPACE_URL}"
    echo
    echo "## Build wait (60s)"
    sleep 60

    echo "## /health"
    curl -fsS "${SPACE_URL}/health" || echo "[warn] /health not yet ready - HF Space may still be building"

    echo
    echo "## /reset"
    RESET=$(curl -fsS -X POST "${SPACE_URL}/reset" \
        -H 'Content-Type: application/json' \
        -d '{"episode_id":0,"base_seed":42,"fault_mode":"cascading_failure"}' || echo '{}')
    echo "$RESET"
    ENV_ID=$(python -c "import json,sys; print(json.loads('''$RESET''').get('env_id',''))" 2>/dev/null || echo "")

    if [ -n "$ENV_ID" ]; then
        echo
        echo "## /state/${ENV_ID}"
        curl -fsS "${SPACE_URL}/state/${ENV_ID}" | python -m json.tool | head -40 || true

        echo
        echo "## DELETE /env/${ENV_ID}"
        curl -fsS -X DELETE "${SPACE_URL}/env/${ENV_ID}" || true
    fi
} | tee "$SMOKE_LOG"

echo
echo "============================================================"
echo " done. Space URL: ${SPACE_URL}"
echo " smoke log     : ${SMOKE_LOG}"
echo "============================================================"
