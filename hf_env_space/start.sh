#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${AIC_REPO_URL:-https://github.com/COolAlien35/AIC.git}"
REPO_REF="${AIC_REPO_REF:-main}"
CHECKOUT_DIR="${AIC_CHECKOUT_DIR:-/app/aic_repo}"

echo "[aic-space] repo: ${REPO_URL} (${REPO_REF})"

if [ ! -d "${CHECKOUT_DIR}/.git" ]; then
  rm -rf "${CHECKOUT_DIR}" 2>/dev/null || true
  git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${CHECKOUT_DIR}"
else
  git -C "${CHECKOUT_DIR}" fetch --depth 1 origin "${REPO_REF}"
  git -C "${CHECKOUT_DIR}" checkout -q "${REPO_REF}"
  git -C "${CHECKOUT_DIR}" reset -q --hard "origin/${REPO_REF}"
fi

cd "${CHECKOUT_DIR}"

# Ensure PEP517/legacy editable backend is available in minimal images.
python -m pip install --no-cache-dir --upgrade pip setuptools wheel >/dev/null

# Install the package (editable) so `aic.server.env_api:app` is importable.
python -m pip install --no-cache-dir -e . >/dev/null

echo "[aic-space] starting uvicorn on :7860"
exec uvicorn aic.server.env_api:app --host 0.0.0.0 --port 7860

