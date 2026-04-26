#!/usr/bin/env bash
# Produce a judge-ready GPU benchmark evidence bundle.
#
# Run this in the HF Space terminal (or any GPU env). It:
#   1. Captures the runtime environment (nvidia-smi, pip freeze, git, python).
#   2. Downloads the trained checkpoint from HF Hub (if not already present).
#   3. Runs scripts/run_final_benchmark.py against it (real per-episode rewards).
#   4. Tees the full console log so the run is auditable.
#   5. Computes SHA-256 for every artifact and writes PROVENANCE.md.
#
# After it finishes, push evidence/gpu_run/ to git from your local box and run
# `python scripts/finalize_gpu_evidence.py` to swap real numbers into results/.
#
# Usage (HF Space terminal):
#     export HF_MODEL_REPO="<your-username>/<your-grpo-repo>"
#     bash scripts/gpu_evidence_run.sh
#
# Optional environment overrides:
#     EPISODES=10           # episodes per scenario (default 10 ≈ 30 min on T4)
#     SEED=42               # benchmark seed
#     CHECKPOINT_DIR=exports  # where the model is materialised
#     SKIP_DOWNLOAD=1       # do not re-clone, use whatever is in $CHECKPOINT_DIR

set -euo pipefail

# ----- Configuration -----
HF_MODEL_REPO="${HF_MODEL_REPO:-}"       # REQUIRED unless SKIP_DOWNLOAD=1
EPISODES="${EPISODES:-10}"
SEED="${SEED:-42}"
EVIDENCE_DIR="evidence/gpu_run"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-exports}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PY="${PY:-python3}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${EVIDENCE_DIR}/run_${RUN_ID}"
RESULTS_DIR="${OUT_DIR}/results"
ENV_DIR="${OUT_DIR}/env"
LOG_FILE="${OUT_DIR}/full_console.log"

mkdir -p "$RESULTS_DIR" "$ENV_DIR"

echo "============================================================"
echo " AIC | GPU evidence run"
echo " run id : ${RUN_ID}"
echo " out    : ${OUT_DIR}"
echo " repo   : ${HF_MODEL_REPO:-<skipping download>}"
echo " eps    : ${EPISODES} per scenario × 6 scenarios × 3 policies"
echo " seed   : ${SEED}"
echo "============================================================"

# ----- 1. Capture environment -----
echo
echo "[1/5] capturing runtime environment ..."
{
  echo "# AIC GPU evidence run"
  echo "run_id=${RUN_ID}"
  echo "started_at_utc=${START_TS}"
  echo "hostname=$(hostname)"
  echo "user=$(whoami)"
  echo "kernel=$(uname -a)"
  echo "python=$($PY -c 'import sys; print(sys.version)')"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  echo "git_status_short=$(git status --porcelain 2>/dev/null | wc -l) modified files"
  echo "hf_model_repo=${HF_MODEL_REPO}"
  echo "checkpoint_dir=${CHECKPOINT_DIR}"
  echo "episodes=${EPISODES}"
  echo "seed=${SEED}"
} > "${ENV_DIR}/run_metadata.txt"

git rev-parse HEAD > "${ENV_DIR}/git_commit.txt" 2>/dev/null || echo "unknown" > "${ENV_DIR}/git_commit.txt"
git status --porcelain > "${ENV_DIR}/git_status.txt" 2>/dev/null || true
git log -1 --format='%H%n%an <%ae>%n%aI%n%s' > "${ENV_DIR}/git_head.txt" 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${ENV_DIR}/nvidia_smi.txt" 2>&1 || true
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv \
    > "${ENV_DIR}/nvidia_smi_summary.csv" 2>&1 || true
else
  echo "nvidia-smi not found on PATH" > "${ENV_DIR}/nvidia_smi.txt"
fi

$PY -c "import torch, sys, json; print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'cuda_device_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), 'bf16_supported': (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)}, indent=2))" \
  > "${ENV_DIR}/torch_runtime.json" 2>&1 || true

$PY -m pip freeze > "${ENV_DIR}/pip_freeze.txt" 2>&1 || true

echo "      → ${ENV_DIR}/{nvidia_smi.txt,pip_freeze.txt,torch_runtime.json,git_commit.txt}"

# ----- 2. Materialise the trained checkpoint -----
echo
echo "[2/5] materialising checkpoint at ${CHECKPOINT_DIR} ..."
if [ "$SKIP_DOWNLOAD" = "1" ]; then
  echo "      SKIP_DOWNLOAD=1 → using existing ${CHECKPOINT_DIR}/"
elif [ -z "${HF_MODEL_REPO}" ]; then
  echo "ERROR: HF_MODEL_REPO is empty. Either:"
  echo "  export HF_MODEL_REPO='<user>/<grpo-repo>'   and re-run, or"
  echo "  export SKIP_DOWNLOAD=1                       to use existing ${CHECKPOINT_DIR}/"
  exit 2
else
  $PY -c "
import os, sys
from pathlib import Path
target = Path('${CHECKPOINT_DIR}')
target.mkdir(parents=True, exist_ok=True)
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f'huggingface_hub missing ({e}); installing ...', flush=True)
    os.system(f'{sys.executable} -m pip install -q --no-cache-dir huggingface_hub')
    from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id='${HF_MODEL_REPO}',
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
print(f'[ok] snapshot at {local}')
print('files:')
for p in sorted(Path(local).iterdir()):
    if p.is_file():
        print(f'  {p.name} ({p.stat().st_size} B)')
" 2>&1 | tee "${ENV_DIR}/hf_snapshot.log"
fi

ls -la "${CHECKPOINT_DIR}" > "${ENV_DIR}/checkpoint_listing.txt" 2>&1 || true

echo "      → ${ENV_DIR}/checkpoint_listing.txt"

# ----- 3. Run the benchmark with full console capture -----
echo
echo "[3/5] running benchmark (${EPISODES} ep/scenario × 3 policies × 6 scenarios) ..."
echo "      results → ${RESULTS_DIR}/"
echo "      log     → ${LOG_FILE}"
echo

{
  echo "##### AIC benchmark stdout for run ${RUN_ID} #####"
  echo "##### started ${START_TS} on $(hostname)        #####"
  echo
  $PY scripts/run_final_benchmark.py \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --output "${RESULTS_DIR}" \
    --checkpoint-path "${CHECKPOINT_DIR}" \
    --strict
  rc=$?
  echo
  echo "##### exit_code=${rc} finished $(date -u +%Y-%m-%dT%H:%M:%SZ) #####"
} 2>&1 | tee "${LOG_FILE}"

# ----- 4. Sidecar artifacts (training callback output, if present) -----
echo
echo "[4/5] copying training callback artifacts (if present) ..."
TRAIN_DIR="${OUT_DIR}/training_artifacts"
mkdir -p "${TRAIN_DIR}"
for src in \
  logs/grpo_progress.jsonl \
  logs/curriculum.jsonl \
  checkpoints/grpo/training_summary.json \
  checkpoints/grpo/grpo_metadata.json \
  checkpoints/grpo/grpo_audit_summary.json \
  exports/training_summary.json
do
  if [ -f "${src}" ]; then
    dst="${TRAIN_DIR}/$(basename "$src")"
    cp -p "${src}" "${dst}"
    echo "      copied ${src} → ${dst}"
  fi
done

# Persist a head/tail of full grpo_progress.jsonl as a text summary
if [ -f logs/grpo_progress.jsonl ]; then
  {
    echo "# Head (first 5 steps)"
    head -n 5 logs/grpo_progress.jsonl || true
    echo
    echo "# Tail (last 5 steps)"
    tail -n 5 logs/grpo_progress.jsonl || true
    echo
    echo "# total_steps=$(wc -l < logs/grpo_progress.jsonl)"
  } > "${TRAIN_DIR}/grpo_progress_summary.txt"
fi

# ----- 5. Manifest + provenance -----
echo
echo "[5/5] writing MANIFEST.sha256 and PROVENANCE.md ..."

(
  cd "${OUT_DIR}"
  if command -v shasum >/dev/null 2>&1; then
    HASHER="shasum -a 256"
  else
    HASHER="sha256sum"
  fi
  find . -type f ! -name MANIFEST.sha256 ! -name PROVENANCE.md \
    | sort | xargs $HASHER > MANIFEST.sha256
)

END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "${OUT_DIR}/PROVENANCE.md" <<EOF
# AIC GPU Evidence Run — \`${RUN_ID}\`

## Provenance

| field | value |
|---|---|
| run_id | \`${RUN_ID}\` |
| started_at_utc | \`${START_TS}\` |
| finished_at_utc | \`${END_TS}\` |
| hostname | \`$(hostname)\` |
| git_commit | \`$(cat ${ENV_DIR}/git_commit.txt 2>/dev/null || echo unknown)\` |
| hf_model_repo | \`${HF_MODEL_REPO:-<not downloaded; SKIP_DOWNLOAD=1>}\` |
| checkpoint_dir | \`${CHECKPOINT_DIR}\` |
| episodes per scenario | \`${EPISODES}\` |
| seed | \`${SEED}\` |
| benchmark script | \`scripts/run_final_benchmark.py --episodes ${EPISODES} --seed ${SEED} --output ${RESULTS_DIR} --checkpoint-path ${CHECKPOINT_DIR} --strict\` |

## Files

\`\`\`
$(cd "${OUT_DIR}" && find . -type f | sort)
\`\`\`

## How to verify

\`\`\`bash
# from repo root, after pulling this folder:
cd ${OUT_DIR}
shasum -a 256 -c MANIFEST.sha256       # or: sha256sum -c MANIFEST.sha256
cat results/statistical_test.json
\`\`\`

## Reproducing this run

\`\`\`bash
# On any GPU machine with this repo checked out:
export HF_MODEL_REPO='${HF_MODEL_REPO:-<your-user>/<your-grpo-repo>}'
EPISODES=${EPISODES} SEED=${SEED} bash scripts/gpu_evidence_run.sh
\`\`\`
EOF

echo
echo "============================================================"
echo " DONE."
echo " bundle: ${OUT_DIR}"
echo " key files:"
echo "   ${RESULTS_DIR}/benchmark_summary.csv"
echo "   ${RESULTS_DIR}/benchmark_by_scenario.csv"
echo "   ${RESULTS_DIR}/benchmark_episodes.csv"
echo "   ${RESULTS_DIR}/statistical_test.json"
echo "   ${RESULTS_DIR}/benchmark_run_config.json"
echo "   ${LOG_FILE}"
echo "   ${ENV_DIR}/{nvidia_smi.txt,pip_freeze.txt,torch_runtime.json,git_commit.txt}"
echo "   ${OUT_DIR}/MANIFEST.sha256"
echo "   ${OUT_DIR}/PROVENANCE.md"
echo
echo " next steps:"
echo "   1) git add ${OUT_DIR} && git commit -m 'evidence: real GPU benchmark run ${RUN_ID}'"
echo "   2) git push, then locally run:"
echo "      python scripts/finalize_gpu_evidence.py --run ${OUT_DIR}"
echo "============================================================"
