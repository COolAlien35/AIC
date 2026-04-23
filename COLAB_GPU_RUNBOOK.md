# Colab GPU Runbook (GRPO Proof)

Use this runbook to close the remaining GPU-only gaps from `REMAINING_GAPS.md`.

## 1) Start Colab correctly

- Open [Google Colab](https://colab.research.google.com/)
- Runtime -> Change runtime type
- Hardware accelerator -> `GPU` (T4/A100/L4 all fine)
- Save

## 2) One-cell setup + run

Run this in a Colab code cell:

```bash
%%bash
set -euo pipefail

if [ ! -d "/content/AIC" ]; then
  git clone https://github.com/COolAlien35/AIC.git /content/AIC
fi
cd /content/AIC

chmod +x scripts/colab_gpu_proof.sh
./scripts/colab_gpu_proof.sh
```

## 3) If your fork/private repo is needed

Replace clone command with your repo:

```bash
%%bash
set -euo pipefail
git clone https://github.com/<your-user>/AIC.git /content/AIC
cd /content/AIC
chmod +x scripts/colab_gpu_proof.sh
./scripts/colab_gpu_proof.sh
```

## 4) Download artifacts back to your machine

Run in a new Colab cell:

```python
import shutil
from google.colab import files

shutil.make_archive("/content/aic_gpu_proof_artifacts", "zip", "/content/AIC", "results")
shutil.make_archive("/content/aic_gpu_proof_checkpoints", "zip", "/content/AIC", "checkpoints")
shutil.make_archive("/content/aic_gpu_proof_logs", "zip", "/content/AIC", "logs")

files.download("/content/aic_gpu_proof_artifacts.zip")
files.download("/content/aic_gpu_proof_checkpoints.zip")
files.download("/content/aic_gpu_proof_logs.zip")
```

## 5) Expected outputs to verify

- `checkpoints/grpo/`
- `logs/eval/policy_benchmark.jsonl`
- `results/benchmark_summary.csv`
- `results/reward_curve.png`
- `results/verifier_pass_rate.png`
- `results/before_after_demo.md`
- `results/evidence_manifest.json`
- `results/evidence_manifest.md`

## 6) Common failure fixes

- **No GPU visible:** rerun with GPU runtime enabled; verify with `!nvidia-smi`.
- **Python 3.12 not available in Colab:** script falls back to `python3 -m venv`.
- **OOM on GRPO:** restart runtime and retry on a higher-memory GPU tier.
- **Long run interrupted:** rerun `./scripts/colab_gpu_proof.sh`; artifacts are regenerated.
