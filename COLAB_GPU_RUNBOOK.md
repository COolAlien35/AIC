# 🚀 Colab GPU Runbook — GRPO Training

**Goal:** reproduce the real **80-step GRPO run** that produced
[`results/grpo_reward_curve.png`](results/grpo_reward_curve.png),
[`results/grpo_training_summary.json`](results/grpo_training_summary.json), and the
trained adapter under `checkpoints/grpo/` — on a free Colab T4 GPU.

> **Wall-clock target:** ~6.19 h on a single T4 (or ~2.5 h on an L4) ·
> **Output reward:** −15.10 → −10.24 (+4.86) · **Final loss:** 0.0026.
> Full numbers in [`DESIGN.md §8`](DESIGN.md#8-training-architecture).

---

## 1) Pick a Colab runtime

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb)

- Open [Google Colab](https://colab.research.google.com/) → **File → Open notebook → GitHub** → paste
  `https://github.com/COolAlien35/AIC/blob/main/train_colab.ipynb`.
- **Runtime → Change runtime type** → set **Hardware accelerator** to **GPU** and the
  **GPU type** to **T4 (free)**, **L4**, or **A100** — all three are tested.

| GPU | VRAM | Expected GRPO wall-clock (80 steps) |
|---|---|---|
| **T4 (free)** | 16 GB | **~6.2 h** ✅ what we used |
| L4 | 22 GB | ~2.5 h |
| A100 | 40 GB | ~1.0 h |

---

## 2) The recommended path — run the notebook end-to-end

Open `train_colab.ipynb` in Colab and **Runtime → Run all**. The notebook will:

1. Clone `https://github.com/COolAlien35/AIC.git` into `/content/AIC`.
2. Install pinned `requirements.txt` (TRL, Unsloth, PEFT, accelerate, etc.).
3. Run `scripts/run_pipeline.py verify` (env health check).
4. Run the **SFT warm-up** on Qwen2.5-3B-Instruct (LoRA r=16, α=32, 4-bit NF4).
5. Run the **80-step GRPO** loop (TRL `GRPOTrainer` + Unsloth) and log every step into
   `logs/grpo_progress.jsonl`.
6. Generate plots via `scripts/plot_grpo_progress.py`:
   - `results/grpo_reward_curve.png`
   - `results/grpo_loss_curve.png`
   - `results/grpo_kl_curve.png`
   - `results/grpo_training_summary.json`
7. Run the multi-policy benchmark via `scripts/run_final_benchmark.py` and write
   `results/benchmark_summary.csv` + `results/benchmark_by_task_grader.csv`.

---

## 3) Or use the one-shot bash entrypoint

If you'd rather run everything from a single bash cell (skipping the notebook UI):

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

This calls [`scripts/colab_gpu_proof.sh`](scripts/colab_gpu_proof.sh) which runs:

```text
run_hackathon.py verify    # dependency + env sanity
run_hackathon.py grpo      # full GRPO loop
scripts/run_final_benchmark.py
run_hackathon.py plots demo
eval/test_export.py --source checkpoints/grpo
```

---

## 4) Download artefacts back to your laptop

After the run finishes, in a new Colab cell:

```python
import shutil
from google.colab import files

shutil.make_archive("/content/aic_grpo_results",      "zip", "/content/AIC", "results")
shutil.make_archive("/content/aic_grpo_checkpoints",  "zip", "/content/AIC", "checkpoints")
shutil.make_archive("/content/aic_grpo_logs",         "zip", "/content/AIC", "logs")

files.download("/content/aic_grpo_results.zip")
files.download("/content/aic_grpo_checkpoints.zip")
files.download("/content/aic_grpo_logs.zip")
```

Or push the trained adapter straight to the Hub:

```python
import os, subprocess
os.environ["HF_TOKEN"] = "<your token>"
subprocess.run([
    "/content/AIC/.venv/bin/python", "eval/test_export.py",
    "--source", "/content/AIC/checkpoints/grpo",
    "--push",   "--hub-repo", "<your-username>/aic-orchestrator",
], check=True)
```

---

## 5) Expected outputs to verify

| File | What it proves |
|---|---|
| `checkpoints/grpo/` | LoRA adapter + tokenizer survived training |
| `logs/grpo_progress.jsonl` | One JSON row per GRPO step (real training log) |
| `results/grpo_training_summary.json` | Headline numbers (`total_steps`, `reward_delta`, `training_time_minutes`) |
| `results/grpo_reward_curve.png` | Reward curve plot (rubric requirement) |
| `results/grpo_loss_curve.png` | Loss curve plot (rubric requirement) |
| `results/grpo_kl_curve.png` | KL-divergence curve plot |
| `results/benchmark_summary.csv` | frozen / adaptive / random_safe / trained scorecard |
| `results/benchmark_by_task_grader.csv` | Per-task 0–1 grader scores |
| `results/before_after_demo.md` | Qualitative before/after trace excerpt |
| `results/evidence_manifest.json` / `.md` | SHA-256 of every artefact, repro commands |

Confirm with:

```bash
ls -lh checkpoints/grpo/ logs/grpo_progress.jsonl results/grpo_*.png
cat results/grpo_training_summary.json
```

---

## 6) Common failures and fixes

| Symptom | Fix |
|---|---|
| `nvidia-smi` returns nothing | Re-select **Runtime → Change runtime type → GPU** and **Reconnect**. |
| `python3.12` not available in Colab | The shell script falls back to `python3 -m venv` automatically — no action needed. |
| `CUDA out of memory` during GRPO | Restart the runtime, drop GRPO `group_size` from 4 to 2 in `aic/training/config.py`, or move to L4/A100. |
| Notebook session disconnects mid-run | Re-running `scripts/colab_gpu_proof.sh` is idempotent; checkpoints under `checkpoints/grpo/` are reused. |
| `unsloth` import fails on Apple Silicon (local) | Unsloth is GPU-only; this runbook is **Colab-only**. The `train_colab.ipynb` notebook works there. |
| Want to skip GRPO and just see the env | Run the 60-second CPU path from [`README.md`](README.md#-reproduce-in-60-seconds-cpu-only) instead. |

---

## 7) Post-run sanity checklist

Once the run finishes:

- [ ] `cat results/grpo_training_summary.json` shows `"total_steps": 80` and a positive `reward_delta`.
- [ ] `ls results/grpo_*_curve.png` returns 3 PNGs.
- [ ] `./.venv/bin/python scripts/run_final_benchmark.py --policy-dir checkpoints/grpo` writes `benchmark_summary.csv`.
- [ ] `./.venv/bin/python scripts/score_tasks.py --policy checkpoints/grpo --episodes 3` prints a 3-row 0–1 task-grader table.
- [ ] All artefacts make it into `submission/` after `python scripts/build_submission_bundle.py`.

---

*Companion docs:* [`README.md`](README.md) · [`DESIGN.md`](DESIGN.md) · [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md)
