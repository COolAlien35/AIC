---
title: AIC Training (Private)
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
hardware: t4-medium
private: true
pinned: true
short_description: Adaptive Incident Choreographer SFT + GRPO training Space
---

# 🛰️ AIC Training Space (Private)

Private JupyterLab Space for training the Adaptive Incident Choreographer on
**Qwen2.5-3B-Instruct** with **LoRA r=16, α=32** + **4-bit NF4** + **GRPO** (TRL `GRPOTrainer` + Unsloth).

> **Two distinct AIC Spaces — don't confuse them:**
>
> | Space | Purpose | Visibility |
> |---|---|---|
> | [`KINGKK007/aic-openenv-env`](https://huggingface.co/spaces/KINGKK007/aic-openenv-env) | Canonical OpenEnv environment server (Docker, FastAPI). Judges pull this. | Public |
> | [`KINGKK007/aic-incident-command-center`](https://huggingface.co/spaces/KINGKK007/aic-incident-command-center) | Interactive Gradio demo. | Public |
> | **This Space** (private training Space) | Owner-only training rig with JupyterLab + GPU. | **Private** |

## Boot sequence

1. Build pulls the CUDA 12.1 base image and installs JupyterLab.
2. `start.sh` clones `https://github.com/COolAlien35/AIC.git` into
   `/workspace/aic-repo` and `pip install`s `requirements.txt` + the package.
3. JupyterLab listens on port **7860** (the public Space URL).

## Required Space secrets

Set these in **Settings → Variables and secrets**:

| Name | Required | Purpose |
|------|---|---|
| `JUPYTER_TOKEN` | ✅ yes | Token typed into the JupyterLab login prompt |
| `HF_TOKEN`      | optional | Push trained adapters to the HF Hub |
| `OPENAI_API_KEY` | optional | Run `scripts/openai_baseline.py` from inside the Space |

## How to use

After the Space boots, open the URL → JupyterLab prompts for the token →
open a Terminal and run:

```bash
cd /workspace/aic-repo
git pull origin main          # always pick up latest fixes
nvidia-smi                    # confirm T4/L4/A100 visible

# 3-stage pipeline
./.venv/bin/python scripts/run_pipeline.py verify   # ~30s — env + GPU sanity
./.venv/bin/python scripts/run_pipeline.py smoke    # ~8 min — 1-step SFT + 2-step GRPO
./.venv/bin/python scripts/run_pipeline.py full     # ~2.5h on L4, ~6.2h on T4
```

The `full` stage is what produced the canonical results we ship in the source repo:

| Metric | Value | File |
|---|---|---|
| GRPO total steps | **80** | `logs/grpo_progress.jsonl` |
| Initial reward | −15.10 | `results/grpo_training_summary.json` |
| Final reward | −10.24 | `results/grpo_training_summary.json` |
| Reward delta | **+4.86** | `results/grpo_training_summary.json` |
| Final loss | 0.0026 | `results/grpo_training_summary.json` |
| Wall-clock | **6.19 h** on T4 | `results/grpo_training_summary.json` |

Or open the notebooks under `space/notebooks/` and run them top-to-bottom for an
interactive walk-through.

## Note on persistence

This Space does **not** mount paid persistent storage. Everything under
`/workspace` is wiped when the container restarts. Use Notebook 02
(`space/notebooks/02_export_and_download.ipynb`) to zip and download artefacts
**before** stopping the Space, or stream them straight to HF Hub:

```bash
./.venv/bin/python eval/test_export.py \
    --source /workspace/aic-repo/checkpoints/grpo \
    --push --hub-repo <your-username>/aic-orchestrator
```

---

*Companion docs:* [`COLAB_GPU_RUNBOOK.md`](https://github.com/COolAlien35/AIC/blob/main/COLAB_GPU_RUNBOOK.md) ·
[`DESIGN.md`](https://github.com/COolAlien35/AIC/blob/main/DESIGN.md) ·
[`README.md`](https://github.com/COolAlien35/AIC/blob/main/README.md)
