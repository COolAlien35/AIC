---
title: AIC Training (Private)
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: l4x4
startup_duration_timeout: 1h
private: true
pinned: true
short_description: Adaptive Incident Choreographer SFT + GRPO training Space (4xL4 DDP)
---

# AIC Training Space

Private JupyterLab Space for training the Adaptive Incident Choreographer
on Qwen2.5-3B-Instruct with LoRA + 4-bit + GRPO. Default clone branch:
`feat/4xl4-ddp-space` (override with Space variable `AIC_REPO_BRANCH`).

## Required Space secrets

| Name | Purpose |
|------|---------|
| `JUPYTER_TOKEN` | Jupyter login (if enabled) |
| `HF_TOKEN` | Recommended: upload `exports/` to Hub for durability |
| `AIC_REPO_BRANCH` | Optional; default `feat/4xl4-ddp-space` |
| `AIC_HUB_REPO` | Optional model repo for upload |

## 4xL4 DDP run order

```bash
cd /workspace/aic-repo
git fetch origin feat/4xl4-ddp-space && git checkout feat/4xl4-ddp-space
python scripts/run_pipeline.py verify
AIC_SFT_MAX_STEPS=20 python scripts/run_pipeline.py smoke
accelerate launch --config_file=space/accelerate_config.yaml scripts/run_pipeline.py full
```

Smoke must **not** use `accelerate launch`. Full training uses 4 processes.

## Persistence

Use `HF_TOKEN` + hub push (non-fatal if missing) or download `exports/` before restart.
