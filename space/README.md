---
title: AIC Training (Private)
emoji: 🛰️
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
hardware: t4-medium
private: true
pinned: true
short_description: Adaptive Incident Choreographer SFT + GRPO training Space
---

# AIC Training Space

Private JupyterLab Space for training the Adaptive Incident Choreographer
on Qwen2.5-3B-Instruct with LoRA + 4-bit + GRPO. The Space starts a
JupyterLab server that clones the GitHub repo on first boot and lets you
run the smoke + full pipelines from the notebooks under `/workspace/aic-repo/space/notebooks/`.

## Boot sequence

1. Build pulls the CUDA 12.1 base image and installs JupyterLab.
2. `start.sh` clones `https://github.com/COolAlien35/AIC.git` into
   `/workspace/aic-repo` and `pip install`s requirements + the package.
3. JupyterLab listens on port 7860 (the public Space URL).

## Required Space secrets

| Name | Purpose |
|------|---------|
| `JUPYTER_TOKEN` | The token you'll type in the JupyterLab login prompt |
| `HF_TOKEN` (optional) | If you later want to push checkpoints to the Hub |

## How to use

After the Space boots, open the URL → JupyterLab prompts for the token →
open a Terminal and run:

```bash
cd /workspace/aic-repo
git pull origin main          # always pick up latest fixes
nvidia-smi                    # confirm T4 visible
python scripts/run_pipeline.py verify
python scripts/run_pipeline.py smoke   # ~8 minutes, must pass before --full
python scripts/run_pipeline.py full    # ~2.5 hours
```

Or open the notebooks under `space/notebooks/` and run them top-to-bottom.

## Note on persistence

This Space does NOT mount paid persistent storage. Everything under
`/workspace` is wiped when the container restarts. Use Notebook 02 to
zip and download the artifacts before stopping the Space.
