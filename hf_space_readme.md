---
title: Adaptive Incident Choreographer
emoji: 🚨
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
short_description: Multi-agent trust calibration under adversarial conditions
tags:
  - reinforcement-learning
  - multi-agent
  - incident-response
  - openenv
  - trust-calibration
---

# 🚨 Adaptive Incident Choreographer (AIC) — Live Demo

**Multi-Agent Trust Calibration Under Adversarial Conditions**

> **Note for hackathon judges:** this Space is the **interactive Gradio demo**.
> The canonical **OpenEnv environment server** that judges should pull/evaluate
> lives at **[`KINGKK007/aic-openenv-env`](https://huggingface.co/spaces/KINGKK007/aic-openenv-env)**
> (Docker SDK, FastAPI, exposes `/health`, `/reset`, `/step`, `/state/{env_id}`,
> `/render/{env_id}`, `DELETE /env/{env_id}`).

Step through a simulated cascading production incident. Observe how specialist
AI agents propose remediations, an adversarial agent injects misleading advice,
and the orchestrator must decide who to trust — all under an SLA deadline.

## Features

- 🌐 **12-metric world state** with real-time health tracking
- 🤖 **4 specialist agents** (DB, Infra, App, Adversarial)
- 🎭 **Adversarial recommendations** — correct only 50% of the time
- 📊 **4-component reward system** (health, SLA, trust, explanation)
- ⚡ **4 fault modes** — cascading failure, memory leak, DB saturation, network storm
- 🔐 **OpenEnv-compliant** — inherits from `openenv.env.Env`

## How to Use

1. Set episode ID, seed, and fault mode
2. Click **Create & Reset**
3. Type an action or click **Observe**
4. Watch metrics evolve step by step

## Links

- **OpenEnv environment server (judges pull this):** https://huggingface.co/spaces/KINGKK007/aic-openenv-env
- [GitHub Repository](https://github.com/COolAlien35/AIC)
- [Colab Training Notebook](https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb)
- [Blog Post](https://huggingface.co/blog/COolAlien35/adaptive-incident-choreographer)
