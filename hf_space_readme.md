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
  - grpo
  - trl
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
and the orchestrator must decide who to trust — all under a 20-step SLA deadline.

## Features

- 🌐 **12-metric world state** with real-time health tracking (DB, infra, app KPIs)
- 🤖 **6 specialist agents** (DB, Infra, App, Network, Security, Adversarial)
- 🎭 **Adversarial recommendations** — cycle of lie / partial truth / truth, seeded per episode
- 📊 **8-component reward function** (R1 health · R2 SLA · R3 calibrated trust · R4 explanation · R5 format · R6 verifier · R7 reasoning · R8 progress) **+ R9 over-confidence penalty**
- ⚡ **6 brutal scenarios** — cache stampede, canary blackout, regional outage, adversarial misroute, credential compromise, schema migration disaster
- 🔐 **OpenEnv-compliant** — `aic.env.aic_environment.AICEnvironment` subclasses `openenv.env.Env` directly
- 🎯 **3 deterministic 0.0–1.0 task graders** (`db_pool_recovery` · `canary_blackout` · `adversarial_misroute`)

## How to use the demo

1. Set episode ID, seed, and fault mode.
2. Click **Create & Reset**.
3. Type an action or click **Observe**.
4. Watch the 12 KPIs evolve, trust scores recalibrate, and the verifier accept/veto each step.

## Real GRPO training (linked from source repo)

| Metric | Value |
|---|---|
| Algorithm | GRPO (TRL `GRPOTrainer` + Unsloth) |
| Base model | `Qwen2.5-3B-Instruct` (LoRA r=16, α=32) |
| Hardware | NVIDIA T4 16 GB on Colab |
| Total steps | **80** |
| Reward delta | **−15.10 → −10.24** (+4.86) |
| Wall-clock | **6.19 hours** |

Source: [`results/grpo_training_summary.json`](https://github.com/COolAlien35/AIC/blob/main/results/grpo_training_summary.json).

## Links

- **OpenEnv environment server (judges pull this):** https://huggingface.co/spaces/KINGKK007/aic-openenv-env
- **GitHub repository (source of truth):** https://github.com/COolAlien35/AIC
- **Engineering design doc:** https://github.com/COolAlien35/AIC/blob/main/DESIGN.md
- **Colab training notebook:** https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb
- **Video walkthrough script (record-ready):** https://github.com/COolAlien35/AIC/blob/main/VIDEO_SCRIPT.md
