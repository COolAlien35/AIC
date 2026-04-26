---
title: AIC OpenEnv Environment
emoji: "\U0001F6A8"
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Judge env server (OpenEnv + FastAPI)
tags:
  - openenv
  - rlve
  - reinforcement-learning
  - multi-agent
  - incident-response
  - trl
  - grpo
  - unsloth
---

# AIC — Adaptive Incident Choreographer (OpenEnv environment)

This Space is the **canonical OpenEnv environment server** for the AIC project.
Judges should pull this URL to evaluate the environment.

> Companion docs in the source repo: [`README.md`](https://github.com/COolAlien35/AIC/blob/main/README.md) ·
> [`DESIGN.md`](https://github.com/COolAlien35/AIC/blob/main/DESIGN.md) ·
> [`openenv.yaml`](https://github.com/COolAlien35/AIC/blob/main/openenv.yaml).

## TL;DR (what judges should do)

1. Confirm liveness: `GET /health`
2. Start an episode: `POST /reset` → get `env_id`
3. Inspect full state: `GET /state/{env_id}`

This Space is **environment-only** (FastAPI). Training happens in Colab; results are linked below.

## What this is

AIC is a multi-agent, adversarial incident-response environment. **Six specialist
agents** (DB, infra, app, network, security, plus an adversary that lies on a fixed
seed-derived cycle) propose actions every step. A deterministic
[**Recovery Verifier**](https://github.com/COolAlien35/AIC/blob/main/aic/agents/recovery_verifier_agent.py)
gates execution by risk score and blast radius. The world state evolves through a
12-KPI service-topology DAG with cascading faults across **six brutal scenarios**.

The action space is a structured `OrchestratorDecision` JSON; the observation includes
candidate recommendations, current metrics, trust scores, scenario metadata, and
schema-drift state. The reward is a **9-component verifiable reward function (R1–R9)**
plus an over-confidence penalty (R9) and a terminal SLA-met bonus (R2).

**Why this is hard (and why “standard AI” fails):**
- incidents are **multi-step** (20-step SLA budget)
- telemetry can be **corrupted** (NaNs / field renames / unit shifts)
- actions must be **safe** (verifier-gated)
- one agent is **adversarial** (plausible but destructive advice)

## API

| Method | Endpoint              | Purpose                            |
|--------|-----------------------|------------------------------------|
| GET    | `/health`             | Liveness                           |
| POST   | `/reset`              | Start a new episode, get `env_id`  |
| POST   | `/step`               | Submit an `OrchestratorDecision`   |
| GET    | `/state/{env_id}`     | Full structured environment state  |
| GET    | `/render/{env_id}`    | ANSI render                        |
| DELETE | `/env/{env_id}`       | Free resources                     |

OpenEnv manifest: [`openenv.yaml`](https://github.com/COolAlien35/AIC/blob/main/openenv.yaml).
The class is `aic.env.aic_environment.AICEnvironment`, which subclasses `openenv.env.Env` directly.

## Quick smoke test

```bash
HOST="https://kingkk007-aic-training.hf.space"

# liveness
curl -s "$HOST/health"

# reset → step → state
ENV_ID=$(curl -sX POST "$HOST/reset" \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":0,"base_seed":42,"fault_mode":"cascading_failure"}' | jq -r .env_id)

curl -s "$HOST/state/$ENV_ID" | \
  jq '.state | {step, scenario_name, health_score, is_within_sla, sla_remaining_steps}'
```

## Evidence of real training (GRPO)

These artifacts are committed in the GitHub repo and generated from the raw training log:

- Reward curve: `results/grpo_reward_curve.png`
- Loss curve: `results/grpo_loss_curve.png`
- KL curve: `results/grpo_kl_curve.png`
- Raw per-step log: `logs/grpo_progress.jsonl`

The headline story is simple: **reward improves from −15.10 → −10.24 in 80 GRPO steps** on a Colab T4.

## Tasks (3, with deterministic 0.0–1.0 graders)

| ID                      | Difficulty | Success threshold | Grader file                                                                                        |
|-------------------------|------------|------------------:|----------------------------------------------------------------------------------------------------|
| `db_pool_recovery`      | easy       |             0.60  | [`aic/tasks/task_db_pool_recovery.py`](https://github.com/COolAlien35/AIC/blob/main/aic/tasks/task_db_pool_recovery.py)             |
| `canary_blackout`       | medium     |             0.55  | [`aic/tasks/task_canary_blackout.py`](https://github.com/COolAlien35/AIC/blob/main/aic/tasks/task_canary_blackout.py)               |
| `adversarial_misroute`  | hard       |             0.50  | [`aic/tasks/task_adversarial_misroute.py`](https://github.com/COolAlien35/AIC/blob/main/aic/tasks/task_adversarial_misroute.py)     |

Every grader is a pure function `EpisodeTrace → float ∈ [0, 1]`.

## Scenarios bundled

| ID | Scenario name | Hard part |
|---:|---|---|
| 0 | DB pool recovery from cache stampede | Cascading retry storm pollutes telemetry |
| 1 | Canary failure recovery during telemetry blackout | NaN-ed metrics for 3 steps mid-episode |
| 2 | Regional outage with split-brain | Conflicting health signals across regions |
| 3 | Adversarial misrouting during DB schema migration | Adversary lies persistently + drift renames a field |
| 4 | Credential compromise + service degradation | Security-vs-availability trade-off |
| 5 | Schema migration disaster | Field rename + unit shift + NaN blackout |

## Project links

- **Source code (canonical):** https://github.com/COolAlien35/AIC
- **2-minute video walkthrough:** linked from the source-repo README
- **Results dashboard (hosted):** https://huggingface.co/spaces/KINGKK007/aic-results-dashboard
- **Judge quickstart notebook:** https://colab.research.google.com/github/COolAlien35/AIC/blob/main/judge_colab.ipynb
- **All-in-one notebook:** https://colab.research.google.com/github/COolAlien35/AIC/blob/main/AIC_all_in_one.ipynb
- **Training notebook:** https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb
- **OpenAI baseline script (judges run with their own key):** [`scripts/openai_baseline.py`](https://github.com/COolAlien35/AIC/blob/main/scripts/openai_baseline.py)
- **Public trained LoRA adapter:** https://huggingface.co/COolAlien35/aic-grpo-adapter-14

## Real GRPO training run (snapshot)

| Metric | Value |
|---|---|
| Algorithm | GRPO (TRL `GRPOTrainer` + Unsloth) |
| Base model | `Qwen2.5-3B-Instruct` (LoRA r=16, α=32) |
| Hardware | NVIDIA T4 16 GB on Google Colab |
| Total steps | **80** |
| Initial reward | −15.10 |
| Final reward | −10.24 |
| Reward delta | **+4.86** |
| Wall-clock | **6.19 hours** (371.3 min) |
| Source log | [`logs/grpo_progress.jsonl`](https://github.com/COolAlien35/AIC/blob/main/logs/grpo_progress.jsonl) |

License: MIT.
