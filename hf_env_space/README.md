---
title: AIC OpenEnv Environment
emoji: "\U0001F6A8"
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Adversarial multi-agent incident-response OpenEnv (FastAPI / Docker)
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

# AIC - Adaptive Incident Choreographer (OpenEnv environment)

This Space is the **canonical OpenEnv environment server** for the AIC project.
Judges should pull this URL to evaluate the environment.

## What this is

AIC is a multi-agent, adversarial incident-response environment. Six specialist
agents propose actions every step; an adversarial agent injects misleading
recommendations; a deterministic Recovery Verifier gates execution by risk
score and blast radius; the world state evolves through a 12-KPI service
topology with cascading faults across six brutal scenarios.

The action space is a structured `OrchestratorDecision` JSON; the observation
includes candidate recommendations, current metrics, trust scores, scenario
metadata, and schema-drift state. The reward is an 8-component
verifiable-reward function (R1-R8) plus an over-confidence penalty (R9) and a
terminal SLA-met bonus (R2).

## API

| Method | Endpoint              | Purpose                            |
|--------|-----------------------|------------------------------------|
| GET    | `/health`             | Liveness                           |
| POST   | `/reset`              | Start a new episode, get `env_id`  |
| POST   | `/step`               | Submit an `OrchestratorDecision`   |
| GET    | `/state/{env_id}`     | Full structured environment state  |
| GET    | `/render/{env_id}`    | ANSI render                        |
| DELETE | `/env/{env_id}`       | Free resources                     |

OpenEnv manifest: see [`openenv.yaml`](./openenv.yaml).

## Quick smoke test

```bash
HOST="https://kingkk007-aic-openenv-env.hf.space"
curl -s "$HOST/health"
ENV_ID=$(curl -sX POST "$HOST/reset" -H 'Content-Type: application/json' \
  -d '{"episode_id":0,"base_seed":42,"fault_mode":"cascading_failure"}' | jq -r .env_id)
curl -s "$HOST/state/$ENV_ID" | jq '.state | {step,scenario_name,health_score,is_within_sla}'
```

## Project links

- **Source code (canonical):** https://github.com/COolAlien35/AIC
- **Live demo (Gradio UI):** https://huggingface.co/spaces/KINGKK007/aic-incident-command-center
- **2-minute video walkthrough:** see top-level README of source repo
- **Real GRPO training run (loss + reward plots):** see `results/` in source repo
- **Colab training notebook:** `train_colab.ipynb` in source repo

## Tasks (3, with deterministic 0.0-1.0 graders)

| ID                      | Difficulty | Grader file                                       |
|-------------------------|------------|---------------------------------------------------|
| `db_pool_recovery`      | easy       | `aic/tasks/task_db_pool_recovery.py`              |
| `cache_stampede`        | medium     | `aic/tasks/task_cache_stampede.py`                |
| `adversarial_misroute`  | hard       | `aic/tasks/task_adversarial_misroute.py`          |

License: MIT.
