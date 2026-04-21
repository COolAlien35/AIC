# 🚨 Adaptive Incident Choreographer (AIC)

**RL-driven multi-agent incident response — trust calibration under adversarial conditions**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/COolAlien35/AIC)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Submission Checklist

### ✅ OpenEnv Usage

`AICEnvironment` inherits from `openenv.env.Env` (OpenEnv v0.1.13), making it fully
OpenEnv-compliant:

```python
from openenv.env import Env as OpenEnvBase

class AICEnvironment(OpenEnvBase):
    def __init__(self, ...):
        super().__init__(
            name="AICEnvironment",
            state_space=state_space,
            action_space=action_space,
            episode_max_length=SLA_STEPS,   # 20 steps
        )

    def reset(self, ...) -> dict:
        ...  # Returns observation dict

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        ...  # Returns (observation, reward, done, info)
```

**File:** [`aic/env/aic_environment.py`](aic/env/aic_environment.py)

---

### ✅ Colab Training Notebook

A Jupyter notebook with PPO training via HuggingFace TRL + LoRA fine-tuning:

👉 **[Open in Colab](https://colab.research.google.com/github/COolAlien35/AIC/blob/main/train_colab.ipynb)**

**File:** [`train_colab.ipynb`](train_colab.ipynb)

**What it does:**
1. Installs all dependencies (openenv, trl, transformers, peft)
2. Loads the `AICEnvironment` and verifies OpenEnv inheritance
3. Initializes `PPOTrainer` with `Qwen/Qwen2-0.5B-Instruct` + LoRA (r=8, α=32)
4. Runs episodic PPO training using R1 health-recovery reward
5. Saves model checkpoint and plots reward curve

---

### ✅ Blog / Video

👉 **[Hugging Face Blog Post](https://huggingface.co/blog/COolAlien35/adaptive-incident-choreographer)**

---

### ✅ Hugging Face Space

👉 **[Live Demo on HF Spaces](https://huggingface.co/spaces/COolAlien35/AIC)**

**File:** [`app.py`](app.py) — Gradio-based interactive demo of the AIC environment

---

## 🏗️ Project Description

AIC simulates a **cascading production incident** where multiple services degrade
simultaneously. A team of specialist AI agents (DB, Infrastructure, Application)
analyze their metric slices and propose remediation actions. An **adversarial agent**
injects plausible-but-wrong recommendations 50% of the time. An **orchestrator agent**
must decide who to trust, which actions to take, and explain its reasoning — all under
a 20-step SLA deadline.

### Key Features

| Feature | Description |
|---|---|
| **Multi-Agent Architecture** | DB, Infra, App specialist agents + adversarial agent |
| **Trust Calibration** | Bayesian trust scores updated per-step based on outcomes |
| **Schema Drift** | Silent API contract changes mid-episode (field rename, unit shift, null injection) |
| **4-Component Reward** | R1 health recovery · R2 SLA bonus · R3 trust calibration · R4 explanation quality |
| **Deadlock Detection** | Resource lock manager with timeout-based deadlock resolution |
| **Causal Coupling** | DB→App latency lag simulates realistic cross-service dependencies |
| **Dashboard** | Streamlit mission-control UI with trust evolution, reward curves, replay |
| **OpenEnv Compliant** | Inherits from `openenv.env.Env` for standard RL integration |

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              Orchestrator Agent                  │
│   (trust scores · action selection · traces)     │
├──────┬──────┬──────┬────────────────────────────┤
│  DB  │Infra │ App  │ Adversarial                │
│Agent │Agent │Agent │ Agent (50% correct)         │
├──────┴──────┴──────┴────────────────────────────┤
│           AICEnvironment (OpenEnv)               │
│  WorldState · FaultInjector · SchemaDrift        │
│  LockManager · RewardEngine                      │
└─────────────────────────────────────────────────┘
```

### 12-Metric World State

The environment tracks 12 real-time operational metrics across three service layers:

| Layer | Metrics | Targets |
|---|---|---|
| **Database** | `db_latency_ms`, `conn_pool_pct`, `replication_lag_ms` | 50ms, 60%, 10ms |
| **Infrastructure** | `cpu_pct`, `mem_pct`, `pod_restarts`, `net_io_mbps` | 45%, 60%, 0, 100 MB/s |
| **Application** | `error_rate_pct`, `p95_latency_ms`, `queue_depth` | 0.5%, 200ms, 50 |
| **Global** | `throughput_rps`, `sla_compliance_pct` | 1000 rps, 99.9% |

All metrics start in a **severely degraded state** (e.g. db_latency at 850ms, error rate at 18.5%) and the system must recover within 20 steps.

### Reward System

| Component | Signal | Type | Range |
|---|---|---|---|
| **R1** | Health recovery (metrics → targets) | Dense, every step | Weighted by layer |
| **R2** | SLA completion bonus (early = more) | Sparse, episode end | 0 – 50 |
| **R3** | Trust calibration (override adversary correctly) | Per interaction | -20 to +15 |
| **R4** | Explanation quality (prediction + causal reasoning) | Delayed 2 steps | -5 to +5 |

### Adversarial Agent Design

The adversarial agent is **structurally indistinguishable** from honest agents — same output format, same confidence range, same metric targeting. It is correct exactly 50% of the time (deterministic per-episode schedule), making trust calibration a genuine challenge.

6 counterfactual templates each recommend an action that would be **correct for a different failure mode**, creating plausible but harmful advice for the current scenario.

### Schema Drift Injection

Mid-episode, the environment silently alters API contracts:
- **Field rename:** `p95_latency_ms` → `p95_latency`
- **Unit shift:** `replication_lag_ms` reports in seconds instead of ms (÷1000)
- **Silent null:** `conn_pool_pct` returns `None` for 3 consecutive steps

The orchestrator must detect and handle these anomalies.

---

## 🚀 Installation

### Prerequisites

- Python ≥ 3.11
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/COolAlien35/AIC.git
cd AIC

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# (Optional) Set up Anthropic API key for LLM mode
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## 🏃 How to Run Locally

### Run a Single Episode (CLI)

```bash
# Rule-based mode (no API key needed)
python scripts/run_episode.py --no-llm

# LLM mode (requires ANTHROPIC_API_KEY)
python scripts/run_episode.py --episode 0
```

### Run Training Loop

```bash
# 100-episode simulation with checkpoints
python -m aic.training.train --num_episodes 100

# Quick 5-episode test
python -m aic.training.train --num_episodes 5
```

### Run Benchmarks

```bash
# Benchmark untrained (frozen trust) agent
python scripts/benchmark_untrained.py --episodes 20

# Pre-cache all demo data for dashboard
python scripts/pre_cache_demo.py --episodes 10
```

### Launch Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### Launch Gradio Demo (HF Space)

```bash
python app.py
# Opens at http://localhost:7860
```

### Run Tests

```bash
python -m pytest -q
# 147 passed
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openenv` | 0.1.13 | OpenEnv-compliant base environment |
| `torch` | 2.2.2 | PyTorch for model training |
| `transformers` | 4.40.2 | HuggingFace model loading |
| `trl` | 0.8.6 | PPO trainer for RL fine-tuning |
| `gymnasium` | 0.29.1 | RL environment utilities |
| `pydantic` | 2.7.1 | Data validation and schemas |
| `streamlit` | 1.35.0 | Dashboard UI |
| `plotly` | 5.22.0 | Interactive charts |
| `numpy` | 1.26.4 | Numerical computation |
| `anthropic` | 0.28.0 | Claude API for LLM agents |
| `rich` | 13.7.1 | CLI visualization |
| `pandas` | 2.2.2 | Data analysis |
| `gradio` | ≥4.0.0 | HF Space demo UI |
| `pytest` | 8.2.0 | Testing framework |

---

## 📁 Project Structure

```
AIC/
├── aic/
│   ├── agents/                # Specialist + adversarial + orchestrator agents
│   │   ├── base_agent.py      # Abstract base class
│   │   ├── db_agent.py        # Database specialist (LLM + rule-based)
│   │   ├── infra_agent.py     # Infrastructure specialist
│   │   ├── app_agent.py       # Application specialist
│   │   ├── adversarial_agent.py  # 50% correct counterfactual agent
│   │   └── orchestrator_agent.py # Lead decision-maker with trust tracking
│   ├── env/                   # Environment mechanics
│   │   ├── aic_environment.py # OpenEnv-compliant wrapper
│   │   ├── world_state.py     # 12-metric simulation core
│   │   ├── fault_injector.py  # 4 fault modes with decay
│   │   ├── schema_drift.py    # 3 drift types (rename, unit, null)
│   │   ├── lock_manager.py    # Deadlock detection and penalties
│   │   └── reward_engine.py   # R1/R2/R3/R4 reward components
│   ├── schemas/               # Pydantic v2 data contracts
│   │   ├── observations.py    # Per-agent observation models
│   │   └── traces.py          # ExplanationTrace, SubAgentRecommendation
│   ├── training/              # Training loop and configuration
│   │   ├── config.py          # All hyperparameters (model, PPO, LoRA)
│   │   └── train.py           # Episodic rollout with reward logging
│   └── utils/                 # Shared utilities
│       ├── constants.py       # All magic numbers centralized
│       ├── seeding.py         # Deterministic episode-level RNG
│       └── logging_utils.py   # JSONL step logging + episode summaries
├── dashboard/                 # Streamlit mission-control UI
│   ├── app.py                 # 3-column dashboard with trust, rewards, replay
│   └── assets/                # CSS, cached trajectories
├── scripts/                   # CLI workflows
│   ├── run_episode.py         # Rich-formatted single episode runner
│   ├── benchmark_untrained.py # Frozen-trust baseline benchmark
│   └── pre_cache_demo.py      # Generate all dashboard data
├── tests/                     # 147 automated tests
├── app.py                     # Gradio HF Space demo
├── train_colab.ipynb          # Colab training notebook (PPO + LoRA)
├── requirements.txt           # Pinned dependencies (incl. openenv, gradio)
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

---

## 🔬 How It Works

### Episode Flow

1. **Initialize** — 12 metrics start in severely degraded state
2. **Fault injection** — cascading failure applies ongoing drift (decays over time)
3. **Observation slicing** — each agent sees only its service's metrics
4. **Schema drift** — mid-episode, API contracts silently change
5. **Recommendations** — 4 agents (3 honest + 1 adversarial) propose actions
6. **Orchestration** — orchestrator picks an action, tracks trust, emits explanation trace
7. **World update** — action deltas + fault drift + noise + causal lag
8. **Reward** — R1 (health) + R3 (trust calibration) + R4 (explanation quality) per step
9. **Repeat** for 20 steps, then R2 (SLA bonus) at episode end

### Trust Update Rule

```
trust_new = (1 - α) × trust_old + α × outcome_score
```

Where `α = 0.1` (TRUST_UPDATE_RATE) and `outcome_score` is 1.0 if metrics improved toward targets, 0.0 otherwise.

### Causal Coupling

DB connection pool changes propagate to application latency with a 2-step lag:

```
p95_latency(t+2) += 0.4 × Δconn_pool_pct(t)
```

This creates realistic cross-service dependencies that agents must reason about.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Pulkit Pandey
