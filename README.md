<p align="center">
  <h1 align="center">🚨 Adaptive Incident Choreographer (AIC)</h1>
  <p align="center"><strong>The Autonomous Incident War Room</strong></p>
  <p align="center">
    Multi-Agent Trust Calibration Under Adversarial Conditions<br>
    <em>From alert to resolution — with mathematical guarantees</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Agents-6_Specialists-blue" alt="Agents">
  <img src="https://img.shields.io/badge/Scenarios-6_Brutal-red" alt="Scenarios">
  <img src="https://img.shields.io/badge/Safety-Recovery_Verifier-green" alt="Safety">
  <img src="https://img.shields.io/badge/RAG-Runbook_Retrieval-purple" alt="RAG">
  <img src="https://img.shields.io/badge/Tests-166_Passing-brightgreen" alt="Tests">
</p>

---

## 🎯 The Pitch

AIC is an **Autonomous Incident War Room** that orchestrates multiple AI agents to resolve cascading production failures before SLA timers expire — while an adversarial agent actively sabotages recovery.

Unlike static runbook automation, AIC:
- **Thinks before acting** — Root cause analysis → Knowledge retrieval → Counterfactual simulation → Safety verification
- **Learns who to trust** — Dynamic trust calibration detects and suppresses the adversarial agent within 3 steps
- **Never takes unsafe actions** — A deterministic Recovery Verifier gates every action with risk scoring and blast radius analysis
- **Explains every decision** — Full reasoning chain: Hypothesis → Evidence → Simulation → Verification, logged as structured JSONL

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR — "Thinking" Loop                │
│                                                                  │
│   ① HYPOTHESIZE  →  ② RETRIEVE  →  ③ SIMULATE  →  ④ VERIFY    │
│   RootCauseAnalyst   KnowledgeAgent  Counterfactual   Recovery  │
│   (Bayesian)         (RAG/Runbooks)  Simulator        Verifier  │
│                                                                  │
│   Trust Scores: [db:0.85] [infra:0.78] [app:0.82] [adv:0.12]  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ selects best verified action
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────────┐
        │ DB Agent │ │Infra Agent│   │Network Agent │
        │ (trust)  │ │ (trust)   │   │ (trust)      │
        └──────────┘ └──────────┘    └──────────────┘
        ┌──────────┐ ┌──────────┐    ┌──────────────┐
        │App Agent │ │Security  │    │ Adversarial  │
        │ (trust)  │ │ Agent    │    │ Agent 🎭     │
        └──────────┘ └──────────┘    └──────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────────────────────────────────────┐
        │         SERVICE TOPOLOGY (DAG)            │
        │   Gateway → App → Cache/Queue → DB       │
        │   (causal propagation with coupling)      │
        └──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────────┐
        │  World   │ │ Scenario │    │  Business    │
        │  State   │ │ Registry │    │  Impact      │
        │ (12 KPIs)│ │ (6 brutal│    │  Layer       │
        └──────────┘ └──────────┘    └──────────────┘
```

## 🏆 Bonus Prize Mapping

| Prize | Requirement | AIC Implementation |
|-------|------------|-------------------|
| **Fleet AI** (Scalable Oversight) | Multi-agent coordination with safety guarantees | 6 specialist agents + RecoveryVerifier safety gate + max-3-veto cascade with deadlock prevention |
| **Halluminate** (Adversary Discovery) | Detect and handle adversarial/hallucinated outputs | Adversarial agent detection via trust calibration + Root Cause Analyst with Bayesian hypothesis ranking |
| **Patronus AI** (Safety & Eval) | Enterprise safety guardrails and evaluation | Benchmark suite with 3 baselines × 6 scenarios + 0% unsafe action rate via deterministic verifier |
| **Scaler AI Labs** (Enterprise RAG) | Knowledge retrieval with hallucination prevention | KnowledgeAgent with keyword RAG over 6 runbooks + confidence threshold (returns "No match" if < 0.3) |

## 📊 Benchmark Results

AIC is evaluated against 3 baselines across 6 brutal scenarios:

| Policy | Avg MTTR | Adversary Suppression | Unsafe Rate | Revenue Impact |
|--------|----------|----------------------|-------------|----------------|
| **AIC (Trained)** | **Fastest** | **Highest** | **0.0%** | **Highest** |
| AIC (Untrained) | Moderate | Low | 0.0% | Moderate |
| HighestConfidenceOnly | Slow | None | High | Low |
| MajorityVote | Slow | None | High | Low |
| NoTrustOrchestrator | Slow | None | High | Low |

Run the benchmark:
```bash
python scripts/run_final_benchmark.py
```

## 🧪 The 6 Brutal Scenarios

| # | Scenario | Root Cause | Telemetry Corruption |
|---|----------|------------|---------------------|
| 0 | Cache Stampede | Cache TTL alignment | None |
| 1 | Canary Failure | Buggy deployment | NaN blackout on error_rate |
| 2 | Regional Outage | AZ failure | NaN blackout on net_io |
| 3 | Schema Migration | Botched migration | Field rename: db_latency_ms → db_latency |
| 4 | Credential Compromise | Leaked API keys | NaN blackout on sla_compliance |
| 5 | Queue Cascade | Consumer rebalance | Unit shift on queue_depth |

## 🛡️ Safety Guarantees

The **Recovery Verifier Agent** provides deterministic safety:

1. **Risk Gate**: Actions with `risk_score > 0.8` are **always vetoed**
2. **Blast Radius Gate**: Actions with `blast_radius="high"` require a `rollback_plan`
3. **Cascade Protection**: Up to 3 veto attempts before falling back to "Wait and Observe"
4. **Audit Trail**: Every veto is logged with reasoning in the ExplanationTrace

## 🧠 The "Thinking" Loop

Every orchestrator decision follows a 5-step reasoning chain:

```
Step 1: HYPOTHESIZE — RootCauseAnalyst updates Bayesian beliefs over 6 scenarios
Step 2: RETRIEVE    — KnowledgeAgent searches runbooks for matching remediation
Step 3: SIMULATE    — CounterfactualSimulator previews top 3 actions with coupling
Step 4: SELECT      — Re-rank candidates by simulated impact score
Step 5: VERIFY      — RecoveryVerifier gates action with risk/blast analysis
```

All 5 steps are logged in the `ExplanationTrace` for full auditability.

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run a Single Episode
```bash
python scripts/run_episode.py --no-llm
```

### Run the Benchmark Suite
```bash
python scripts/run_final_benchmark.py
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
AIC/
├── aic/
│   ├── agents/
│   │   ├── orchestrator_agent.py    # Lead orchestrator with Thinking loop
│   │   ├── db_agent.py              # Database specialist
│   │   ├── infra_agent.py           # Infrastructure specialist
│   │   ├── app_agent.py             # Application specialist
│   │   ├── network_agent.py         # Network specialist (Phase 9)
│   │   ├── security_agent.py        # Security specialist (Phase 9)
│   │   ├── adversarial_agent.py     # The saboteur 🎭
│   │   ├── recovery_verifier_agent.py # Safety gate (Phase 9)
│   │   ├── knowledge_agent.py       # RAG over runbooks (Phase 10)
│   │   └── root_cause_analyst_agent.py # Bayesian root cause (Phase 10)
│   ├── env/
│   │   ├── world_state.py           # 12-metric production environment
│   │   ├── service_topology.py      # DAG with causal propagation
│   │   ├── scenario_registry.py     # 6 brutal scenarios
│   │   ├── business_impact.py       # Revenue/SLA impact modeling
│   │   ├── counterfactual_simulator.py # Action preview sandbox (Phase 10)
│   │   └── reward_engine.py         # R1-R4 reward decomposition
│   ├── evals/
│   │   └── benchmark_suite.py       # 3 baselines × 6 scenarios (Phase 11)
│   ├── knowledge/
│   │   └── runbooks/                # 6 incident runbooks (Phase 10)
│   ├── schemas/
│   │   └── traces.py                # Pydantic models for traces
│   └── training/
│       └── train.py                 # Episodic training loop
├── dashboard/
│   ├── app.py                       # Streamlit War Room
│   └── components/
│       ├── topology_viz.py          # Live topology map (Phase 11)
│       └── impact_viz.py            # Revenue & timeline (Phase 11)
├── tests/
│   ├── test_adversarial_agent.py
│   ├── test_reward_engine.py
│   ├── test_safety_tier.py          # Phase 9 tests
│   ├── test_intelligence_moat.py    # Phase 10 tests
│   ├── test_topology_scenarios.py   # Phase 8 tests
│   └── test_scaffold.py
├── scripts/
│   ├── run_episode.py               # Single episode runner
│   └── run_final_benchmark.py       # Benchmark suite runner
└── requirements.txt
```

## 📈 Reward Decomposition

| Component | Signal | Range | Type |
|-----------|--------|-------|------|
| **R1** (Health) | Distance from target metrics | per-step | Outcome |
| **R2** (SLA) | Bonus/penalty at episode end | terminal | Outcome |
| **R3** (Trust) | Correct override vs blind trust | +15 / -20 | Process |
| **R4** (Explain) | Prediction accuracy + reasoning quality | per-step | Process |
| **R5** (Format) | Action schema compliance | +2 / -5 | Format |
| **R6** (Verifier) | Recovery verifier approval | +3 / -8 | Safety |
| **R7** (Reasoning) | Causal coherence of reasoning trace | [-1, +3] | Process |
| **R8** (Progress) | Delta toward metric targets | [-0.5, +2] | Process |

---

## 🏆 What We Built & Results

### Full Training Pipeline

```
SFT Data Generation → Supervised Fine-Tuning → GRPO Reinforcement Learning → Export → Deploy
     (32 episodes)        (LoRA on Qwen2)         (env-as-reward)          (merge)   (HF Space)
```

**Pipeline Components:**

| Stage | Script | Status | Output |
|-------|--------|--------|--------|
| SFT Data | `generate_sft_data.py` | ✅ 640 records | `artifacts/sft/orchestrator_sft.jsonl` |
| SFT Training | `run_sft.py` | ✅ LoRA checkpoint | `checkpoints/sft/` |
| GRPO Training | `train_grpo.py` | ✅ Ready | `checkpoints/grpo/` |
| Model Export | `eval/test_export.py` | ✅ Validated | `checkpoints/exported/` |
| Deployment | `deploy/` | ✅ Dockerfile + guide | HF Space ready |

### Curriculum Learning

The `CurriculumScheduler` (`aic/training/curriculum.py`) implements progressive difficulty:

| Tier | Fault Modes | SLA Steps | Features |
|------|-------------|-----------|----------|
| **Easy** | cascading_failure only | 30 | Basic agents, no drift |
| **Medium** | +memory_leak, +db_saturation | 20 | All agents, standard SLA |
| **Hard** | +network_storm | 15 | Schema drift, tough adversary |

### Reward Hacking Protection

The `RewardAuditLoop` (`aic/training/reward_audit.py`) provides:
- **Repeated action detection**: Flags agents exploiting the same action pattern
- **Reward spike detection**: Catches reward without state change
- **Wall-clock + step timeouts**: Prevents infinite episodes
- **Post-episode clamping**: Zeros rewards for flagged episodes

### Process-Aware Feedback (R7 + R8)

Two new modular reward components in `reward_engine.py`:
- **R7 (Reasoning Quality)**: Scores causal coherence — does the reasoning reference metrics, use analytical language, and stay consistent with observed changes?
- **R8 (Progress Signal)**: Partial credit for moving metrics toward targets, even if the episode doesn't fully resolve

### Result Artifacts

- [`results/reward_curve.png`](results/reward_curve.png) — Reward curve across training episodes
- [`results/verifier_pass_rate.png`](results/verifier_pass_rate.png) — Verifier approval rate before vs after
- [`results/before_after_demo.md`](results/before_after_demo.md) — Side-by-side episode comparisons

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate SFT data + train (minimal CPU run)
python3 run_hackathon.py all

# Or step by step:
python3 run_hackathon.py plots demo   # Generate result artifacts
python3 run_hackathon.py sft          # Run minimal SFT training
python3 run_hackathon.py grpo         # Run GRPO (GPU recommended)

# Evaluate
python3 scripts/benchmark_untrained.py --episodes 10
python3 eval/test_export.py --source checkpoints/sft

# Deploy to HuggingFace Spaces
# See deploy/deploy_instructions.md
```

---

## ✅ Submission Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| OpenEnv-compliant environment | ✅ | `AICEnvironment` inherits from `openenv.env.Env` |
| Structured actions & observations | ✅ | Pydantic schemas in `aic/schemas/` |
| Multi-component reward (R1–R8) | ✅ | `aic/env/reward_engine.py` |
| SFT data pipeline | ✅ | 640 records in `artifacts/sft/` |
| SFT training run | ✅ | LoRA checkpoint in `checkpoints/sft/` |
| GRPO training setup | ✅ | `aic/training/train_grpo.py` + config |
| Curriculum learning | ✅ | `aic/training/curriculum.py` |
| Reward hacking protection | ✅ | `aic/training/reward_audit.py` |
| Process-aware feedback | ✅ | R7 + R8 in `reward_engine.py` |
| Model export validation | ✅ | `eval/test_export.py` |
| Deployment artifacts | ✅ | `deploy/Dockerfile` + instructions |
| Reward curve plot | ✅ | `results/reward_curve.png` |
| Verifier pass rate plot | ✅ | `results/verifier_pass_rate.png` |
| Before/after demo | ✅ | `results/before_after_demo.md` |
| Colab notebook | ✅ | `train_colab.ipynb` |
| Interactive demo (Gradio) | ✅ | `app.py` |
| FastAPI server | ✅ | `aic/server/env_api.py` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with</strong> OpenEnv · TRL · Pydantic · Plotly · Streamlit · Gradio
</p>
