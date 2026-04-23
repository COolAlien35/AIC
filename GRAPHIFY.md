# AIC Repository Graph

## High-Level Architecture

```text
AIC/
├─ app.py
├─ run_hackathon.py / run_training_smoke.py
├─ requirements.txt
├─ README.md / DESIGN.md / report.md / plan.md
│
├─ aic/
│  ├─ agents/
│  │  ├─ orchestrator_agent.py        -> central policy / decision maker
│  │  ├─ db_agent.py                  -> DB specialist
│  │  ├─ infra_agent.py               -> infra specialist
│  │  ├─ app_agent.py                 -> app specialist
│  │  ├─ network_agent.py             -> network specialist
│  │  ├─ security_agent.py            -> security specialist
│  │  ├─ adversarial_agent.py         -> deceptive recommendation source
│  │  ├─ recovery_verifier_agent.py   -> safety gate / verifier
│  │  ├─ root_cause_analyst_agent.py  -> diagnosis
│  │  ├─ knowledge_agent.py           -> runbook retrieval
│  │  ├─ incident_commander_agent.py  -> strategy mode
│  │  ├─ debate_coordinator.py        -> debate / challenge layer
│  │  └─ observability_agent.py       -> telemetry integrity checks
│  │
│  ├─ env/
│  │  ├─ aic_environment.py           -> main OpenEnv-compatible RL environment
│  │  ├─ world_state.py               -> incident state + metric evolution
│  │  ├─ fault_injector.py            -> injects failures
│  │  ├─ schema_drift.py              -> drift corruption layer
│  │  ├─ reward_engine.py             -> reward decomposition / anti-hacking checks
│  │  ├─ lock_manager.py              -> coordination / deadlock simulation
│  │  ├─ counterfactual_simulator.py  -> what-if scoring
│  │  └─ business_impact.py           -> severity / business cost
│  │
│  ├─ schemas/
│  │  ├─ actions.py                   -> structured action schema
│  │  ├─ observations.py              -> observation schema
│  │  └─ traces.py                    -> reasoning / trace schema
│  │
│  ├─ training/
│  │  ├─ train.py                     -> baseline rollouts through env
│  │  ├─ generate_sft_data.py         -> SFT data generation
│  │  ├─ run_sft.py                   -> supervised warm start
│  │  ├─ train_grpo.py                -> GRPO / RLVR training path
│  │  ├─ prompting.py                 -> prompt formatting
│  │  ├─ rollout_env.py               -> env-policy bridge
│  │  ├─ modeling_unsloth.py          -> optional Unsloth loading
│  │  ├─ export_model.py              -> export / validation
│  │  ├─ curriculum.py                -> easy -> medium -> hard schedule
│  │  └─ reward_audit.py              -> reward hacking audit loop
│  │
│  ├─ evals/
│  │  ├─ benchmark_suite.py           -> benchmark policies
│  │  ├─ arena.py                     -> arena scoring
│  │  ├─ leaderboard.py               -> leaderboard utilities
│  │  └─ rl_eval.py                   -> held-out RL evaluation hooks
│  │
│  ├─ server/
│  │  └─ env_api.py                   -> FastAPI server exposing reset/step/render
│  │
│  ├─ knowledge/runbooks/             -> retrieval corpus
│  └─ utils/                          -> constants, logging, seeding, helpers
│
├─ dashboard/
│  ├─ app.py                          -> Streamlit dashboard
│  ├─ components/                     -> impact, topology, debate, leaderboard, etc.
│  └─ assets/                         -> trajectories, styles, comparison CSVs
│
├─ scripts/
│  ├─ run_episode.py                  -> single episode demo
│  ├─ run_final_benchmark.py          -> benchmark runner
│  ├─ run_env_server.py               -> local FastAPI launcher
│  ├─ benchmark_untrained.py          -> frozen baseline
│  ├─ pre_cache_demo.py               -> dashboard cache prep
│  └─ generate_plots.py               -> reporting visuals
│
├─ artifacts/
│  ├─ sft/                            -> generated SFT dataset
│  └─ grpo/                           -> GRPO prompt dataset
│
├─ checkpoints/
│  ├─ sft/                            -> SFT outputs
│  └─ grpo/                           -> GRPO outputs
│
├─ logs/                              -> episode summaries / reward curves / arena logs
├─ results/                           -> plots + before/after demo artifacts
├─ deploy/                            -> Dockerfile + deploy instructions
└─ tests/                             -> env, reward, parser, eval, training tests
```

## Dependency Flow Graph

```text
Specialist Agents ─┐
Adversarial Agent ─┼─> Orchestrator Agent ──> Structured Action
Knowledge/Analysis ┘                              │
                                                   v
                                           AICEnvironment
                                      ┌────────┼─────────┐
                                      v        v         v
                                 WorldState  Verifier  RewardEngine
                                      │                    │
                                      └──────> Traces <────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     v                             v                             v
                 SFT Data                    GRPO / RLVR                    Dashboard / API
          (generate_sft_data.py)           (train_grpo.py)               (Streamlit + FastAPI)
```

## Suggested Reuse

- Use this file in docs/reviews to explain the repo structure quickly.
- Convert it into Mermaid later if you want a rendered diagram in GitHub.
- Keep it updated whenever new top-level modules or training stages are added.