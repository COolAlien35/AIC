# FINAL SUBMISSION — Adaptive Incident Choreographer (AIC)

## 1) Problem statement

Modern production incidents are multi-causal, fast-moving, and noisy. A single static playbook often fails under cascading faults, telemetry drift, and adversarial or low-quality recommendations.  
AIC addresses this by simulating incident response as a structured control problem where an orchestrator must recover system health while maintaining safety and auditability.

## 2) Environment design

AIC models a degraded production stack with:

- Multi-service metric state and causal coupling
- Fault injection across multiple incident modes
- Schema drift and telemetry corruption
- Specialist recommendation agents plus an adversarial source
- Step-wise episode rollouts with structured traces

This creates an environment where policies are evaluated on recovery quality, robustness, and safety rather than only raw reward.

## 3) Reward and verifier design

Reward includes recovery and process-aware components (health, SLA outcomes, trust calibration, reasoning quality, and progress signals).  
A deterministic Recovery Verifier gates unsafe actions before they are applied:

- Risk gate (high-risk actions are vetoed)
- Blast-radius checks (rollback requirements for risky operations)
- Fallback behavior after repeated vetoes
- Full decision trace logging for auditability

## 4) Baseline vs adaptive results (real run artifacts)

From the regenerated benchmark evidence:

- `baseline_frozen_trust`: avg total reward `-287.4086`, avg final health `0.2458`
- `baseline_adaptive_trust`: avg total reward `-291.6119`, avg final health `0.2332`
- Episodes evaluated: `3` per policy (`10000..10002`)
- Success rate: `0.0` in this short proof benchmark

Interpretation: this run demonstrates reproducible policy differences and full evidence plumbing; it is not claimed as large-scale convergence.

## 5) Safeguards against reward hacking

- Deterministic verifier gating blocks unsafe high-risk actions
- Structured traces support post-hoc auditing of action justification
- Benchmark outputs are generated from executed runs (no synthetic projection in final evidence pass)
- Consistency checks were run between JSONL logs, summary CSV, and demo markdown values

## 6) Mac-safe reproducibility commands

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_hackathon.py verify plots demo
./.venv/bin/python run_hackathon.py sft
```

Expected key outputs:

- `results/reward_curve.png`
- `results/verifier_pass_rate.png`
- `results/before_after_demo.md`
- `logs/eval/policy_benchmark.jsonl`
- `results/benchmark_summary.csv`
- `checkpoints/sft/sft_metadata.json`
- `results/evidence_manifest.json`
- `results/evidence_manifest.md`

## 7) Remote deployment proof

Live Space URL:

- [https://huggingface.co/spaces/KINGKK007/aic-incident-command-center](https://huggingface.co/spaces/KINGKK007/aic-incident-command-center)

## 8) Optional future GPU/GRPO path

The repository includes SFT/GRPO training paths for model-scale runs.  
For judge-safe reproducibility on Mac CPU, the final proof uses benchmark + plotting + demo + minimal SFT smoke training.  
Future work is to run longer GPU-backed GRPO and report held-out uplift at larger scale.

## 9) Artifact hosting note

To keep the repository lightweight and reviewable, large generated artifacts are hosted externally and not committed as source files:

- `checkpoints/grpo/` (generated training checkpoint)
- `exports/` (generated export output)

Hosted artifacts:

- GRPO checkpoint bundle: [Google Drive (grpo)](https://drive.google.com/drive/folders/1RJcu7AWuEDmLBhUYMbikPRtOGAu1sTHD?usp=share_link)
- Export bundle: [Google Drive (exports)](https://drive.google.com/drive/folders/1PjW-gbnr-RtPg_qk5fFXu5Zz1N-uGfJo?usp=share_link)

Regeneration commands:

```bash
./.venv/bin/python run_hackathon.py grpo
./.venv/bin/python eval/test_export.py --source checkpoints/grpo
```

Suggested deferred proof commands (GPU):

```bash
./.venv/bin/python run_hackathon.py grpo
./.venv/bin/python scripts/run_final_benchmark.py
./.venv/bin/python eval/test_export.py --source checkpoints/grpo
```
