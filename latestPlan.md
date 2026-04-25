# 24-Hour Multi-Agent Execution Plan (Competitive-First)

## Outcome Target
Ship a judge-ready competitive multi-agent environment with:
- Real strategic behavior under partial observability.
- Clear trained-vs-untrained evidence with readable plots.
- OpenEnv + HF Space + reproducible training pipeline.
- Honest, high-impact story aligned to judging weights (40/30/20/10).

## Strategic Principle
Given 24 hours and two HF accounts, optimize for **evidence reliability** over one risky giant run:
- Use parallel medium runs to increase probability of measurable uplift.
- Lock environment/data/reward first, then run training bursts.
- Keep one canonical benchmark path and one canonical claim set.

## Recommended Model + Compute Strategy
- **Primary model:** Qwen/Qwen2.5-3B-Instruct  
- **Fallback model:** Qwen/Qwen2.5-1.5B-Instruct  
- **Last-resort speed model:** Qwen/Qwen2-0.5B-Instruct  
- **Hardware:** prioritize T4 medium for 3B; use T4 small only for calibration/smoke runs  

**Parallelization:**
- Account A: baseline + run set A (seeds/scenarios split 1)  
- Account B: run set B (seeds/scenarios split 2)  

**Budget policy:**
- 15% credits on calibration runs  
- 65% on 3B primary runs across both accounts  
- 20% reserve for final confirmation run + export  

---

## Phase 0 (0–2h): Hard Freeze and Integrity Pass

### 0.1 Canonicalize execution paths
Confirm these are the only judge-facing paths:
- Env server: `/aic/server/env_api.py`
- OpenEnv manifest: `/openenv.yaml`
- Benchmark runner: `/scripts/run_final_benchmark.py`
- Training entry: `/aic/training/train_grpo.py`

- De-emphasize stale/conflicting outputs in docs and presentation.

### 0.2 Ensure trained policy path is real
- In benchmark path, ensure `trained_grpo` uses model checkpoint inference (not heuristic aliasing).
- Keep heuristic orchestrator as explicit separate baseline label.

### 0.3 Reproducibility baseline
- Freeze dependencies and seeds; enforce deterministic seed schedule in all runs.
- Keep one artifact schema in `results/` and one evidence manifest.

---

## Phase 1 (2–6h): Competitive Multi-Agent Environment Upgrade

### 1.1 Competitive mechanics (high impact for 40%)
Implement strategic incentives in environment loop:
- Resource auction / quota bidding layer
- Private observations per agent + noisy shared signals
- Agent utility asymmetry

**Senior-RL requirements:**
- Log `(obs_i, action_i, utility_i)` per step
- Add counterfactual credit tags
- Non-stationarity controls:
  - fixed training curriculum
  - held-out eval seeds
  - explicit train/eval mode switches
- Strategic pressure schedules:
  - early: low adversarial intensity
  - later: higher deception + correlated failures
- Coordination bottlenecks:
  - limited slots
  - action contention penalties
  - delayed effects

**Primary files:**
- `aic/env/aic_environment.py`
- `aic/env/world_state.py`
- `aic/env/scenario_registry.py`
- `aic/agents/*.py`

---

### 1.2 Multi-agent coordination proof
Add episode metrics:
- Coalition quality
- Conflict rate
- Adversary manipulation success rate
- Regret vs oracle-safe action

Diagnostics:
- Influence matrix
- Trust calibration curves
- Outcome attribution
- Mode collapse detector

---

### 1.3 Agent activation matrix
- Ensure all intended agents are active in benchmark
- Emit per-episode agent manifest
- Add test coverage for agent inclusion

---

## Phase 2 (6–9h): Data Generation Quality Upgrade

### 2.1 SFT dataset integrity
- Train/val split by episode IDs
- Scenario balancing
- Per-agent action distribution checks
- Dedup + leakage checks

**Standards:**
- schema-valid completion rate
- duplicate prompt rate
- entropy metrics
- recommendation diversity

- Hard gates:
  - min diversity
  - max duplicate threshold
  - scenario balance

- Include difficult negatives:
  - adversarial bad recommendations
  - conflicting specialist outputs
  - ambiguous states

- Add dataset fingerprint (hash + config)

**Primary files:**
- `aic/training/generate_sft_data.py`
- `aic/training/prompting.py`
- `aic/training/rollout_env.py`

---

### 2.2 Scenario realism fix
- Remove ambiguous label→fault remaps
- Either implement real distinct dynamics OR report taxonomy honestly

**Hardening:**
- Scenario contract table
- Held-out stress scenarios
- Tag samples with metadata:
  - `scenario_name`
  - `fault_mode`
  - `difficulty_tier`
  - `adversarial_intensity`

---

## Phase 3 (9–12h): Reward System Hardening

### 3.1 Competitive rubric composition
- Global recovery utility
- Strategic robustness
- Coordination efficiency
- Belief accuracy
- Safety constraints

**Hardening:**
- Normalize reward scales
- Weight schedule:
  - early: safety + format
  - later: strategic + long-horizon
- Add delayed-return proxies
- Penalize overconfidence + low-information actions
- Track reward drift

**Primary files:**
- `aic/env/reward_engine.py`
- `aic/training/reward_audit.py`

---

### 3.2 Reward gaming tests
Test for:
- no-op farming
- keyword inflation
- stalling
- verifier bypass

**Invariants:**
- unsafe actions never outperform safe ones
- no-op loops ≤ 0 expected return

**Red-team:**
- prompt injection
- confidence spoofing
- oscillation behavior

---

## Phase 4 (12–20h): Parallel Training Campaign

### 4.1 Campaign design
Run 6–8 jobs:
- 2 calibration (short)
- 4 primary (medium)
- 1 reserve

Grid:
- seeds (A/B/C)
- prompt lengths
- GRPO steps
- temperature

---

### 4.2 Success gate
Keep run only if:
- schema-valid actions pass threshold
- verifier improves/stable
- reward uplift vs baseline
- no catastrophic failures

---

### 4.3 Promote best checkpoint
- Select best via held-out benchmark
- Export + lock metadata

---

## Phase 5 (20–22h): Evaluation and Evidence Lock

### 5.1 Canonical benchmark rerun
- frozen baseline
- heuristic baseline
- trained policy

- Use enough episodes to reduce variance

---

### 5.2 Statistical evidence
Generate:
- reward curves
- policy comparison plots
- confidence intervals

**Primary files:**
- `scripts/run_final_benchmark.py`
- `results/benchmark_summary.csv`
- `results/statistical_test.json`

---

## Phase 6 (22–24h): Submission Packaging

### 6.1 Storyline structure
- Problem
- What each agent does
- What changed after training
- Why it matters

---

### 6.2 Required assets
- HF Space + API proof
- Training script
- README + short video
- Evidence manifest

**Docs:**
- `README.md`
- `FINAL_SUBMISSION.md`
- `deploy_instructions.md`

---

## Implementation Sequence (Do Not Deviate)
1. Benchmark integrity fix  
2. Competitive mechanics + full agents  
3. Data integrity + splits  
4. Reward hardening  
5. Parallel training  
6. Evidence + packaging  

---

## High-Risk Pitfalls
- Fake GRPO improvement
- Train/eval mismatch
- Over-polished UI, weak evidence
- Single long run without calibration

---

## Done Criteria (Judge-Ready)
- Competitive multi-agent environment
- All metrics from canonical pipeline
- Trained policy actually used
- Reproducible plots
- Complete submission package

---

## Suggested 2-Account Split
- Account A: seeds 11/13/17 (config A)
- Account B: seeds 19/23/29 (config B)
- Final winner via held-out aggregate

---

## To-dos
- Ensure real checkpoint inference in benchmark  
- Implement competitive mechanics + partial observability  
- Add SFT/GRPO data integrity checks  
- Strengthen reward + exploit tests  
- Run parallel training  
- Package final evidence  