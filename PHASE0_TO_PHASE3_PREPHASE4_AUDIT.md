# Phase 0–3 Pre-Phase-4 Audit

Status: **Audit complete**  
Fixes applied: **None**  
Intent: Identify pre-Phase-4 blockers and define what must be fixed before the Phase 4 training campaign.

---

## Executive Verdict

Phases 0–3 are **not yet clean enough to be treated as Phase-4-ready**.

The codebase foundation is reasonably strong:
- targeted pre-Phase-4 regression suite passed: **190/190 tests**
- current SFT artifacts are present and structurally valid
- current SFT completions inspected from disk had **0 invalid completions**
- train/val files exist and show **0 episode-ID overlap**

However, there are still several **high-severity integration and benchmark-integrity issues** that could directly undermine Phase 4 evidence quality if left unfixed.

The most important blockers are:
1. **Current “trained_grpo” evidence is not backed by a real checkpoint in existing artifacts.**
2. **Canonical environment/training paths still run `FaultInjector`, not the true 6-scenario `ScenarioEngine`.**
3. **SFT/GRPO data labels are not fully aligned with the actual executed dynamics.**
4. **Reward audit protections are not integrated into the GRPO training path.**

---

## Audit Scope

This audit focused on pre-Phase-4 readiness for the paths called out in `latestPlan.md`:

- `aic/server/env_api.py`
- `openenv.yaml`
- `scripts/run_final_benchmark.py`
- `aic/training/train_grpo.py`
- `aic/env/aic_environment.py`
- `aic/env/world_state.py`
- `aic/env/scenario_registry.py`
- `aic/env/reward_engine.py`
- `aic/training/generate_sft_data.py`
- `aic/training/data_integrity.py`
- `aic/training/scenario_contract.py`
- `aic/training/reward_audit.py`
- relevant test suites under `tests/`

---

## Evidence Collected

### 1. Test execution

Command run:

```bash
python -m pytest -q tests/test_action_parser.py tests/test_adversarial_agent.py tests/test_world_state.py tests/test_topology_scenarios.py tests/test_data_integrity.py tests/test_reward_engine.py tests/test_reward_gaming.py tests/test_reward_hacking.py tests/test_benchmark_suite.py
```

Result:

- **190 passed in 1.21s**

### 2. Dataset artifact inspection

Observed from `artifacts/sft/`:

- `orchestrator_sft.jsonl` exists
- `train.jsonl` exists
- `val.jsonl` exists
- `dataset_fingerprint.json` exists
- `generation_fingerprint.json` exists
- `agent_action_distribution.json` exists

Observed counts from current on-disk artifacts:

- raw records: **3243**
- train records: **2755**
- val records: **482**
- missing prompts: **0**
- invalid completions: **0**
- episode overlap: **0**
- prompt overlap between train/val: **1**
- completion overlap between train/val: **139**

Observed gate report from `artifacts/sft/dataset_fingerprint.json`:

- dataset marked `passed: true`
- duplicate prompt rate: **0.260117**
- action entropy: **4.7629**
- scenario imbalance ratio: **1.0206**

### 3. Benchmark/evidence inspection

Observed from existing artifacts:

- `checkpoints/grpo` does **not** exist
- existing smoke benchmark configs show `trained_policy_source: "fallback"`
- smoke configs also show missing checkpoint errors
- `results/statistical_test.json` currently shows:
  - `p_value: 0.575840701056916`
  - `significant: false`
  - `effect_size_label: "small"`
- `results/benchmark_summary.csv` currently shows:
  - `baseline_adaptive` success rate: **0.0**
  - `baseline_frozen` success rate: **0.0**
  - `trained_grpo` success rate: **0.0**

---

## Detailed Findings

## Phase 0 — Hard Freeze and Integrity Pass

### Finding P0-1 — Existing trained-policy evidence is not backed by a real checkpoint

**Severity:** Critical  
**Impact on Phase 4:** Judge-facing training evidence can be mislabeled or overstated.

#### Evidence
- `scripts/run_final_benchmark.py` supports fallback behavior when checkpoint loading fails.
- `checkpoints/grpo` is currently absent.
- Existing smoke benchmark run configs in `results/phase0_smoke/`, `results/phase1_smoke/`, and `results/phase1_competitive_smoke/` explicitly record:
  - missing checkpoint path
  - `trained_policy_source: "fallback"`

#### Risk
If this is not fail-closed, benchmark output can look like trained-policy evidence while actually measuring heuristic fallback behavior.

---

### Finding P0-2 — There are multiple benchmark stacks with different semantics

**Severity:** High  
**Impact on Phase 4:** Weakens benchmark integrity and reproducibility.

#### Evidence
- `scripts/run_final_benchmark.py` runs through `AICEnvironment`
- `aic/evals/benchmark_suite.py` runs a separate `ScenarioEngine + WorldState` stack directly

#### Risk
Metrics and conclusions can drift depending on which benchmark path generated the artifact.

---

### Finding P0-3 — Evaluation reproducibility is incomplete

**Severity:** Medium  
**Impact on Phase 4:** Re-runs may not exactly reproduce results.

#### Evidence
- `scripts/run_final_benchmark.py` inference path uses `temperature=0.3` and `do_sample=True`
- `requirements.txt` still contains non-frozen `>=` dependencies

#### Risk
Even with fixed seeds, eval output is not maximally deterministic and environment reproducibility is not fully locked.

---

## Phase 1 — Competitive Multi-Agent Environment Upgrade

### Finding P1-1 — Canonical environment still uses `FaultInjector`, not `ScenarioEngine`

**Severity:** Critical  
**Impact on Phase 4:** Phase-4 training may not match the planned brutal scenario dynamics.

#### Evidence
- `AICEnvironment` instantiates `FaultInjector`
- repo search shows `ScenarioEngine(...)` usage only in `aic/evals/benchmark_suite.py`

#### Risk
The environment used for OpenEnv/training/benchmark is not yet unified around the real scenario registry.

---

### Finding P1-2 — Competitive bidding/quota logic is only partially implemented

**Severity:** High

#### Evidence
- `bid` and `action_cost` exist in schemas and prompts
- episode budget exists in `AICEnvironment`
- there is no actual auction or explicit bid-based winner-selection layer in the environment loop

#### Risk
The environment exposes competitive-looking fields without actually enforcing the competitive mechanism described in `latestPlan.md`.

---

### Finding P1-3 — Per-agent utility logging required by the plan is missing

**Severity:** High

#### Evidence
Planned logging requirement: `(obs_i, action_i, utility_i)` per step.  
Current environment logs step summaries and candidate recommendations, but not explicit per-agent private observation/action/utility tuples.

---

### Finding P1-4 — Counterfactual credit tags are not integrated into canonical env outputs

**Severity:** Medium

#### Evidence
Counterfactual-related code exists in the repo, but the canonical `AICEnvironment.step()` path does not emit counterfactual credit tags in the per-step info/logging schema.

---

### Finding P1-5 — Coordination bottleneck mechanics are incomplete in execution path

**Severity:** Medium

#### Evidence
- `ResourceLockManager` exists
- reward has lock-related handling
- canonical env loop does not actively call `request_lock()` / `release_lock()` during action execution

#### Risk
Contention/deadlock logic exists as infrastructure but is not truly active in the canonical environment path.

---

### Finding P1-6 — Planned coordination diagnostics are only partially present

**Severity:** Medium

#### Evidence
Current env includes simple diagnostics such as:
- candidate count
- conflict rate
- adversary presence/selection/override

Missing relative to plan:
- coalition quality
- adversary manipulation success rate
- regret vs oracle-safe action
- influence matrix
- trust calibration curves
- outcome attribution
- mode collapse detector

---

## Phase 2 — Data Generation Quality Upgrade

### Finding P2-1 — SFT/GRPO data is tagged with canonical scenarios but generated from remapped `FaultInjector` dynamics

**Severity:** Critical  
**Impact on Phase 4:** Training data can be semantically misaligned with the scenario story being claimed.

#### Evidence
- `generate_sft_data.py` iterates canonical scenario IDs and names
- actual env execution uses `fault_mode=meta.fault_injector_mode`
- `train_grpo.py` prompt generation also uses `fault_mode=meta.fault_injector_mode`
- canonical env still runs `FaultInjector`, not `ScenarioEngine`

#### Risk
Samples are labeled as if they come from six distinct scenario dynamics, but actual rollouts still collapse through the four-mode fault injector mapping.

---

### Finding P2-2 — `fault_mode` metadata field is semantically incorrect

**Severity:** High

#### Evidence
`tag_sample_metadata()` writes:

- `fault_mode = meta.root_cause_node`

But the executed environment uses:

- `fault_mode = meta.fault_injector_mode`

#### Risk
The dataset metadata field named `fault_mode` does not describe the actual executed env mode.

---

### Finding P2-3 — Some schema-drift supervision labels are hardcoded instead of env-derived

**Severity:** High

#### Evidence
`generate_sft_data.py` hardcodes drift-field labels for several drift types rather than deriving them directly from the env’s active corruption behavior.

#### Risk
The model may be trained on drift annotations that do not perfectly correspond to the actual injector behavior.

---

### Finding P2-4 — Leakage validation is weaker than the plan requires

**Severity:** High

#### Evidence
- current on-disk split shows **1 prompt overlap** and **139 completion overlaps**
- `verify_no_leakage()` reports `clean` purely from zero episode overlap

#### Risk
Prompt overlap is not treated as a hard failure even though the plan calls for stronger leakage discipline.

---

### Finding P2-5 — `max_episode_overlap` gate exists but is not enforced in gate evaluation

**Severity:** Medium

#### Evidence
`DataQualityGates.max_episode_overlap` is defined, but corresponding enforcement is not part of `analyze_dataset()` gate checks.

---

### Finding P2-6 — Claimed per-agent action distribution check is weaker than advertised

**Severity:** Medium

#### Evidence
`check_agent_action_distribution()` groups examples by scenario and override/drift flags rather than actual selected-agent distribution.

#### Risk
This does not provide the per-agent action distribution guarantee implied by the plan text.

---

## Phase 3 — Reward System Hardening

### Finding P3-1 — Reward audit is not integrated into the GRPO training path

**Severity:** Critical  
**Impact on Phase 4:** Real Phase-4 training path is not protected by the audit/clamping system.

#### Evidence
- `RewardAuditLoop` is integrated in `aic/training/train.py`
- `RewardAuditLoop` is not used in `aic/training/train_grpo.py`

#### Risk
Reward hardening exists for the heuristic training path, but not for the Phase-4-critical GRPO path.

---

### Finding P3-2 — Some reward components depend on mechanics not active in canonical env execution

**Severity:** High

#### Evidence
Lock/contention penalties exist in the reward engine, but lock acquisition/release is not actually used in the action-execution flow.

#### Risk
The rubric partially rewards or penalizes behavior around mechanics that are not truly live.

---

### Finding P3-3 — Pre-Phase-4 outcome evidence is still weak

**Severity:** Medium

#### Evidence
- current benchmark significance is **not significant**
- current benchmark summary shows **0 success rate for all listed policies**

#### Interpretation
This is not proof that the reward system is broken, but it is proof that the current evidence is not yet strong enough to justify a high-confidence Phase 4 run.

---

## Why the Tests Still Passed

The pre-Phase-4 test suite passing is real and valuable, but it mostly demonstrates:
- unit-level correctness of schemas/utilities
- local integrity checks for data/reward helpers
- limited benchmark invariants

It does **not** fully prove:
- canonical env/benchmark/data path unification
- true checkpoint-backed trained-policy evidence
- scenario-registry alignment across env + SFT + GRPO + benchmark
- reward-audit integration inside the actual GRPO loop

That is why the system can look “green” at the test level while still having Phase-4 blockers.

---

## Final Risk Classification

### Must-fix before Phase 4
- P0-1: trained checkpoint integrity / fallback mislabel risk
- P1-1: canonical env still using `FaultInjector`
- P2-1: dataset generated from remapped dynamics instead of canonical scenario engine
- P3-1: reward audit absent from GRPO path

### Should-fix before Phase 4 if time allows
- P0-2: benchmark path unification
- P2-2: metadata field correctness
- P2-3: env-derived schema-drift labeling
- P2-4: stronger leakage enforcement
- P3-2: reward terms aligned with active mechanics

### Nice-to-have after blockers are resolved
- full coordination diagnostics suite
- additional statistical stability passes
- stricter deterministic packaging/reproducibility enforcement

---

## Linked Remediation Plan

See: `PHASE0_TO_PHASE3_REMEDIATION_TASKS.md`
