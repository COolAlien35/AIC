# Phase 0–3 Remediation Tasks Before Phase 4

Status: **Planned only**  
Fixes applied in this document: **None**

This file converts the audit findings into an implementation checklist.  
It is intentionally ordered to protect **Phase 4 evidence quality first**.

---

## Implementation Objective

Before starting the Phase 4 parallel training campaign, ensure that:

1. benchmark evidence is canonical and fail-closed,
2. environment dynamics match the six-scenario story,
3. SFT/GRPO data is generated from the real canonical env path,
4. reward exploit protections are active in the actual GRPO loop.

---

## Workstream A — Phase 0 Integrity Lock

### A1. Make trained-policy evaluation fail-closed
- [ ] Update benchmark policy loading so `trained_grpo` cannot silently masquerade as a trained checkpoint when model loading fails.
- [ ] Separate policy labels clearly if fallback behavior must remain available for debugging.
- [ ] Add checkpoint preflight validation for model/tokenizer/metadata presence.

**Primary files**
- `scripts/run_final_benchmark.py`
- `scripts/generate_evidence_manifest.py`

**Acceptance criteria**
- Benchmark output never reports `trained_grpo` unless a real checkpoint was loaded.
- Evidence manifest explicitly records checkpoint integrity status.

---

### A2. Collapse judge-facing benchmarking to one canonical path
- [ ] Designate `scripts/run_final_benchmark.py` as the only judge-facing benchmark path.
- [ ] Demote or relabel `aic/evals/benchmark_suite.py` as non-canonical / dev-only.
- [ ] Remove ambiguity in docs and evidence generation.

**Primary files**
- `scripts/run_final_benchmark.py`
- `aic/evals/benchmark_suite.py`
- `README.md`
- `FINAL_SUBMISSION.md`

**Acceptance criteria**
- Only one benchmark path is used for official results.
- Docs and evidence references are consistent.

---

### A3. Tighten reproducibility
- [ ] Freeze remaining `>=` dependency ranges to exact versions.
- [ ] Make evaluation inference deterministic or explicitly separate deterministic eval from sampled demo generation.
- [ ] Persist seed/config manifests for every benchmark run.

**Primary files**
- `requirements.txt`
- `scripts/run_final_benchmark.py`

**Acceptance criteria**
- Same checkpoint + same seed + same config produce stable eval output.

---

## Workstream B — Canonical Competitive Environment

### B1. Unify env on the six-scenario registry
- [ ] Add canonical `scenario_id` support to `AICEnvironment`.
- [ ] Replace or wrap `FaultInjector` usage so the canonical env path runs `ScenarioEngine` dynamics.
- [ ] Ensure OpenEnv server, SFT generation, GRPO prompt generation, and final benchmark all use the same scenario source.

**Primary files**
- `aic/env/aic_environment.py`
- `aic/env/scenario_registry.py`
- `aic/server/env_api.py`
- `openenv.yaml`

**Acceptance criteria**
- Canonical env is scenario-registry-backed end to end.

---

### B2. Implement the competitive resource-allocation layer for real
- [ ] Turn `bid` and `action_cost` into actual execution-time selection constraints.
- [ ] Introduce real quota/auction/scarcity logic instead of prompt-only fields.
- [ ] Decide whether verifier action bypasses the auction or participates with explicit semantics.

**Primary files**
- `aic/env/aic_environment.py`
- `aic/schemas/actions.py`
- `aic/schemas/traces.py`
- `aic/agents/*.py`

**Acceptance criteria**
- Competition changes which actions execute, not just how they are described.

---

### B3. Activate per-agent utilities, contention, and attribution
- [ ] Define utility computation per agent.
- [ ] Log `(obs_i, action_i, utility_i)` per step.
- [ ] Activate lock/contention handling through real `request_lock()` / `release_lock()` usage or replace with a simpler active contention model.
- [ ] Integrate counterfactual credit tags into canonical step outputs.

**Primary files**
- `aic/env/aic_environment.py`
- `aic/env/lock_manager.py`
- `aic/env/counterfactual_simulator.py`
- `aic/utils/logging_utils.py`

**Acceptance criteria**
- Coordination and scarcity are reflected in both execution and logs.

---

### B4. Complete the missing Phase 1 diagnostics
- [ ] Add coalition quality.
- [ ] Add adversary manipulation success rate.
- [ ] Add regret vs oracle-safe action.
- [ ] Add influence / attribution outputs.
- [ ] Add trust calibration and mode-collapse diagnostics.

**Primary files**
- `aic/env/aic_environment.py`
- `scripts/run_final_benchmark.py`
- plotting/evidence scripts under `scripts/`

**Acceptance criteria**
- Required Phase-1 judge-facing metrics are emitted in the canonical pipeline.

---

## Workstream C — Phase 2 Data Integrity and Realism Repair

### C1. Regenerate SFT/GRPO data from the canonical env path
- [ ] Stop using scenario-name labels with remapped four-mode dynamics.
- [ ] Generate SFT and GRPO prompt data directly from the unified scenario-backed env.
- [ ] Ensure each sample’s scenario metadata is derived from the actually executed scenario.

**Primary files**
- `aic/training/generate_sft_data.py`
- `aic/training/train_grpo.py`
- `aic/training/scenario_contract.py`

**Acceptance criteria**
- Data labels and executed dynamics are identical in meaning.

---

### C2. Fix metadata semantics
- [ ] Redefine `fault_mode` metadata to reflect the real executed env mode or rename it to eliminate ambiguity.
- [ ] Keep `scenario_id`, `scenario_name`, `root_cause_node`, and executed fault semantics distinct and explicit.

**Primary files**
- `aic/training/scenario_contract.py`
- `aic/training/generate_sft_data.py`

**Acceptance criteria**
- No metadata field is overloaded or misleading.

---

### C3. Derive schema-drift labels from the environment, not hardcoded heuristics
- [ ] Source drift detection labels from active corruption state in the env/injector.
- [ ] Validate drift field labels against the real corruption behavior.

**Primary files**
- `aic/training/generate_sft_data.py`
- `aic/env/schema_drift.py`
- `aic/env/aic_environment.py`

**Acceptance criteria**
- Supervised drift labels match actual env behavior.

---

### C4. Strengthen leakage and gate enforcement
- [ ] Treat prompt overlap as an explicit validation outcome, not an ignored side metric.
- [ ] Decide policy for completion overlap and document it.
- [ ] Enforce `max_episode_overlap` and any new overlap gates in `analyze_dataset()`.
- [ ] Make the integrity pipeline fail hard when unacceptable leakage is present.

**Primary files**
- `aic/training/data_integrity.py`
- `tests/test_data_integrity.py`

**Acceptance criteria**
- Integrity “clean” status reflects the actual leakage policy.

---

### C5. Replace pseudo agent-distribution reporting with real action-selection statistics
- [ ] Measure actual selected-agent frequencies.
- [ ] Report per-scenario and global distributions.
- [ ] Add imbalance alerts if one agent dominates unexpectedly.

**Primary files**
- `aic/training/data_integrity.py`
- `aic/training/generate_sft_data.py`

**Acceptance criteria**
- “Per-agent action distribution” means true selected-agent behavior.

---

## Workstream D — Phase 3 Reward Hardening on the Real RL Path

### D1. Integrate reward audit into GRPO
- [ ] Wire `RewardAuditLoop` into `aic/training/train_grpo.py`.
- [ ] Ensure audit clamping/termination can affect rewards returned to GRPO.
- [ ] Persist GRPO audit summaries alongside training logs.

**Primary files**
- `aic/training/train_grpo.py`
- `aic/training/reward_audit.py`

**Acceptance criteria**
- Reward exploit protection is active in the actual Phase 4 training path.

---

### D2. Align reward terms with active environment mechanics
- [ ] Remove or defer reward terms that depend on inactive mechanics.
- [ ] Or activate the missing mechanics first and keep the terms.
- [ ] Verify no reward component is effectively dead or misleading.

**Primary files**
- `aic/env/reward_engine.py`
- `aic/env/aic_environment.py`

**Acceptance criteria**
- Every reward term maps to an actually active behavior/mechanic.

---

### D3. Extend exploit tests to the GRPO path
- [ ] Add path-specific tests for no-op farming.
- [ ] Add path-specific tests for verifier stalling.
- [ ] Add path-specific tests for unsafe-action bypass.
- [ ] Add path-specific tests for confidence spoofing and prompt injection.

**Primary files**
- `tests/test_reward_gaming.py`
- `tests/test_reward_hacking.py`
- `tests/test_training_loop.py`
- new GRPO-specific tests as needed

**Acceptance criteria**
- Reward-gaming protections are verified on the same path used for Phase 4.

---

## Recommended Execution Order

1. **A1–A3** — benchmark integrity lock
2. **B1** — scenario-registry unification in canonical env
3. **C1–C5** — regenerate and re-validate training data
4. **B2–B4** — finish competitive env mechanics and diagnostics
5. **D1–D3** — reward hardening on real GRPO path
6. **Only then begin Phase 4 training campaign**

---

## Definition of Done Before Phase 4

- [ ] `trained_grpo` evaluation is backed by a real checkpoint and cannot silently fall back
- [ ] canonical env/benchmark/data generation all use the same six-scenario dynamics
- [ ] regenerated SFT/GRPO data passes strengthened integrity and leakage gates
- [ ] reward audit is active in the GRPO training path
- [ ] benchmark evidence is canonical, reproducible, and clearly labeled

---

## Suggested Validation Pass After Fixes (Future Work)

After implementation, rerun the following before any serious Phase 4 campaign:

- full pre-Phase-4 pytest suite
- SFT regeneration + integrity pipeline
- GRPO prompt dataset regeneration sanity checks
- canonical benchmark with real checkpoint loading
- evidence manifest regeneration

This document intentionally stops at planning.  
No code fixes are applied here.
