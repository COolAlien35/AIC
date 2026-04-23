# Remaining Gaps vs Hackathon Guide

## Current Status

The repo is now **architecturally aligned** with the hackathon guide in the following areas:

- Environment-first design via `aic/env/aic_environment.py`
- Structured action and observation schemas via `aic/schemas/`
- Multiple reward functions and anti-hacking checks via `aic/env/reward_engine.py` and `aic/training/reward_audit.py`
- Curriculum scaffolding via `aic/training/curriculum.py`
- SFT and GRPO codepaths via `aic/training/generate_sft_data.py`, `run_sft.py`, `train_grpo.py`, `modeling_unsloth.py`
- Local deployment via `aic/server/env_api.py` and `scripts/run_env_server.py`
- Evaluation/demo scaffolding via `aic/evals/rl_eval.py`, `results/`, and smoke scripts

However, the project is **not yet fully hackathon-complete operationally**. The main gaps are execution proof, evidence quality, and final integration.

---

## Remaining Gaps

### 1) Real improvement evidence is still weak

**Problem**
- `results/before_after_demo.md` currently shows **0.00 improvement** between untrained and trained runs.
- `run_hackathon.py` generates a **projected** after-training reward curve instead of evidence from an actual trained RL policy.

**Why this matters**
- Judges will ask for measurable before/after behavior from actual training, not projected uplift.

**Files to update**
- `results/before_after_demo.md`
- `run_hackathon.py`
- `scripts/generate_plots.py`
- `README.md`

**What to do**
- Replace projected curves with curves generated from real trained policy outputs.
- Recompute before/after comparison using:
  - baseline heuristic policy
  - SFT model
  - GRPO model
- Export benchmark tables and screenshots from actual results.

---

### 2) End-to-end SFT/GRPO evidence is not yet judge-proof

**Problem**
- SFT/GRPO artifacts exist in `artifacts/` and `checkpoints/`, but it is not yet clearly documented whether:
  - SFT completed successfully beyond smoke testing,
  - GRPO completed successfully with real rewards,
  - trained checkpoints outperform baselines on held-out evaluation.

**Files to update**
- `run_training_smoke.py`
- `aic/training/run_sft.py`
- `aic/training/train_grpo.py`
- `aic/evals/rl_eval.py`
- `results/`

**What to do**
- Run one clean end-to-end pass:
  1. generate SFT data
  2. run SFT
  3. run GRPO
  4. evaluate both on held-out seeds
- Save:
  - training logs
  - evaluation metrics
  - checkpoint metadata
  - benchmark summary markdown/CSV

---

### 3) Training stack is still runtime-fragile

**Problem**
- The code expects `trl`, `datasets`, `peft`, and `unsloth`, but actual runtime availability may still vary by environment.

**Files to update**
- `requirements.txt`
- `run_training_smoke.py`
- `run_hackathon.sh`
- `README.md`

**What to do**
- Pin and verify all training dependencies.
- Add one explicit environment verification step before training.
- Print clear dependency diagnostics in smoke scripts.
- Confirm whether Unsloth is truly used or only falling back to Transformers.

---

### 4) Reward audit exists, but is not fully integrated into the real RL loop

**Problem**
- `aic/training/reward_audit.py` exists, but it is not yet clearly wired into GRPO rollouts and reporting.

**Files to update**
- `aic/training/train_grpo.py`
- `aic/training/reward_audit.py`
- `results/`
- `README.md`

**What to do**
- Call the audit loop during reward computation / rollout collection.
- Log flagged episodes to `logs/audit/` during real training.
- Clamp/filter suspicious episodes before optimizer updates.
- Report audit statistics in final results.

---

### 5) Curriculum exists, but needs to be actually used in training

**Problem**
- `aic/training/curriculum.py` defines difficulty tiers, but curriculum progression is not yet clearly integrated into `train_grpo.py` or training orchestration.

**Files to update**
- `aic/training/curriculum.py`
- `aic/training/train_grpo.py`
- `run_hackathon.py`
- `results/`

**What to do**
- Use curriculum scheduler in real training.
- Log tier transitions.
- Prove that early easy tasks produce non-zero reward.
- Show progression from easy → medium → hard in outputs.

---

### 6) Remote OpenEnv / HF deployment still needs proof

**Problem**
- Local FastAPI environment works, but remote deployment evidence is still partial.

**Files to update**
- `deploy/deploy_instructions.md`
- `deploy/Dockerfile`
- `hf_space_readme.md`
- `README.md`

**What to do**
- Do one verified remote deployment to HF Space / OpenEnv-compatible host.
- Record the live URL.
- Confirm reset/step works remotely.
- Add the deployment proof to docs.

---

### 7) Export validation needs to be run on a real trained checkpoint

**Problem**
- Export helpers exist, but they still need validation against a real trained model.

**Files to update**
- `aic/training/export_model.py`
- `eval/test_export.py`
- `results/`

**What to do**
- Export SFT and/or GRPO checkpoint.
- Reload exported model.
- Run a prompt-level smoke test.
- Verify inference quality survives export.

---

### 8) Demo/report messaging should distinguish real vs projected results

**Problem**
- Some outputs are real, some are placeholders or projected, but the messaging is not yet fully clean for reviewers.

**Files to update**
- `README.md`
- `report.md`
- `results/before_after_demo.md`
- `run_hackathon.py`

**What to do**
- Remove or relabel projected metrics.
- Only claim verified improvements.
- Clearly separate:
  - implemented pipeline
  - smoke-tested pipeline
  - verified trained uplift

---

## Recommended Next Sequence

### Phase 1 — Produce real training proof
1. Run SFT end-to-end on current dataset.
2. Run a small GRPO job with real reward.
3. Save checkpoints and logs.

### Phase 2 — Evaluate properly
4. Benchmark baseline vs SFT vs GRPO on held-out seeds.
5. Generate real reward curves and verifier metrics.
6. Replace placeholder before/after demo.

### Phase 3 — Strengthen training integrity
7. Integrate reward audit into real GRPO loop.
8. Integrate curriculum into real training execution.
9. Log audit flags and curriculum progression.

### Phase 4 — Finalize deployment and export
10. Validate export on trained model.
11. Deploy environment remotely.
12. Verify remote reset/step.

### Phase 5 — Clean submission narrative
13. Update README/report/demo to reflect only verified evidence.
14. Prepare final benchmark + safeguards summary for judges.

---

## Bottom Line

The project is now **much closer to hackathon-ready**. The remaining work is mostly not architectural anymore — it is about:

- proving real trained improvement,
- integrating audit/curriculum into actual training runs,
- validating export,
- and producing clean deployment/demo evidence.

Once those are done, the project should satisfy the hackathon guide much more convincingly.