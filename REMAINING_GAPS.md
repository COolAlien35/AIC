# Remaining Gaps (Only Next Work)

This file tracks only unresolved items.

## 1) Prove real trained uplift (SFT/GRPO vs baseline)

**Gap**
- Current evidence pass is reproducible, but does not yet prove consistent trained uplift over baseline across held-out seeds.

**Next actions**
- Run a longer SFT + GRPO training pass (GPU recommended).
- Evaluate baseline vs SFT vs GRPO on held-out seeds.
- Publish aggregate uplift metrics (reward, final health, verifier pass rate, MTTR).

## 2) Make end-to-end training evidence judge-proof

**Gap**
- Need one fully documented training-to-eval chain with artifacts that clearly show what was trained, how long, and with what outcome.

**Next actions**
- Save run config, training logs, checkpoint metadata, and eval summaries in a single evidence bundle.
- Add one concise results table for judges (baseline/SFT/GRPO).
- Ensure all claims in docs reference generated artifacts.

## 3) Harden training dependency reliability

**Gap**
- `unsloth` is optional/missing in the Mac proof environment; runtime behavior must be explicit and deterministic across environments.

**Next actions**
- Decide official path: `unsloth` required vs optional fallback.
- Update dependency docs and preflight checks accordingly.
- Keep clear diagnostics in training scripts for dependency mode (Unsloth vs Transformers fallback).

## 4) Integrate reward-audit outputs into GRPO results

**Gap**
- Reward-audit logic exists, but final GRPO reports need explicit audit statistics and filtering behavior.

**Next actions**
- Log flagged episodes during GRPO runs.
- Report audit counts/rates in final benchmark output.
- Document how flagged episodes affect optimization (filter/clamp policy).

## 5) Integrate curriculum progression into real training runs

**Gap**
- Curriculum scaffolding exists; needs evidence that training actually progresses through tiers and benefits from it.

**Next actions**
- Record tier transitions during training.
- Report reward/health trend by tier.
- Include curriculum evidence in final training report.

## 6) Validate model export on a genuinely trained checkpoint

**Gap**
- Export path exists, but needs proof on a non-smoke, actually trained checkpoint.

**Next actions**
- Export trained SFT/GRPO checkpoint.
- Reload and run inference/eval smoke checks.
- Save export validation results in `results/`.

## Recommended execution order

1. Run GPU-backed SFT + GRPO + held-out eval.
2. Add audit + curriculum reporting to that run.
3. Validate export on resulting checkpoint.
4. Update final evidence docs with measured uplift only.