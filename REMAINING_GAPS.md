# Remaining Gaps (Deferred, GPU Required)

This file lists only unresolved work after the Mac CPU submission-hardening pass.

## 1) Prove trained uplift vs baseline on held-out seeds

**Gap**
- We have reproducible CPU-safe evidence, but no GPU-scale proof yet that trained policy consistently outperforms baseline.

**Run**
```bash
./.venv/bin/python run_hackathon.py grpo
./.venv/bin/python scripts/run_final_benchmark.py
```

**Expected outputs**
- `checkpoints/grpo/` (trained checkpoint artifacts)
- `logs/eval/policy_benchmark.jsonl` with trained-policy rows
- `results/benchmark_summary.csv` showing baseline vs trained aggregates
- Updated `results/reward_curve.png` and `results/verifier_pass_rate.png`

## 2) Publish judge-proof training evidence bundle

**Gap**
- Need one end-to-end, single-location proof bundle (config + logs + checkpoints + eval summary) from an actual GPU training run.

**Run**
```bash
./.venv/bin/python run_hackathon.py verify grpo plots demo
```

**Expected outputs**
- `results/evidence_manifest.json`
- `results/evidence_manifest.md`
- GRPO checkpoint metadata in `checkpoints/grpo/`
- Benchmark/demo artifacts regenerated from that run

## 3) Reward-audit reporting in final GRPO evidence

**Gap**
- Audit files are produced, but final judge-facing summary must explicitly report audit counts/rates and optimization handling.

**Run / verify**
```bash
./.venv/bin/python run_hackathon.py grpo
```

**Expected outputs**
- `logs/audit/` summaries for GRPO episodes
- Documented aggregate audit stats in final results docs

## 4) Curriculum progression evidence in trained run

**Gap**
- Curriculum scheduler exists, but we still need proof of tier progression and outcome trend in a real training run.

**Run / verify**
```bash
./.venv/bin/python run_hackathon.py grpo
```

**Expected outputs**
- Training logs/artifacts showing tier transitions
- Reported reward/health trend by tier in final docs

## 5) Export validation on a genuinely trained checkpoint

**Gap**
- Export flow exists, but validation on a fully trained GRPO checkpoint is still pending.

**Run**
```bash
./.venv/bin/python eval/test_export.py --source checkpoints/grpo
```

**Expected outputs**
- Exported model artifacts (target export directory)
- Reload/inference smoke-test pass logs
- Export validation result referenced in submission docs

## Execution order

1. GRPO training on GPU.
2. Held-out benchmark and plot/demo regeneration.
3. Audit + curriculum evidence extraction.
4. Export validation on trained checkpoint.
5. Final docs update with measured uplift only.