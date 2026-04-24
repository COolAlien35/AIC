# 🟢 PHASE 4 — BENCHMARK + STATISTICAL PROOF
> **Time: 1 hour | Deliverables: `results/benchmark_summary.csv` + `results/statistical_test.json`**  
> This is the number judges will quote in their decision. Make it real.

---

## CELL 8 — Run Full Benchmark

```python
import subprocess, sys

result = subprocess.run([
    sys.executable, "scripts/run_final_benchmark.py",
    "--episodes", "5",    # 5 per scenario × 6 scenarios = 30 episodes per policy
    "--scenarios", "all",
    "--policies", "baseline_frozen,baseline_adaptive,trained_grpo",
], capture_output=False)

assert result.returncode == 0, "Benchmark FAILED — check traceback above"
```

---

## CELL 9 — Validate Results + Print Summary

```python
import pandas as pd, json

df = pd.read_csv("results/benchmark_summary.csv")
stats = json.loads(open("results/statistical_test.json").read())

print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(df.to_string(index=False))

print("\nSTATISTICAL TEST")
print(f"  Improvement: {stats['improvement']:+.2f} ({stats['improvement_pct']:+.1f}%)")
print(f"  p-value: {stats['p_value']:.4f} ({'SIGNIFICANT ✅' if stats['significant'] else 'not significant ⚠️'})")
print(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['effect_size_label']} effect)")

# Critical sanity check
trained_row = df[df["policy"] == "trained_grpo"]
assert not trained_row.empty, "❌ TRAINED POLICY MISSING FROM RESULTS"

trained_avg = float(trained_row["avg_reward"].iloc[0])
baseline_avg = float(df[df["policy"] == "baseline_frozen"]["avg_reward"].iloc[0])

if trained_avg > baseline_avg:
    print(f"\n✅ Trained model ({trained_avg:.2f}) beats baseline ({baseline_avg:.2f})")
else:
    print(f"\n⚠️ WARNING: Trained model ({trained_avg:.2f}) is WORSE than baseline ({baseline_avg:.2f})")
    print("   → See recovery steps below")
```

---

## If the Trained Model Is Worse Than Baseline

This is rare but possible. Diagnose in order:

**Step 1:** Check `logs/grpo_progress.jsonl` — did reward improve during training at all?

```python
import json
from pathlib import Path

entries = [json.loads(l) for l in open("logs/grpo_progress.jsonl") if l.strip()]
print(f"First reward: {entries[0]['reward']:.4f}")
print(f"Last reward:  {entries[-1]['reward']:.4f}")
print(f"Delta:        {entries[-1]['reward'] - entries[0]['reward']:+.4f}")
```

**Step 2 — If reward improved in training but not in benchmark:**  
Eval vs train distribution mismatch. Run benchmark on training scenarios only first:

```python
subprocess.run([sys.executable, "scripts/run_final_benchmark.py",
    "--episodes", "5", "--scenarios", "cascading_failure,memory_leak",
    "--policies", "trained_grpo"])
```

**Step 3 — If reward never improved:**  
Check reward function is returning non-zero values:

```python
# Add to RewardEngine temporarily:
def compute(self, ...):
    reward = self._original_compute(...)
    print(f"DEBUG reward: {reward}")  # Must not always be the same number
    return reward
```

**Step 4 — If reward is always the exact same number:**  
The environment is broken, not the model. Check `AICEnvironment.step()` returns.

---

## ✅ Phase 4 Completion Criteria

- [ ] `results/benchmark_summary.csv` exists with all 3 policies
- [ ] `results/benchmark_by_scenario.csv` exists
- [ ] `results/statistical_test.json` exists with `p_value`, `cohens_d`, `improvement`
- [ ] `trained_grpo` avg_reward > `baseline_frozen` avg_reward
- [ ] `p_value < 0.05` (statistically significant improvement)

**→ Next: [PHASE_5_EVIDENCE_BUNDLE.md](PHASE_5_EVIDENCE_BUNDLE.md)**  
*(Can be done in parallel while GPU runs Phases 2–3)*
