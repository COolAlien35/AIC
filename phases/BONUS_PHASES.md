# 🔬 BONUS PHASES — If You Have Extra Time (72-hour track)
> Execute in order. Each adds measurable value to your score.

---

## BONUS A — Ablation Study
**Time: ~2 hours**

Shows judges *why* each component matters — not just that it works.

```bash
python scripts/run_final_benchmark.py --policy baseline_frozen --tag ablation_baseline
python scripts/run_final_benchmark.py --policy sft_only --tag ablation_sft_only
python scripts/run_final_benchmark.py --policy grpo_no_curriculum --tag ablation_no_curriculum
python scripts/run_final_benchmark.py --policy trained_grpo --tag ablation_full
```

Expected result table (fill in with your real numbers):

| Configuration | Avg Reward | Success % | Improvement |
|---------------|------------|-----------|-------------|
| Frozen baseline | -287 | 0.0% | — |
| SFT only | -250 | 2–5% | +13% |
| GRPO (no curriculum) | -230 | 5–10% | +20% |
| GRPO + curriculum ★ | -180 | 15–25% | +37% |

**Pitch line:** *"Each ablation confirms the contribution of each training stage. SFT alone gives +13%. GRPO alone gives +20%. Together with curriculum, +37%."*

---

## BONUS B — Adversarial Override Analysis
**Time: ~1 hour**

Quantifies exactly how often your trained model rejects the adversarial agent:

```python
import pandas as pd, json
from pathlib import Path

override_results = []
for result_file in Path("results/episode_details/").glob("*.json"):
    data = json.loads(result_file.read_text())
    if data.get("adversarial_present"):
        override_results.append({
            "policy": data["policy"],
            "correctly_overrode": data.get("override_adversary", False),
            "scenario": data["scenario"],
        })

df = pd.DataFrame(override_results)
print(df.groupby("policy")["correctly_overrode"].mean())
# Expected: trained_grpo → 80%+, baseline_frozen → ~50% (random)
```

**What this proves:** Your model isn't just getting lucky on rewards — it's specifically learning to detect adversarial inputs.

---

## BONUS C — Per-Scenario Breakdown Heatmap
**Time: ~1 hour**

Shows judges which fault scenarios your model handles best and worst:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

scenario_df = pd.read_csv("results/benchmark_by_scenario.csv")
pivot = scenario_df.pivot(index="scenario", columns="policy", values="success_rate")

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn",
            center=0.1, ax=ax, linewidths=0.5)
ax.set_title("Success Rate by Scenario and Policy\n(Green = better)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/scenario_heatmap.png", dpi=150)
print("✅ Scenario heatmap saved")
```

---

## BONUS D — HuggingFace Model Upload
**Time: ~30 minutes**

Gets you a public model card and a citable URL:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="exports/aic-orchestrator-trained",
    repo_id="YOUR_USERNAME/aic-orchestrator-grpo",
    repo_type="model",
)
print("✅ Model uploaded to HuggingFace Hub")
```

Then add to README:
```markdown
🤗 **Model**: [YOUR_USERNAME/aic-orchestrator-grpo](https://huggingface.co/YOUR_USERNAME/aic-orchestrator-grpo)
```

---

## Win Probability Summary

| Stop Point | Phases Complete | Win Probability |
|------------|----------------|-----------------|
| A | 0–4 | 60% |
| B | + 5–6 | 75% |
| C | + 7 | 80% |
| D | + All Bonus | 85–90% |
