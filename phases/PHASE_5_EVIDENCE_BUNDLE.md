# 🟣 PHASE 5 — EVIDENCE BUNDLE + README SURGERY
> **Time: 45 minutes | Run this in PARALLEL on your local machine while GPU handles Phases 2–4**  
> Deliverables: `results/reward_curve.png`, `results/policy_comparison.png`, `results/evidence_manifest.json`, updated `README.md`

---

## 5.1 — Generate Reward Curve Plot

Add this function to `scripts/generate_plots.py`:

```python
def plot_grpo_reward_curve(log_path: str = "logs/grpo_progress.jsonl",
                            out_path: str = "results/reward_curve.png"):
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    entries = [json.loads(l) for l in open(log_path) if l.strip()]
    steps = [e["step"] for e in entries]
    rewards = [e["reward"] for e in entries]

    # Smooth with rolling average
    window = max(5, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    smooth_steps = steps[window-1:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, rewards, alpha=0.3, color="#4A90D9", linewidth=0.8, label="Raw reward")
    ax.plot(smooth_steps, smoothed, color="#1A5F9E", linewidth=2.5, label=f"Smoothed (window={window})")

    # Shade training phases
    if len(steps) > 20:
        ax.axvspan(steps[0], steps[len(steps)//3], alpha=0.08, color="red", label="Exploration phase")
        ax.axvspan(steps[len(steps)//3], steps[2*len(steps)//3], alpha=0.08, color="yellow", label="Learning phase")
        ax.axvspan(steps[2*len(steps)//3], steps[-1], alpha=0.08, color="green", label="Convergence phase")

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title("AIC — GRPO Training Reward Curve\n(Adaptive Incident Choreographer)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Start/end annotations
    ax.annotate(f"Start: {rewards[0]:.3f}", xy=(steps[0], rewards[0]),
                xytext=(steps[len(steps)//5], rewards[0]+abs(rewards[0])*0.1),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.annotate(f"End: {rewards[-1]:.3f}", xy=(steps[-1], rewards[-1]),
                xytext=(steps[3*len(steps)//4], rewards[-1]-abs(rewards[-1])*0.1),
                arrowprops=dict(arrowstyle="->"), fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Reward curve saved: {out_path}")
```

---

## 5.2 — Generate Policy Comparison Chart

```python
def plot_policy_comparison(benchmark_csv: str = "results/benchmark_summary.csv",
                            out_path: str = "results/policy_comparison.png"):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(benchmark_csv)

    colors = {
        "baseline_frozen": "#E74C3C",
        "baseline_adaptive": "#F39C12",
        "trained_grpo": "#27AE60"
    }
    labels = {
        "baseline_frozen": "Frozen Baseline",
        "baseline_adaptive": "Adaptive Baseline",
        "trained_grpo": "Trained (GRPO) ★"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in [
        (ax1, "avg_reward", "Average Reward per Episode"),
        (ax2, "success_rate", "Success Rate"),
    ]:
        for i, row in df.iterrows():
            policy = row["policy"]
            value = row[metric]
            std = row.get("std_reward", 0) if metric == "avg_reward" else 0
            ax.bar(i, value, color=colors.get(policy, "#888"),
                   yerr=std, capsize=5, width=0.6)
            ax.text(i, value + (std or 0) + abs(value)*0.02,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([labels.get(p, p) for p in df["policy"]], rotation=15, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AIC Policy Comparison — Benchmark Results\n(30 episodes × 6 scenarios each)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Policy comparison chart saved: {out_path}")
```

### Run Both Plots

```python
# Run after Phase 4 completes and logs/grpo_progress.jsonl is downloaded
plot_grpo_reward_curve()
plot_policy_comparison()
```

---

## 5.3 — Evidence Manifest Generator

Create `scripts/generate_evidence_manifest.py`:

```python
#!/usr/bin/env python3
"""
Run AFTER all training and benchmarking is complete.
Generates the complete evidence index for submission.
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def check(path: Path, condition: bool = True, min_count: int = 0) -> tuple[str, dict]:
    exists = path.exists()
    if not exists:
        return "❌", {"path": str(path), "status": "MISSING"}

    info = {"path": str(path), "size_kb": round(path.stat().st_size / 1024, 1)}

    if path.suffix == ".jsonl":
        count = sum(1 for _ in open(path))
        info["record_count"] = count
        ok = count >= min_count
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        info["keys"] = list(data.keys())[:5]
        ok = condition
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        info["rows"] = len(df)
        info["columns"] = list(df.columns)
        ok = condition
    elif path.suffix in (".png", ".jpg"):
        ok = True
    else:
        ok = True

    return ("✅" if ok else "⚠️"), info


def generate_manifest():
    print("📋 Generating Evidence Manifest...\n")
    evidence = {}

    # SFT Training Data
    status, info = check(Path("artifacts/sft/orchestrator_sft.jsonl"), min_count=500)
    evidence["sft_training_data"] = {"status": status, **info}

    # SFT Checkpoint
    sft_meta = Path("checkpoints/sft/sft_metadata.json")
    if sft_meta.exists():
        meta = json.loads(sft_meta.read_text())
        evidence["sft_checkpoint"] = {
            "status": "✅" if ("Qwen" in meta.get("model_name", "") or "Llama" in meta.get("model_name", "")) else "⚠️",
            "model_name": meta.get("model_name"),
            "path": "checkpoints/sft/",
        }

    # GRPO Checkpoint
    grpo_summary = Path("checkpoints/grpo/training_summary.json")
    if grpo_summary.exists():
        summary = json.loads(grpo_summary.read_text())
        evidence["grpo_checkpoint"] = {
            "status": "✅" if summary.get("reward_delta", 0) > 0 else "⚠️",
            "reward_delta": summary.get("reward_delta"),
            "total_steps": summary.get("total_steps"),
            "training_time_minutes": summary.get("training_time_minutes"),
        }

    # Benchmark results
    status, info = check(Path("results/benchmark_summary.csv"))
    evidence["benchmark_results"] = {"status": status, **info}

    # Statistical test
    stat_path = Path("results/statistical_test.json")
    if stat_path.exists():
        stat_data = json.loads(stat_path.read_text())
        evidence["statistical_test"] = {
            "status": "✅" if stat_data.get("significant") else "⚠️",
            "p_value": stat_data.get("p_value"),
            "improvement_pct": stat_data.get("improvement_pct"),
            "cohens_d": stat_data.get("cohens_d"),
            "effect_size": stat_data.get("effect_size_label"),
        }

    # Visual assets
    for key, path in [
        ("reward_curve_plot", "results/reward_curve.png"),
        ("policy_comparison_plot", "results/policy_comparison.png"),
    ]:
        status, info = check(Path(path))
        evidence[key] = {"status": status, **info}

    # GRPO logs
    status, info = check(Path("logs/grpo_progress.jsonl"), min_count=50)
    evidence["grpo_training_logs"] = {"status": status, **info}

    # Save manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "evidence": evidence,
        "summary": {
            "total_artifacts": len(evidence),
            "passing": sum(1 for v in evidence.values() if v.get("status") == "✅"),
            "warnings": sum(1 for v in evidence.values() if v.get("status") == "⚠️"),
            "missing": sum(1 for v in evidence.values() if v.get("status") == "❌"),
        }
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/evidence_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Human-readable version
    with open("results/evidence_manifest.md", "w") as f:
        f.write("# AIC — Evidence Manifest\n\n")
        f.write(f"Generated: {manifest['generated_at']}\n\n")
        for key, val in evidence.items():
            f.write(f"## {val.get('status', '?')} {key}\n")
            for k, v in val.items():
                if k != "status":
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")

    print(f"\n✅ Manifest saved: results/evidence_manifest.json")
    print(f"   Passing:  {manifest['summary']['passing']}")
    print(f"   Warnings: {manifest['summary']['warnings']}")
    print(f"   Missing:  {manifest['summary']['missing']}")


if __name__ == "__main__":
    generate_manifest()
```

```bash
python scripts/generate_evidence_manifest.py
```

---

## 5.4 — README Surgery

Find and replace every instance of:

| Placeholder | Replace With |
|-------------|-------------|
| `tiny-gpt2` or `sshleifer/tiny-gpt2` | `Qwen/Qwen2.5-3B-Instruct` |
| `[FILL]` or `[TODO]` | Actual values from your benchmark results |
| Any placeholder reward numbers | Real numbers from `results/statistical_test.json` |

**Quick audit:**

```bash
grep -n "tiny-gpt2\|FILL\|TODO\|\[PLACEHOLDER\]" README.md
# Must return 0 matches before submission
```

**What the README training section must contain:**

```markdown
## Training Results

- **Model**: Qwen/Qwen2.5-3B-Instruct (4-bit quantized, LoRA fine-tuned)
- **SFT Data**: 600+ examples across 6 fault scenarios (cascading_failure, memory_leak, db_connection_saturation, network_storm, schema_migration_failure, credential_compromise)
- **GRPO Training**: 150 steps with 8-component reward decomposition (R1–R8)
- **Reward Improvement**: +[X] reward units ([X]% improvement over frozen baseline)
- **Statistical Test**: p=[X], Cohen's d=[X] ([large/medium] effect size)
- **Adversarial Override Rate**: [X]% on episodes with adversarial agent present

See `results/evidence_manifest.md` for full artifact index.
```

---

## ✅ Phase 5 Completion Criteria

- [ ] `results/reward_curve.png` saved at 150 DPI
- [ ] `results/policy_comparison.png` saved at 150 DPI
- [ ] `results/evidence_manifest.json` generated with 0 missing items
- [ ] `README.md` has zero instances of `tiny-gpt2`, `[FILL]`, or placeholder text

**→ Next: [PHASE_6_EXPORT_DEMO.md](PHASE_6_EXPORT_DEMO.md)**
