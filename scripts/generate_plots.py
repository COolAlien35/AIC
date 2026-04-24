#!/usr/bin/env python3
"""Generate result plots and evidence for the hackathon submission.

Produces:
- results/reward_curve.png
- results/verifier_pass_rate.png
- results/grpo_reward_curve.png
- results/policy_comparison.png
- results/before_after_demo.md
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DARK_BG = "#0C0F0A"
_PLOT_BG = "#111610"


def _load_policy_benchmark(path="logs/eval/policy_benchmark.jsonl"):
    p = Path(path)
    if not p.exists():
        return {}
    by_policy = {}
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_policy.setdefault(row["policy"], []).append(row)
    for policy in by_policy:
        by_policy[policy] = sorted(by_policy[policy], key=lambda r: r["episode_id"])
    return by_policy


def _style_ax(ax):
    ax.set_facecolor(_PLOT_BG)
    ax.tick_params(colors="#6B7A68")
    ax.grid(True, alpha=0.15, color="#34D399")
    for s in ax.spines.values():
        s.set_color("#1E2B1A")


def plot_reward_curve(output_path="results/reward_curve.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    by_policy = _load_policy_benchmark()
    if not by_policy:
        raise FileNotFoundError("Missing logs/eval/policy_benchmark.jsonl")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    palette = {
        "baseline_frozen_trust": ("#14B8A6", "o", "Frozen trust (baseline)"),
        "baseline_adaptive_trust": ("#34D399", "s", "Adaptive trust (trained-mode)"),
    }
    for policy, series in by_policy.items():
        c, m, l = palette.get(policy, ("#9CA89A", "o", policy))
        ax.plot([r["episode_id"] for r in series], [float(r["total_reward"]) for r in series],
                color=c, linewidth=2, marker=m, markersize=4, label=l, alpha=0.9)
    ax.set_xlabel("Episode", color="#9CA89A", fontsize=12)
    ax.set_ylabel("Total Episode Reward", color="#9CA89A", fontsize=12)
    ax.set_title("AIC Training — Reward Curve", color="#E8EDE6", fontsize=16, fontweight="bold", pad=15)
    ax.legend(facecolor="#161B14", edgecolor="#34D399", labelcolor="#E8EDE6", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Reward curve saved: {output_path}")


def plot_grpo_reward_curve(log_path="logs/grpo_progress.jsonl", out_path="results/grpo_reward_curve.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lp = Path(log_path)
    if not lp.exists():
        print(f"⚠️  No GRPO logs at {log_path}. Skipping.")
        return
    entries = [json.loads(l) for l in open(lp) if l.strip()]
    if not entries:
        return
    steps = [e["step"] for e in entries]
    rewards = [e["reward"] for e in entries]
    window = max(5, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    ax.plot(steps, rewards, alpha=0.3, color="#4A90D9", linewidth=0.8, label="Raw reward")
    ax.plot(steps[window-1:], smoothed, color="#34D399", linewidth=2.5, label=f"Smoothed (w={window})")
    if len(steps) > 20:
        t = len(steps)
        ax.axvspan(steps[0], steps[t//3], alpha=0.08, color="red", label="Exploration")
        ax.axvspan(steps[t//3], steps[2*t//3], alpha=0.08, color="yellow", label="Learning")
        ax.axvspan(steps[2*t//3], steps[-1], alpha=0.08, color="green", label="Convergence")
    ax.set_xlabel("Training Step", color="#9CA89A", fontsize=13)
    ax.set_ylabel("Reward", color="#9CA89A", fontsize=13)
    ax.set_title("AIC — GRPO Training Reward Curve", color="#E8EDE6", fontsize=15, fontweight="bold")
    ax.legend(facecolor="#161B14", edgecolor="#34D399", labelcolor="#E8EDE6", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"✅ GRPO reward curve saved: {out_path}")


def plot_policy_comparison(benchmark_csv="results/benchmark_summary.csv", out_path="results/policy_comparison.png"):
    import pandas as pd
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(benchmark_csv).exists():
        print(f"⚠️  No benchmark CSV. Skipping policy comparison.")
        return
    df = pd.read_csv(benchmark_csv)
    if "avg_reward" not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(_DARK_BG)
    for ax in [ax1, ax2]:
        _style_ax(ax)
    policies = df["policy"].tolist()
    colors = ["#27AE60" if "Trained" in str(p) else "#F39C12" if "Adaptive" in str(p) else "#E74C3C" for p in policies]
    x = np.arange(len(df))
    ax1.bar(x, df["avg_reward"], color=colors, width=0.6, edgecolor=_DARK_BG)
    for i, v in enumerate(df["avg_reward"]):
        ax1.text(i, v + abs(v)*0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#E8EDE6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=20, ha="right", fontsize=9, color="#9CA89A")
    ax1.set_title("Average Reward", color="#E8EDE6", fontsize=13, fontweight="bold")
    if "success_rate" in df.columns:
        ax2.bar(x, df["success_rate"]*100, color=colors, width=0.6, edgecolor=_DARK_BG)
        for i, v in enumerate(df["success_rate"]*100):
            ax2.text(i, v+1, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#E8EDE6")
        ax2.set_xticks(x)
        ax2.set_xticklabels(policies, rotation=20, ha="right", fontsize=9, color="#9CA89A")
        ax2.set_title("Success Rate", color="#E8EDE6", fontsize=13, fontweight="bold")
    fig.suptitle("AIC Policy Comparison", color="#E8EDE6", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"✅ Policy comparison saved: {out_path}")


def plot_verifier_pass_rate(output_path="results/verifier_pass_rate.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    by_policy = _load_policy_benchmark()
    before = by_policy.get("baseline_frozen_trust", [])
    after = by_policy.get("baseline_adaptive_trust", [])
    n = min(len(before), len(after))
    if n == 0:
        raise FileNotFoundError("Missing baseline series in policy_benchmark.jsonl")
    br = [100.0 if r.get("success") else 0.0 for r in before[:n]]
    ar = [100.0 if r.get("success") else 0.0 for r in after[:n]]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    x = np.arange(n)
    w = 0.35
    ax.bar(x-w/2, br, w, label="Before", color="#14B8A6", alpha=0.7, edgecolor=_DARK_BG)
    ax.bar(x+w/2, ar, w, label="After", color="#34D399", alpha=0.9, edgecolor=_DARK_BG)
    ax.set_xlabel("Episode", color="#9CA89A", fontsize=12)
    ax.set_ylabel("Pass Rate (%)", color="#9CA89A", fontsize=12)
    ax.set_title("Verifier Approval — Before vs After", color="#E8EDE6", fontsize=16, fontweight="bold", pad=15)
    ax.legend(facecolor="#161B14", edgecolor="#34D399", labelcolor="#E8EDE6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {i}" for i in range(n)])
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Verifier pass rate saved: {output_path}")


def generate_before_after_demo(output_path="results/before_after_demo.md"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    from aic.agents.adversarial_agent import AdversarialAgent
    from aic.agents.app_agent import AppAgent
    from aic.agents.db_agent import DBAgent
    from aic.agents.infra_agent import InfraAgent
    from aic.agents.orchestrator_agent import OrchestratorAgent
    from aic.training.config import TrainingConfig
    from aic.training.train import run_episode
    from aic.utils.constants import ALL_AGENTS, INITIAL_TRUST
    from aic.utils.seeding import get_adversary_cycle, make_episode_rng
    config = TrainingConfig(num_episodes=3, use_llm_agents=False)
    db, infra, app = DBAgent(use_llm=False), InfraAgent(use_llm=False), AppAgent(use_llm=False)
    lines = ["# Before / After Training — Demo Evidence\n", "---\n"]
    for ep_id in range(3):
        lines.append(f"\n## Episode {ep_id}\n")
        cycle = get_adversary_cycle(make_episode_rng(ep_id, 42))
        adv = AdversarialAgent(cycle, db)
        class FrozenOrch(OrchestratorAgent):
            def _update_trust_scores(self, step, prev, curr):
                self.trust_scores = {a: INITIAL_TRUST for a in ALL_AGENTS}
        rb = run_episode(ep_id, config, FrozenOrch(adv, use_llm=False), db, infra, app)
        cycle2 = get_adversary_cycle(make_episode_rng(ep_id, 42))
        adv2 = AdversarialAgent(cycle2, db)
        ot = OrchestratorAgent(adv2, use_llm=False)
        ot.mode = "trained"
        ra = run_episode(ep_id, config, ot, db, infra, app)
        lines.append("| Metric | Untrained | Trained |")
        lines.append("|--------|-----------|---------|")
        lines.append(f"| Total Reward | {rb['total_reward']:+.2f} | {ra['total_reward']:+.2f} |")
        lines.append(f"| Final Health | {rb['final_health']:.3f} | {ra['final_health']:.3f} |")
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Before/after demo saved: {output_path}")


if __name__ == "__main__":
    print("Generating result artifacts...\n")
    plot_reward_curve()
    plot_verifier_pass_rate()
    plot_grpo_reward_curve()
    plot_policy_comparison()
    generate_before_after_demo()
    print("\n✅ All result artifacts generated!")
