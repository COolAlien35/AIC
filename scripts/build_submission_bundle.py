#!/usr/bin/env python3
"""Build a single `submission/` folder with all artifacts.

Run this inside the HF Space (recommended) at `/workspace/aic-repo`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _write_json(dst: Path, obj: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, indent=2))


def _extra_plots(sub_dir: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dark = "#0C0F0A"
    bg = "#111610"

    by_scenario = sub_dir / "benchmarks" / "benchmark_by_scenario.csv"
    by_scenario_norm = sub_dir / "benchmarks" / "benchmark_by_scenario_normalized.csv"
    if not by_scenario.exists() or not by_scenario_norm.exists():
        return

    df = pd.read_csv(by_scenario)
    dfn = pd.read_csv(by_scenario_norm)

    # 1) Reward by scenario: baseline_frozen vs trained_grpo
    pivot = df.pivot_table(index="scenario", columns="policy", values="avg_reward", aggfunc="mean")
    if "baseline_frozen" in pivot.columns and "trained_grpo" in pivot.columns:
        scenarios = pivot.index.tolist()
        x = np.arange(len(scenarios))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(dark)
        ax.set_facecolor(bg)
        ax.bar(x - w / 2, pivot["baseline_frozen"].values, w, label="baseline_frozen", color="#F59E0B")
        ax.bar(x + w / 2, pivot["trained_grpo"].values, w, label="trained_grpo", color="#34D399")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right", color="#9CA89A")
        ax.tick_params(colors="#6B7A68")
        ax.set_title("Avg reward by scenario (baseline_frozen vs trained_grpo)", color="#E8EDE6")
        ax.legend(facecolor="#161B14", edgecolor="#34D399", labelcolor="#E8EDE6")
        out = sub_dir / "plots" / "reward_by_scenario_baseline_vs_trained.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close()

    # 2) Normalized score by scenario (0-1): all policies
    pivn = dfn.pivot_table(index="scenario", columns="policy", values="score_0_1", aggfunc="mean")
    scenarios = pivn.index.tolist()
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(dark)
    ax.set_facecolor(bg)
    colors = {"baseline_frozen": "#F59E0B", "baseline_adaptive": "#60A5FA", "trained_grpo": "#34D399"}
    for pol in pivn.columns:
        ax.plot(x, pivn[pol].values, marker="o", linewidth=2, label=pol, color=colors.get(pol, "#9CA89A"))
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", color="#9CA89A")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#6B7A68")
    ax.set_title("Normalized score (0-1) by scenario", color="#E8EDE6")
    ax.legend(facecolor="#161B14", edgecolor="#34D399", labelcolor="#E8EDE6")
    out = sub_dir / "plots" / "normalized_score_by_scenario.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def main() -> None:
    repo = Path(".").resolve()
    sub = repo / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True, exist_ok=True)

    # Specs + baseline script
    _copy_if_exists(repo / "openenv.yaml", sub / "openenv.yaml")
    _copy_if_exists(repo / "Dockerfile", sub / "Dockerfile")
    _copy_if_exists(repo / "README.md", sub / "README.md")
    _copy_if_exists(repo / "inference.py", sub / "inference.py")

    # Trained model evidence
    _copy_if_exists(repo / "exports", sub / "model" / "exports")
    _copy_if_exists(repo / "checkpoints" / "grpo", sub / "model" / "grpo_adapter")

    # Benchmarks (raw + normalized)
    for p in [
        repo / "results_final" / "benchmark_summary.csv",
        repo / "results_final" / "benchmark_by_scenario.csv",
        repo / "results_final" / "statistical_test.json",
        repo / "results_final" / "benchmark_summary_normalized.csv",
        repo / "results_final" / "benchmark_by_scenario_normalized.csv",
        repo / "results_final" / "normalized_score_manifest.json",
        repo / "results_final" / "inference_stdout.log",
    ]:
        _copy_if_exists(p, sub / "benchmarks" / p.name)

    # Plots
    for name in [
        "policy_comparison.png",
        "grpo_reward_curve.png",
        "reward_curve.png",
        "verifier_pass_rate.png",
    ]:
        _copy_if_exists(repo / "results" / name, sub / "plots" / name)

    _extra_plots(sub)

    # Manifest for quick inspection
    _write_json(
        sub / "manifest.json",
        {
            "created_from": str(repo),
            "contains": {
                "spec": ["openenv.yaml", "Dockerfile", "README.md", "inference.py"],
                "model": ["model/exports", "model/grpo_adapter"],
                "benchmarks": ["benchmarks/*.csv", "benchmarks/*.json", "benchmarks/inference_stdout.log"],
                "plots": ["plots/*.png"],
            },
        },
    )
    print(f"[ok] wrote {sub}")


if __name__ == "__main__":
    main()

