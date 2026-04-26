#!/usr/bin/env python3
"""
Generate evaluation figures from results/benchmark_merged/benchmark_episodes_long*.csv
and write figures_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_DARK_BG = "#0C0F0A"
_PLOT_BG = "#111610"


def _style_ax(ax):
    ax.set_facecolor(_PLOT_BG)
    ax.tick_params(colors="#6B7A68")
    ax.grid(True, alpha=0.15, color="#34D399")
    for s in ax.spines.values():
        s.set_color("#1E2B1A")


def _ci95(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    return 1.96 * float(v.std(ddof=1) / max(np.sqrt(n), 1e-9))


def plot_headline_bar(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    pols = sorted(d["policy"].unique(), key=str)
    means = [d[d["policy"] == p]["reward"].mean() for p in pols]
    c = [len(d[d["policy"] == p]) for p in pols]
    err = [_ci95(d[d["policy"] == p]["reward"].values) for p in pols]
    m = pd.Series(means, index=pols)
    c = pd.Series(c, index=pols)
    err = pd.Series(err, index=pols)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    x = np.arange(len(m))
    col = ["#E74C3C" if "frozen" in str(p) else "#F39C12" if "adaptive" in str(p) else "#27AE60" for p in m.index]
    ax.bar(x, m.values, yerr=err.values, capsize=4, color=col, edgecolor=_DARK_BG, width=0.55, ecolor="#9CA89A")
    ax.set_xticks(x)
    ax.set_xticklabels(m.index, rotation=18, ha="right", color="#9CA89A", fontsize=10)
    ax.set_ylabel("Mean reward (higher = better)", color="#9CA89A", fontsize=11)
    ax.set_title("Policy comparison — mean reward with 95% normal approx. CI", color="#E8EDE6", fontsize=14, fontweight="bold")
    for i, (p, v, n) in enumerate(zip(m.index, m.values, c.values)):
        ax.text(i, v + (abs(v) * 0.02 + 1), f"{v:.1f}\n(n={int(n)})", ha="center", color="#E8EDE6", fontsize=9)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_box_strip(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    pols = d["policy"].unique().tolist()
    data = [d[d["policy"] == p]["reward"].values for p in pols]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    pos = np.arange(1, len(pols) + 1)
    bp = ax.boxplot(
        data, positions=pos, patch_artist=True, widths=0.35,
        boxprops={"facecolor": "#1E2B1A", "edgecolor": "#34D399"},
        medianprops={"color": "#34D399"},
        flierprops={"markerfacecolor": "#9CA89A", "markersize": 4, "alpha": 0.4},
    )
    for i, p in enumerate(pols):
        y = d[d["policy"] == p]["reward"].values
        j = pos[i] + np.random.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(j, y, s=10, color="#6B7A68", alpha=0.55, zorder=3, edgecolors="none")
    ax.set_xticks(pos, pols, rotation=18, ha="right", color="#9CA89A", fontsize=9)
    ax.set_ylabel("Episode reward", color="#9CA89A", fontsize=11)
    ax.set_title("Distribution + all episodes (strip)", color="#E8EDE6", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_violin_trained_runs(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    d = d[d["policy"] == "trained_grpo"]
    if d.empty or "training_run_id" not in d.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    parts = [d[d["training_run_id"] == r]["reward"].values for r in sorted(d["training_run_id"].unique())]
    labels = [f"run {r}" for r in sorted(d["training_run_id"].unique())]
    v = ax.violinplot(parts, showmeans=True, showextrema=True)
    for b in v["bodies"]:
        b.set_facecolor("#27AE60")
        b.set_alpha(0.45)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, color="#9CA89A")
    ax.set_ylabel("Reward (trained only)", color="#9CA89A", fontsize=11)
    ax.set_title("Trained policy — spread across training runs (GPUs / checkpoints)", color="#E8EDE6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    p = d.pivot_table(index="scenario", columns="policy", values="reward", aggfunc="mean")
    p = p.reindex(sorted(p.index))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    im = ax.imshow(p.values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index, color="#9CA89A", fontsize=8)
    ax.set_xticks(np.arange(p.shape[1]))
    ax.set_xticklabels(p.columns, rotation=25, ha="right", color="#9CA89A", fontsize=8)
    ax.set_title("Mean reward: scenario × policy", color="#E8EDE6", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors="#9CA89A")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_dumbbell(by_sc: Path, out: Path) -> None:
    d = pd.read_csv(by_sc)
    t = d[d["policy"] == "trained_grpo"].set_index("scenario")["avg_reward"]
    b = d[d["policy"] == "baseline_frozen"].set_index("scenario")["avg_reward"]
    s = sorted(set(t.index) & set(b.index))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    y = np.arange(len(s))
    for i, sc in enumerate(s):
        ax.plot([b[sc], t[sc]], [i, i], color="#5D6D7E", linewidth=1.4, zorder=1)
        ax.scatter(b[sc], i, s=60, zorder=2, color="#E74C3C", label="baseline_frozen" if i == 0 else "")
        ax.scatter(t[sc], i, s=60, zorder=2, color="#27AE60", label="trained_grpo" if i == 0 else "")
    ax.set_yticks(y, s, fontsize=8, color="#9CA89A")
    ax.set_xlabel("Mean reward (merged)", color="#9CA89A", fontsize=11)
    ax.set_title("Dumbbell: scenario-level baseline → trained (merged)", color="#E8EDE6", fontsize=12, fontweight="bold")
    h1 = patches.Patch(color="#E74C3C", label="baseline_frozen")
    h2 = patches.Patch(color="#27AE60", label="trained_grpo")
    ax.legend(handles=[h1, h2], facecolor="#161B14", labelcolor="#E8EDE6", loc="lower right")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar(by_sc: Path, out: Path) -> None:
    d = pd.read_csv(by_sc)
    t = d[d["policy"] == "trained_grpo"].set_index("scenario")["avg_reward"]
    b = d[d["policy"] == "baseline_frozen"].set_index("scenario")["avg_reward"]
    s = sorted(set(t.index) & set(b.index))
    delta = np.array([t[sc] - b[sc] for sc in s])
    o = np.argsort(delta)
    s2 = [s[i] for i in o]
    d2 = delta[o]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    c = np.where(d2 > 0, "#27AE60", "#C0392B")
    ax.barh(np.arange(len(s2)), d2, color=c, height=0.5, edgecolor=_DARK_BG)
    ax.set_yticks(np.arange(len(s2)), s2, fontsize=8, color="#9CA89A")
    ax.axvline(0, color="#6B7A68", linewidth=0.8)
    ax.set_xlabel("mean(trained) − mean(baseline_frozen)", color="#9CA89A", fontsize=10)
    ax.set_title("Per-scenario reward uplift (larger to the right = more improvement)", color="#E8EDE6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(a)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_ecdf(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    b = d[d["policy"] == "baseline_frozen"]["reward"].values
    t = d[d["policy"] == "trained_grpo"]["reward"].values
    if len(b) < 1 or len(t) < 1:
        return
    xb, yb = _ecdf(b)
    xt, yt = _ecdf(t)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    ax.step(xb, yb, where="post", color="#E74C3C", label="baseline_frozen", linewidth=1.5)
    ax.step(xt, yt, where="post", color="#27AE60", label="trained_grpo (pooled runs)", linewidth=1.5)
    ax.set_xlabel("Episode reward", color="#9CA89A", fontsize=11)
    ax.set_ylabel("ECDF", color="#9CA89A", fontsize=11)
    ax.set_title("ECDF: more mass to the right is better (less negative reward)", color="#E8EDE6", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#161B14", labelcolor="#E8EDE6", edgecolor="#34D399")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_kde(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    b = d[d["policy"] == "baseline_frozen"]["reward"].values
    t = d[d["policy"] == "trained_grpo"]["reward"].values
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    for arr, c, label in ((b, "#E74C3C", "baseline_frozen"), (t, "#27AE60", "trained")):
        if len(arr) < 2:
            continue
        gk = (arr.max() - arr.min()) / 25.0
        if gk < 1e-6:
            gk = 0.1
        xs = np.linspace(arr.min() - 3 * gk, arr.max() + 3 * gk, 200)
        try:
            from scipy import stats

            kde = stats.gaussian_kde(arr, bw_method=0.3)
            ys = kde(xs)
        except Exception:
            ax.hist(arr, density=True, bins=25, color=c, alpha=0.35, label=label)
            continue
        ax.plot(xs, ys, color=c, label=label, alpha=0.9, linewidth=1.8)
    ax.set_xlabel("Reward", color="#9CA89A", fontsize=11)
    ax.set_ylabel("Density", color="#9CA89A", fontsize=11)
    ax.set_title("KDE: pooled baseline vs trained reward", color="#E8EDE6", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#161B14", labelcolor="#E8EDE6", edgecolor="#34D399")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_line_by_scenario(by_sc: Path, out: Path) -> None:
    d = pd.read_csv(by_sc)
    w = d.pivot_table(index="scenario", columns="policy", values="avg_reward", aggfunc="mean")
    s = list(sorted(w.index))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    x = np.arange(len(s))
    m = {
        "baseline_frozen": "#E74C3C",
        "baseline_adaptive": "#F39C12",
        "trained_grpo": "#27AE60",
    }
    for p in w.columns:
        if p in m:
            ax.plot(x, w.loc[s, p].values, "o-", color=m.get(p, "#9CA89A"), label=p, markersize=4, linewidth=1.2)
    ax.set_xticks(x, s, rotation=20, ha="right", color="#9CA89A", fontsize=8)
    ax.set_ylabel("Mean reward (merged)", color="#9CA89A", fontsize=10)
    ax.set_title("Scenarios are separated — environment stress differs by incident type", color="#E8EDE6", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#161B14", labelcolor="#E8EDE6", edgecolor="#34D399", fontsize=7)
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_faceted_policy_run(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    if "training_run_id" not in d.columns:
        return
    g = d.groupby(["training_run_id", "policy"], sort=True)["reward"].apply(list)
    nrun = d["training_run_id"].nunique()
    npol = d["policy"].nunique()
    fig, ax = plt.subplots(nrun, 1, figsize=(8, 2.2 + 2.5 * nrun), sharex=True)
    if nrun == 1:
        ax = [ax]
    fig.patch.set_facecolor(_DARK_BG)
    for r_i, r in enumerate(sorted(d["training_run_id"].unique())):
        _style_ax(ax[r_i])
        sub = d[d["training_run_id"] == r]
        pols = sub["policy"].unique().tolist()
        data = [sub[sub["policy"] == p]["reward"].values for p in pols]
        pos = np.arange(1, len(pols) + 1)
        ax[r_i].boxplot(data, positions=pos, patch_artist=True, widths=0.35)
        ax[r_i].set_xticks(pos, pols, rotation=15, ha="right", color="#9CA89A", fontsize=8)
        ax[r_i].set_ylabel("reward", color="#9CA89A", fontsize=9)
        ax[r_i].set_title(f"training_run_id = {r}", color="#E8EDE6", fontsize=11, loc="left")
    ax[-1].set_xlabel("Policy", color="#9CA89A", fontsize=10)
    fig.suptitle("Reward by policy, faceted by training run (GPU / checkpoint)", color="#E8EDE6", fontsize=12, y=0.99, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_trained_paired_scatter(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    t = d[d["policy"] == "trained_grpo"]
    if t["training_run_id"].nunique() < 2:
        return
    r1 = t[t["training_run_id"] == 1]
    r2 = t[t["training_run_id"] == 2]
    m = r1.merge(
        r2,
        on=["scenario", "episode_index"],
        suffixes=("_1", "_2"),
    )
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    x = m["reward_1"]
    y = m["reward_2"]
    ax.scatter(x, y, c="#27AE60", s=32, alpha=0.6, edgecolors="#111610")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#6B7A68", linewidth=0.8)
    ax.set_xlabel("Trained run 1 reward", color="#9CA89A", fontsize=10)
    ax.set_ylabel("Trained run 2 reward", color="#9CA89A", fontsize=10)
    ax.set_title("Paired: same (scenario, episode) across runs", color="#E8EDE6", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_mean_diff(df: Path, out: Path) -> None:
    d = pd.read_csv(df)
    b = d[d["policy"] == "baseline_frozen"]["reward"].values
    t = d[d["policy"] == "trained_grpo"]["reward"].values
    if len(b) < 2 or len(t) < 2:
        return
    rng = np.random.default_rng(42)
    n_b = 2000
    obs = float(t.mean() - b.mean())
    outg: list[float] = []
    for _ in range(n_b):
        b_s = rng.choice(b, size=len(b), replace=True)
        t_s = rng.choice(t, size=len(t), replace=True)
        outg.append(float(t_s.mean() - b_s.mean()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(_DARK_BG)
    _style_ax(ax)
    ax.hist(outg, bins=50, color="#1E2B1A", edgecolor="#34D399", alpha=0.8)
    ax.axvline(obs, color="#E8EDE6", linewidth=1.2, label=f"observed Δ = {obs:.2f}")
    q = np.quantile(outg, [0.025, 0.975])
    ax.axvspan(q[0], q[1], color="#27AE60", alpha=0.12, label="95% boot. interval")
    ax.set_xlabel("mean_trained − mean_baseline (bootstrap resamples)", color="#9CA89A", fontsize=10)
    ax.set_ylabel("Count", color="#9CA89A", fontsize=10)
    ax.set_title("Appendix: bootstrap of pooled mean difference", color="#E8EDE6", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#161B14", labelcolor="#E8EDE6", edgecolor="#34D399", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _manifest_entry(name: str, cap: str, cols: str, primary: bool) -> dict[str, Any]:
    return {
        "file": f"plots/{name}",
        "caption": cap,
        "required_columns": cols,
        "primary": primary,
    }


def run_all(merged_dir: Path) -> None:
    dfp = merged_dir / "benchmark_episodes_long.csv"
    dfn = merged_dir / "benchmark_episodes_long_normalized.csv"
    bys = merged_dir / "benchmark_by_scenario_merged.csv"
    plots = merged_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    plot_headline_bar(dfp, plots / "fig01_headline_policy_bar_ci.png")
    manifest.append(
        _manifest_entry(
            "fig01_headline_policy_bar_ci.png",
            "Mean reward by policy with approx. 95% CI (merged episodes).",
            "policy, reward",
            True,
        )
    )
    plot_box_strip(dfp, plots / "fig02_box_strip.png")
    manifest.append(
        _manifest_entry(
            "fig02_box_strip.png",
            "Box + strip: show n and outlier pattern per policy.",
            "policy, reward",
            True,
        )
    )
    plot_violin_trained_runs(dfp, plots / "fig03_violin_trained_runs.png")
    manifest.append(
        _manifest_entry(
            "fig03_violin_trained_runs.png",
            "Trained policy only: spread across two training/benchmark runs.",
            "policy, training_run_id, reward",
            True,
        )
    )
    plot_heatmap(dfp, plots / "fig04_heatmap_scenario_policy.png")
    manifest.append(
        _manifest_entry(
            "fig04_heatmap_scenario_policy.png",
            "Scenario × policy mean reward (merged).",
            "scenario, policy, reward",
            True,
        )
    )
    if bys.exists():
        plot_dumbbell(bys, plots / "fig05_dumbbell_baseline_to_trained.png")
        plot_delta_bar(bys, plots / "fig06_delta_by_scenario.png")
        plot_line_by_scenario(bys, plots / "fig10_line_mean_by_scenario.png")
        manifest += [
            _manifest_entry(
                "fig05_dumbbell_baseline_to_trained.png",
                "Per-scenario baseline vs trained mean (merged aggregates).",
                "benchmark_by_scenario_merged: policy, scenario, avg_reward",
                True,
            ),
            _manifest_entry(
                "fig06_delta_by_scenario.png",
                "Per-scenario mean uplift: trained − frozen baseline.",
                "benchmark_by_scenario_merged: policy, scenario, avg_reward",
                True,
            ),
            _manifest_entry(
                "fig10_line_mean_by_scenario.png",
                "Lines across scenarios: benchmark differentiates conditions.",
                "benchmark_by_scenario_merged",
                True,
            ),
        ]
    plot_ecdf(dfp, plots / "fig07_ecdf_baseline_vs_trained.png")
    plot_kde(dfp, plots / "fig08_kde_baseline_vs_trained.png")
    manifest += [
        _manifest_entry("fig07_ecdf_baseline_vs_trained.png", "ECDF: stochastic ordering of rewards.", "policy, reward", True),
        _manifest_entry("fig08_kde_baseline_vs_trained.png", "KDE overlay: baseline vs trained (needs scipy in env).", "policy, reward", True),
    ]
    plot_faceted_policy_run(dfp, plots / "fig11_faceted_by_training_run.png")
    manifest.append(
        _manifest_entry("fig11_faceted_by_training_run.png", "Boxplots by run × policy (replication).", "policy, training_run_id, reward", True)
    )
    dchk = pd.read_csv(dfp)
    if dchk["training_run_id"].nunique() > 1:
        plot_trained_paired_scatter(dfp, plots / "fig12_paired_trained_runs.png")
        manifest.append(
            _manifest_entry(
                "fig12_paired_trained_runs.png",
                "Paired y=x plot for trained on matching (scenario, episode).",
                "training_run_id, scenario, episode_index, reward",
                False,
            )
        )
    plot_bootstrap_mean_diff(dfp, plots / "appendix_bootstrap_mean_diff.png")
    manifest.append(
        _manifest_entry(
            "appendix_bootstrap_mean_diff.png",
            "Appendix: bootstrap distribution of mean trained − mean baseline.",
            "policy, reward",
            False,
        )
    )
    (merged_dir / "figures_manifest.json").write_text(
        json.dumps(
            {
                "merged_dir": str(merged_dir),
                "figures": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] Wrote {len(manifest)} figures under {plots}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("results/benchmark_merged"))
    a = p.parse_args()
    run_all(a.dir)


if __name__ == "__main__":
    main()
