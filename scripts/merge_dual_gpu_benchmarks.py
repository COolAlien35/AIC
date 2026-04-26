#!/usr/bin/env python3
"""
Merge two benchmark episode CSVs (dual-GPU / dual checkpoint), deduplicate
deterministic baselines, add min--max [0,1] reward columns, recompute
summaries, statistical tests, and write provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPS = 1e-9


def load_episodes(path: Path, training_run_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["training_run_id"] = int(training_run_id)
    return df


def dedup_baselines(
    r1: pd.DataFrame, r2: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """If frozen/adaptive rows match between runs, keep one copy in merged set."""
    info: dict[str, Any] = {
        "baseline_duplicate_policy": "drop_run2_on_match",
    }
    pols = ("baseline_frozen", "baseline_adaptive")
    m1 = r1[r1["policy"].isin(pols)].copy()
    m2 = r2[r2["policy"].isin(pols)].copy()
    if m1.empty or m2.empty:
        merged = pd.concat([r1, r2], ignore_index=True)
        info["deduped"] = False
        return merged, info
    c1 = m1.sort_values(
        by=["policy", "scenario", "episode_index", "reward"]
    ).reset_index(drop=True)
    c2 = m2.sort_values(
        by=["policy", "scenario", "episode_index", "reward"]
    ).reset_index(drop=True)
    if c1.shape == c2.shape and np.allclose(
        c1["reward"].values, c2["reward"].values, rtol=0, atol=0
    ):
        t2 = r2[r2["policy"] == "trained_grpo"]
        t1 = r1[r1["policy"] == "trained_grpo"]
        out = pd.concat(
            [m1, t1, t2], ignore_index=True
        )
        info["deduped"] = True
        info["dropped_baseline_rows"] = int(len(m2))
        return out, info
    out = pd.concat([r1, r2], ignore_index=True)
    info["deduped"] = False
    return out, info


def add_reward_minmax_01(
    df: pd.DataFrame, reward_col: str = "reward"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """``reward_01 = (r_max - r) / (r_max - r_min + eps)`` (higher reward = more positive = better 1)."""
    r = df[reward_col].astype(float)
    r_max = float(r.max())
    r_min = float(r.min())
    num = r_max - r
    den = r_max - r_min + EPS
    df = df.copy()
    df["reward_minmax_01"] = (num / den).clip(0, 1)
    cfg = {
        "reward_col": reward_col,
        "r_min": r_min,
        "r_max": r_max,
        "formula": f"(r_max - {reward_col}) / (r_max - r_min + {EPS})",
        "interpretation": "1.0 = best (least negative) reward in merged set, 0.0 = worst",
    }
    return df, cfg


def aggregate_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = (
        df.groupby("policy", sort=False)
        .agg(
            avg_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            success_rate=("success", "mean"),
            num_episodes=("reward", "count"),
        )
        .reset_index()
    )
    by_sc = (
        df.groupby(["policy", "scenario"], sort=False)
        .agg(avg_reward=("reward", "mean"), success_rate=("success", "mean"))
        .reset_index()
    )
    return summ, by_sc


def ttest_and_d(baseline: np.ndarray, trained: np.ndarray) -> dict[str, float]:
    if len(baseline) < 2 or len(trained) < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "significant": False,
            "effect_size_label": "n/a",
            "baseline_mean": float(np.mean(baseline)) if len(baseline) else None,
            "trained_mean": float(np.mean(trained)) if len(trained) else None,
            "improvement": None,
            "improvement_pct": None,
        }
    t_stat, p_value = stats.ttest_ind(baseline, trained)
    pooled = np.sqrt(
        (np.std(baseline, ddof=1) ** 2 + np.std(trained, ddof=1) ** 2) / 2
    )
    d = (np.mean(trained) - np.mean(baseline)) / (pooled + EPS)
    bm = float(np.mean(baseline))
    tm = float(np.mean(trained))
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "significant": bool(p_value < 0.05),
        "effect_size_label": (
            "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        ),
        "baseline_mean": bm,
        "trained_mean": tm,
        "improvement": float(tm - bm),
        "improvement_pct": float((tm - bm) / (abs(bm) + EPS) * 100.0),
    }


def per_run_trained_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rid, g in df.groupby("training_run_id", sort=True):
        tr = g[g["policy"] == "trained_grpo"]["reward"].values
        if len(tr):
            out.append(
                {
                    "training_run_id": int(rid),
                    "trained_mean": float(np.mean(tr)),
                    "trained_std": float(np.std(tr, ddof=1) if len(tr) > 1 else 0.0),
                    "n": int(len(tr)),
                }
            )
    return out


def run_merge(
    run1: Path, run2: Path, out_dir: Path, copy_newone_stats: Path | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = load_episodes(run1, 1)
    b = load_episodes(run2, 2)
    merged, dedup = dedup_baselines(a, b)
    merged = merged.reset_index(drop=True)
    norm_cfg: dict[str, Any]
    merged_norm, ncfg = add_reward_minmax_01(merged, "reward")
    norm_cfg = ncfg
    for col in ("mttr", "adversary_suppression", "unsafe_rate"):
        if col in merged_norm.columns and merged_norm[col].notna().any():
            sub = merged_norm[merged_norm[col].notna()]
            if not sub.empty:
                x = sub[col].astype(float)
                lo, hi = float(x.min()), float(x.max())
                if hi > lo + EPS:
                    name = f"{col}_minmax_01"
                    merged_norm[name] = (x - lo) / (hi - lo)
                    norm_cfg[name] = {"col": col, "lo": lo, "hi": hi}
    summ, bysc = aggregate_tables(merged)
    br = merged[merged["policy"] == "baseline_frozen"]["reward"].values
    tr = merged[merged["policy"] == "trained_grpo"]["reward"].values
    st = ttest_and_d(br, tr)
    merged.to_csv(out_dir / "benchmark_episodes_long.csv", index=False)
    merged_norm.to_csv(
        out_dir / "benchmark_episodes_long_normalized.csv", index=False
    )
    summ.to_csv(out_dir / "benchmark_summary_merged.csv", index=False)
    bysc.to_csv(out_dir / "benchmark_by_scenario_merged.csv", index=False)
    with open(out_dir / "statistical_test_merged.json", "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    (out_dir / "normalization_config.json").write_text(
        json.dumps(
            {**norm_cfg, "eps": EPS},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prov = {
        "run1_csv": str(run1.resolve()),
        "run2_csv": str(run2.resolve()),
        "deduplication": dedup,
        "n_rows_merged": int(len(merged)),
        "per_run_trained": per_run_trained_stats(merged),
    }
    if copy_newone_stats and copy_newone_stats.exists():
        prov["newone_statistical_test_reference"] = json.loads(
            copy_newone_stats.read_text(encoding="utf-8")
        )
    (out_dir / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    by_run = (
        merged.groupby(["training_run_id", "policy"], sort=True)
        .agg(avg_reward=("reward", "mean"), n=("reward", "count"))
        .reset_index()
    )
    by_run.to_csv(out_dir / "benchmark_summary_by_run.csv", index=False)

    # Brief insights
    sc_gap = bysc[bysc["policy"] == "trained_grpo"].merge(
        bysc[bysc["policy"] == "baseline_frozen"],
        on="scenario",
        suffixes=("_t", "_b"),
    )
    if not sc_gap.empty:
        sc_gap["delta"] = sc_gap["avg_reward_t"] - sc_gap["avg_reward_b"]
        top = sc_gap.sort_values("delta", ascending=False)
        insights = {
            "largest_trained_advantage_by_scenario": top[
                ["scenario", "delta", "avg_reward_b", "avg_reward_t"]
            ]
            .head(6)
            .to_dict("records"),
            "n_episodes_pooled": int(len(merged)),
            "ttest_pooled_p_value": st["p_value"],
            "cohens_d_pooled": st["cohens_d"],
        }
        (out_dir / "insights_merged.json").write_text(
            json.dumps(insights, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[ok] Merged -> {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--run1", type=Path, default=Path("data/benchmark_ingest/run1_episodes_from_fixture.csv"))
    p.add_argument("--run2", type=Path, default=Path("data/benchmark_ingest/run2_episodes_from_fixture.csv"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmark_merged"),
    )
    p.add_argument(
        "--ref-newone-stats",
        type=Path,
        default=Path("newone/statistical_test.json"),
        help="Optional: embed newone/ GPU-2 stats in provenance for cross-check.",
    )
    args = p.parse_args()
    ref = args.ref_newone_stats if args.ref_newone_stats.exists() else None
    run_merge(args.run1, args.run2, args.out, ref)


if __name__ == "__main__":
    main()
