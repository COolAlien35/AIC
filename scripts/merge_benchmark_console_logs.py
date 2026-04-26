#!/usr/bin/env python3
"""Merge benchmark console logs from two runs into benchmark CSVs.

This is for the case where you only have stdout logs (not per-episode CSVs).

Input format (must match `scripts/run_final_benchmark.py` prints):

  [bench] Benchmarking: <policy>
    <Scenario Name> ep<idx>: reward=<float>, success=<True|False>

We parse two runs, concatenate episodes, then recompute:
  - results_final/benchmark_episodes_merged.csv
  - results_final/benchmark_by_scenario.csv
  - results_final/benchmark_summary.csv
  - results_final/statistical_test.json
  - results_final/benchmark_run_config.json
  - results_final/*_normalized.csv (+ manifest)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_POLICY_RE = re.compile(r"^\[bench\]\s+Benchmarking:\s+(?P<policy>\S+)\s*$")
_EP_RE = re.compile(
    r"^\s*(?P<scenario>.+?)\s+ep(?P<ep>\d+):\s+reward=(?P<reward>-?\d+(?:\.\d+)?),\s+success=(?P<success>True|False)\s*$"
)


@dataclass(frozen=True)
class ParsedRun:
    run_id: str
    df: pd.DataFrame


def _parse_text(text: str, run_id: str) -> ParsedRun:
    policy: str | None = None
    rows: list[dict] = []

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        mpol = _POLICY_RE.match(line)
        if mpol:
            policy = mpol.group("policy")
            continue

        mep = _EP_RE.match(line)
        if mep and policy is not None:
            rows.append(
                {
                    "run_id": run_id,
                    "policy": policy,
                    "scenario": mep.group("scenario").strip(),
                    "episode_idx": int(mep.group("ep")),
                    "reward": float(mep.group("reward")),
                    "success": (mep.group("success") == "True"),
                }
            )

    if not rows:
        raise ValueError(
            f"No episodes parsed for {run_id}. Ensure the pasted logs include "
            "`[bench] Benchmarking:` lines and per-episode lines like "
            "`Cache Stampede ep0: reward=-123.45, success=False`."
        )

    df = pd.DataFrame(rows)
    # Stable ordering for reproducibility.
    df = df.sort_values(["policy", "scenario", "episode_idx"]).reset_index(drop=True)
    return ParsedRun(run_id=run_id, df=df)


def _stat_test(df: pd.DataFrame) -> dict:
    baseline = df[df["policy"] == "baseline_frozen"]["reward"].to_numpy(dtype=float)
    trained = df[df["policy"] == "trained_grpo"]["reward"].to_numpy(dtype=float)

    have_baseline = baseline.size > 0
    have_trained = trained.size > 0
    out: dict = {
        "baseline_mean": float(np.mean(baseline)) if have_baseline else None,
        "trained_mean": float(np.mean(trained)) if have_trained else None,
        "improvement": (float(np.mean(trained) - np.mean(baseline)) if (have_baseline and have_trained) else None),
        "improvement_pct": (
            float((np.mean(trained) - np.mean(baseline)) / (abs(np.mean(baseline)) + 1e-9) * 100)
            if (have_baseline and have_trained)
            else None
        ),
        "t_statistic": 0.0,
        "p_value": 1.0,
        "significant": False,
        "cohens_d": 0.0,
        "effect_size_label": "small",
        "note": "Stats computed over merged per-episode rewards (runs concatenated).",
    }

    if have_baseline and have_trained and baseline.size >= 2 and trained.size >= 2:
        try:
            from scipy import stats  # type: ignore

            t_stat, p_value = stats.ttest_ind(baseline, trained)
            pooled_std = np.sqrt((np.std(baseline) ** 2 + np.std(trained) ** 2) / 2)
            cohens_d = (np.mean(trained) - np.mean(baseline)) / (pooled_std + 1e-9)
            out["t_statistic"] = float(t_stat)
            out["p_value"] = float(p_value)
            out["significant"] = bool(p_value < 0.05)
            out["cohens_d"] = float(cohens_d)
            out["effect_size_label"] = (
                "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
            )
        except Exception as e:
            out["note"] = f"{out['note']} (scipy unavailable or failed: {e})"

    return out


def _normalize_0_1(out_dir: Path) -> dict:
    scenario_path = out_dir / "benchmark_by_scenario.csv"
    summary_path = out_dir / "benchmark_summary.csv"

    scenario_df = pd.read_csv(scenario_path)

    def _norm_group(group: pd.DataFrame) -> pd.DataFrame:
        lo = float(group["avg_reward"].min())
        hi = float(group["avg_reward"].max())
        if hi == lo:
            group["score_0_1"] = 0.5
        else:
            group["score_0_1"] = (group["avg_reward"] - lo) / (hi - lo)
        return group

    scenario_norm = scenario_df.groupby("scenario", group_keys=False).apply(_norm_group).reset_index(drop=True)
    scenario_norm.to_csv(out_dir / "benchmark_by_scenario_normalized.csv", index=False)

    summary_norm = (
        scenario_norm.groupby("policy")
        .agg(
            avg_reward=("avg_reward", "mean"),
            avg_score_0_1=("score_0_1", "mean"),
            success_rate=("success_rate", "mean"),
        )
        .reset_index()
    )
    if summary_path.exists():
        raw_summary = pd.read_csv(summary_path)
        keep = [c for c in ("policy", "std_reward", "num_episodes") if c in raw_summary.columns]
        if len(keep) > 1:
            summary_norm = summary_norm.merge(raw_summary[keep], on="policy", how="left")
    summary_norm.to_csv(out_dir / "benchmark_summary_normalized.csv", index=False)

    manifest = {
        "input_dir": str(out_dir),
        "scenario_rows": int(len(scenario_norm)),
        "summary_rows": int(len(summary_norm)),
        "best_policy_by_score": (
            str(summary_norm.sort_values("avg_score_0_1", ascending=False).iloc[0]["policy"])
            if len(summary_norm)
            else None
        ),
    }
    (out_dir / "normalized_score_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def merge(run_a_text: str, run_b_text: str, out_dir: str = "results_final") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_a = _parse_text(run_a_text, run_id="run_a")
    run_b = _parse_text(run_b_text, run_id="run_b")

    merged = pd.concat([run_a.df, run_b.df], ignore_index=True)

    # Save merged per-episode table (this is the “ground truth” for any summary).
    merged_path = out / "benchmark_episodes_merged.csv"
    merged.to_csv(merged_path, index=False)

    summary = (
        merged.groupby("policy")
        .agg(
            avg_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            success_rate=("success", "mean"),
            num_episodes=("reward", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out / "benchmark_summary.csv", index=False)

    by_scenario = (
        merged.groupby(["policy", "scenario"])
        .agg(
            avg_reward=("reward", "mean"),
            success_rate=("success", "mean"),
            num_episodes=("reward", "count"),
        )
        .reset_index()
    )
    by_scenario.to_csv(out / "benchmark_by_scenario.csv", index=False)

    stats_out = _stat_test(merged)
    (out / "statistical_test.json").write_text(json.dumps(stats_out, indent=2))

    run_cfg = {
        "source": "merged_from_console_logs",
        "runs": {
            "run_a": {"episodes_per_scenario": int(run_a.df["episode_idx"].nunique())},
            "run_b": {"episodes_per_scenario": int(run_b.df["episode_idx"].nunique())},
        },
        "total_rows": int(len(merged)),
        "policies": sorted(merged["policy"].unique().tolist()),
        "scenarios": sorted(merged["scenario"].unique().tolist()),
        "output_files": [
            "benchmark_episodes_merged.csv",
            "benchmark_by_scenario.csv",
            "benchmark_summary.csv",
            "statistical_test.json",
            "benchmark_by_scenario_normalized.csv",
            "benchmark_summary_normalized.csv",
            "normalized_score_manifest.json",
        ],
    }
    (out / "benchmark_run_config.json").write_text(json.dumps(run_cfg, indent=2))

    _normalize_0_1(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="Path to RUN_A stdout log text file")
    parser.add_argument("--run-b", required=True, help="Path to RUN_B stdout log text file")
    parser.add_argument("--out", default="results_final", help="Output directory (default: results_final)")
    args = parser.parse_args()

    run_a_text = Path(args.run_a).read_text(encoding="utf-8", errors="replace")
    run_b_text = Path(args.run_b).read_text(encoding="utf-8", errors="replace")
    out = merge(run_a_text, run_b_text, out_dir=args.out)
    print(f"[ok] wrote merged benchmarks to {out}")


if __name__ == "__main__":
    main()

