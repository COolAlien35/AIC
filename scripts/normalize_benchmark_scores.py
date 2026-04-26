#!/usr/bin/env python3
"""Create derived 0-1 benchmark scores from raw rewards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize(input_dir: str = "results_final") -> dict:
    out = Path(input_dir)
    scenario_path = out / "benchmark_by_scenario.csv"
    summary_path = out / "benchmark_summary.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"missing {scenario_path}")

    scenario_df = pd.read_csv(scenario_path)
    if "avg_reward" not in scenario_df.columns:
        raise ValueError(f"{scenario_path} must include avg_reward")

    def _norm_group(group: pd.DataFrame) -> pd.DataFrame:
        lo = float(group["avg_reward"].min())
        hi = float(group["avg_reward"].max())
        if hi == lo:
            group["score_0_1"] = 0.5
        else:
            group["score_0_1"] = (group["avg_reward"] - lo) / (hi - lo)
        return group

    scenario_norm = (
        scenario_df.groupby("scenario", group_keys=False)
        .apply(_norm_group)
        .reset_index(drop=True)
    )
    scenario_norm.to_csv(out / "benchmark_by_scenario_normalized.csv", index=False)

    summary_norm = scenario_norm.groupby("policy").agg(
        avg_reward=("avg_reward", "mean"),
        avg_score_0_1=("score_0_1", "mean"),
        success_rate=("success_rate", "mean"),
    ).reset_index()
    if summary_path.exists():
        raw_summary = pd.read_csv(summary_path)
        keep = [c for c in ("policy", "std_reward", "num_episodes") if c in raw_summary.columns]
        if len(keep) > 1:
            summary_norm = summary_norm.merge(raw_summary[keep], on="policy", how="left")
    summary_norm.to_csv(out / "benchmark_summary_normalized.csv", index=False)

    result = {
        "input_dir": str(out),
        "scenario_rows": int(len(scenario_norm)),
        "summary_rows": int(len(summary_norm)),
        "best_policy_by_score": (
            str(summary_norm.sort_values("avg_score_0_1", ascending=False).iloc[0]["policy"])
            if len(summary_norm)
            else None
        ),
    }
    with open(out / "normalized_score_manifest.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results_final")
    args = parser.parse_args()
    result = normalize(args.input_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

