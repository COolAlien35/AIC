#!/usr/bin/env python3
"""Build dashboard/site/data.js from the canonical results files.

Run this whenever results/* change so the static dashboard stays in sync
with the source-of-truth artefacts. The output is a single JS file that
exposes ``window.AIC_DATA``, so the dashboard works straight from a
``file://`` URL with no server.
"""
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
SITE = pathlib.Path(__file__).resolve().parent
PLOTS_OUT = SITE / "plots"


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _opt_load_json(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    out: dict = {}

    # GRPO progress (drop the trailing zero-row that the writer appends)
    progress_path = REPO / "logs" / "grpo_progress.jsonl"
    grpo_rows: list[dict] = []
    if progress_path.exists():
        with progress_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("step") and r.get("reward") and r["reward"] != 0:
                    grpo_rows.append(
                        {
                            "step": r["step"],
                            "reward": round(r["reward"], 4),
                            "loss": round(r.get("loss", 0), 6),
                            "kl": round(r.get("kl", 0), 6),
                            "elapsed_minutes": round(r.get("elapsed_minutes", 0), 2),
                        }
                    )
    out["grpo_progress"] = grpo_rows
    out["grpo_summary"] = _opt_load_json(REPO / "results" / "grpo_training_summary.json") or {}

    # Merged benchmark
    merged = REPO / "results" / "benchmark_merged"
    out["benchmark_summary"] = [
        {
            "policy": r["policy"],
            "avg_reward": float(r["avg_reward"]),
            "std_reward": float(r["std_reward"]),
            "num_episodes": int(r["num_episodes"]),
        }
        for r in _read_csv(merged / "benchmark_summary_merged.csv")
    ]
    out["benchmark_by_scenario"] = [
        {
            "policy": r["policy"],
            "scenario": r["scenario"],
            "avg_reward": float(r["avg_reward"]),
        }
        for r in _read_csv(merged / "benchmark_by_scenario_merged.csv")
    ]
    out["episodes"] = [
        {
            "policy": r["policy"],
            "scenario": r["scenario"],
            "reward": float(r["reward"]),
            "training_run_id": int(r["training_run_id"]),
        }
        for r in _read_csv(merged / "benchmark_episodes_long.csv")
    ]
    out["stats"] = _opt_load_json(merged / "statistical_test_merged.json") or {}
    out["insights"] = _opt_load_json(merged / "insights_merged.json") or {}
    out["normalization"] = _opt_load_json(merged / "normalization_config.json") or {}
    out["provenance"] = _opt_load_json(merged / "provenance.json") or {}

    # Task graders
    out["task_grader"] = _opt_load_json(REPO / "results" / "inference_summary.json") or {}
    tg_csv = REPO / "results" / "benchmark_by_task_grader.csv"
    out["task_grader_table"] = (
        [
            {
                "policy": r["policy"],
                "task_id": r["task_id"],
                "difficulty": r["difficulty"],
                "scenario_name": r["scenario_name"],
                "mean_score_0_1": float(r["mean_score_0_1"]),
                "success_threshold": float(r["success_threshold"]),
            }
            for r in _read_csv(tg_csv)
        ]
        if tg_csv.exists()
        else []
    )

    # Persist
    js = "window.AIC_DATA = " + json.dumps(out, indent=2) + ";\n"
    (SITE / "data.js").write_text(js)

    # Copy / refresh plots
    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in (merged / "plots").glob("*.png"):
        shutil.copy2(src, PLOTS_OUT / src.name)
        copied += 1
    for name in ("grpo_reward_curve.png", "grpo_loss_curve.png", "grpo_kl_curve.png"):
        src = REPO / "results" / name
        if src.exists():
            shutil.copy2(src, PLOTS_OUT / name)
            copied += 1

    print(
        f"✓ wrote {SITE/'data.js'}  "
        f"({len(grpo_rows)} GRPO steps, {len(out['episodes'])} episodes, "
        f"{copied} plots refreshed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
