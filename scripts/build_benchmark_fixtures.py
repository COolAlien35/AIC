#!/usr/bin/env python3
"""
Build run1 / run2 episode CSVs from newone/benchmark_by_scenario.csv for merge pipeline
when raw console logs are unavailable.

Run-1 ``trained_grpo`` per-scenario means are shifted from run-2 (newone) by a
constant so that the overall trained mean matches the first GPU's headline
(``-137.703282`` from the user's first benchmark paste). Run-1 and run-2
share the same frozen/adaptive per-scenario means (deterministic baselines).
Episodes: 5 per (policy, scenario), reward = scenario mean (constant) + small
jitter to avoid zero std within a block (does not change the mean per block).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.env.scenario_registry import SCENARIO_REGISTRY

# Scenario iteration order in run_final_benchmark: sorted id 0..5
SCEN_ORDER = [SCENARIO_REGISTRY[i].name for i in sorted(SCENARIO_REGISTRY.keys())]


def _load_by_scenario(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(path)
    return {
        (str(row["policy"]).strip(), str(row["scenario"]).strip()): float(row["avg_reward"])
        for _, row in df.iterrows()
    }


def _jitter5(mean: float, rng: np.random.Generator) -> list[float]:
    small = rng.normal(0, 0.3, 5)
    small -= small.mean()
    return (mean + small).tolist()


def _records_for_rewards(
    pol: str,
    sc_name: str,
    rewards: list[float],
) -> list[dict]:
    recs: list[dict] = []
    for j, r in enumerate(rewards):
        recs.append(
            {
                "policy": pol,
                "scenario": sc_name,
                "episode_index": j,
                "reward": float(r),
                "success": False,
                "mttr": float("nan"),
                "adversary_suppression": float("nan"),
                "unsafe_rate": float("nan"),
                "trained_policy_source": "n/a",
                "trained_policy_checkpoint": "n/a",
            }
        )
    return recs


def build_paired_runs(
    by_sc: dict[tuple[str, str], float],
    run1_trained_offset: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Baselines are identical in both runs (shared RNG for frozen + adaptive)."""
    rng_b = np.random.default_rng(seed)
    rng_t1 = np.random.default_rng(seed + 1000)
    rng_t2 = np.random.default_rng(seed + 2000)
    cache_baseline_rewards: dict[tuple[str, str], list[float]] = {}
    for pol in ("baseline_frozen", "baseline_adaptive"):
        for sc_name in SCEN_ORDER:
            m = by_sc[(pol, sc_name)]
            cache_baseline_rewards[(pol, sc_name)] = _jitter5(m, rng_b)

    def run_trained_rewards(pol, offset: float, rng: np.random.Generator) -> list[dict]:
        out: list[dict] = []
        for sc_name in SCEN_ORDER:
            m = by_sc[(pol, sc_name)] + offset
            rewards = _jitter5(m, rng)
            out.extend(_records_for_rewards(pol, sc_name, rewards))
        return out

    r1: list[dict] = []
    r2: list[dict] = []
    for pol in ("baseline_frozen", "baseline_adaptive"):
        for sc_name in SCEN_ORDER:
            rlist = cache_baseline_rewards[(pol, sc_name)]
            r1.extend(_records_for_rewards(pol, sc_name, rlist))
            r2.extend(_records_for_rewards(pol, sc_name, rlist))
    r1.extend(
        run_trained_rewards("trained_grpo", run1_trained_offset, rng_t1)
    )
    r2.extend(run_trained_rewards("trained_grpo", 0.0, rng_t2))
    return r1, r2


def _mean_trained_from_fixtures(by_sc: dict[tuple[str, str], float]) -> float:
    s = 0.0
    n = 0
    for sc in SCEN_ORDER:
        s += by_sc[("trained_grpo", sc)]
        n += 1
    return s / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--by-scenario",
        type=Path,
        default=Path("newone/benchmark_by_scenario.csv"),
        help="Reference aggregate CSV (e.g. GPU-2 newone).",
    )
    p.add_argument(
        "--run1-trained-mean",
        type=float,
        default=-137.703282,
        help="Target overall trained mean for run 1 (first GPU report).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/benchmark_ingest")
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.by_scenario.exists():
        print(f"Missing {args.by_scenario}", file=sys.stderr)
        sys.exit(1)
    by_sc = _load_by_scenario(args.by_scenario)
    m2 = _mean_trained_from_fixtures(by_sc)
    offset = float(args.run1_trained_mean - m2)

    run1_records, run2_records = build_paired_runs(by_sc, offset, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = args.out_dir / "run1_episodes_from_fixture.csv"
    p2 = args.out_dir / "run2_episodes_from_fixture.csv"
    pd.DataFrame(run1_records).to_csv(p1, index=False)
    pd.DataFrame(run2_records).to_csv(p2, index=False)

    prov = {
        "source": "build_benchmark_fixtures.py",
        "reference_by_scenario_csv": str(args.by_scenario),
        "run2_trained_offset": 0.0,
        "run1_trained_offset": offset,
        "run2_merged_trained_mean_expected": m2,
        "run1_target_trained_mean": args.run1_trained_mean,
        "note": (
            "Per-episode values are small-jittered around scenario means. "
            "Replace with `benchmark_episodes.csv` from a real `run_final_benchmark` "
            "or with output of `ingest_benchmark_log.py` when you have a console log."
        ),
    }
    (args.out_dir / "fixtures_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] {p1} ({len(run1_records)} rows)")
    print(f"[ok] {p2} ({len(run2_records)} rows)")


if __name__ == "__main__":
    main()
