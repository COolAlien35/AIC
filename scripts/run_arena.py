#!/usr/bin/env python3
# scripts/run_arena.py
"""
CLI runner for the AIC Benchmark Arena.

Usage:
    python scripts/run_arena.py
    python scripts/run_arena.py --seed 123 --output logs/custom_arena.json
    python scripts/run_arena.py --quiet
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.evals.arena import run_arena
from aic.evals.leaderboard import load_leaderboard, format_leaderboard_table, get_aic_vs_best_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AIC Benchmark Arena — AIC vs 5 baselines across 6 scenarios."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Episode RNG seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="logs/arena_results.json",
        help="Output JSON path (default: logs/arena_results.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-scenario output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🏟️  AIC BENCHMARK ARENA")
    print("=" * 60)
    print(f"  Seed:   {args.seed}")
    print(f"  Output: {args.output}")
    print(f"  Policies: AIC (Trained), AIC (Untrained), + 5 baselines")
    print(f"  Scenarios: 6 brutal scenarios")
    print()

    start = time.time()
    run_arena(output_path=args.output, episode_seed=args.seed, verbose=not args.quiet)
    elapsed = time.time() - start

    print(f"\n⏱  Arena completed in {elapsed:.1f}s")
    print(f"📄 Results saved → {args.output}")

    # Load and display leaderboard
    entries = load_leaderboard(args.output)
    if entries:
        print("\n" + format_leaderboard_table(entries))

        adv = get_aic_vs_best_baseline(entries)
        if adv:
            print("\n🚀 AIC ADVANTAGE SUMMARY:")
            print(f"  vs {adv['baseline_policy']}:")
            print(f"  ├─ Composite score  +{adv['composite_advantage']:.3f}")
            print(f"  ├─ MTTR improvement  {adv['mttr_improvement_steps']:.1f} steps faster")
            print(f"  ├─ SLA success      +{adv['sla_improvement_pct']:.1f}%")
            print(f"  ├─ Adv. suppression +{adv['adv_suppression_delta_pct']:.1f}%")
            print(f"  ├─ Unsafe actions   -{adv['unsafe_rate_delta_pct']:.1f}%")
            print(f"  ├─ Revenue saved    +${adv['revenue_delta_usd']:,.0f}")
            print(f"  └─ Scenario wins     {adv['scenario_wins_aic']}/6 vs {adv['scenario_wins_best_baseline']}/6")


if __name__ == "__main__":
    main()
