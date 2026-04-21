#!/usr/bin/env python3
# scripts/run_final_benchmark.py
"""
Run the full AIC benchmark suite: AIC vs 3 baselines across 6 scenarios.

Usage:
    python scripts/run_final_benchmark.py
    python scripts/run_final_benchmark.py --output logs/benchmark_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from rich.console import Console
from rich.table import Table
from rich import box

from aic.evals.benchmark_suite import run_full_benchmark, get_summary_table


console = Console()


def main():
    parser = argparse.ArgumentParser(description="AIC Benchmark Suite")
    parser.add_argument(
        "--output", default="logs/benchmark_results.csv",
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    console.print("\n[bold cyan]═══ AIC Benchmark Suite ═══[/bold cyan]\n")
    console.print("Running AIC vs 3 baselines across 6 brutal scenarios...\n")

    results = run_full_benchmark(args.output, args.seed)
    summary = get_summary_table(results)

    # Print detailed results
    table = Table(
        title="📊 Benchmark Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Policy", style="bold")
    table.add_column("Avg MTTR", justify="right")
    table.add_column("Adv Suppression", justify="right")
    table.add_column("Unsafe Rate", justify="right")
    table.add_column("Revenue Saved", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("SLA Success", justify="right")

    for policy, metrics in summary.items():
        is_aic = "AIC (Trained)" in policy
        style = "bold green" if is_aic else ""
        table.add_row(
            policy,
            f"{metrics['avg_mttr']:.1f} steps",
            f"{metrics['avg_adversary_suppression']:.1f}%",
            f"{metrics['avg_unsafe_rate']:.1f}%",
            f"${metrics['total_revenue_saved']:,.0f}",
            f"{metrics['avg_reward']:+.1f}",
            f"{metrics['sla_success_rate']:.0f}%",
            style=style,
        )

    console.print(table)
    console.print(f"\n[dim]Results saved to: {args.output}[/dim]\n")

    # Print per-scenario breakdown for AIC
    console.print("[bold]Per-Scenario Breakdown (AIC Trained):[/bold]")
    aic_results = [r for r in results if r.policy_name == "AIC (Trained)"]
    for r in aic_results:
        emoji = "✅" if r.sla_met else "❌"
        console.print(
            f"  {emoji} {r.scenario_name}: MTTR={r.mttr_steps} steps, "
            f"Reward={r.total_reward:+.1f}, Unsafe={r.unsafe_action_rate:.0%}"
        )

    console.print()


if __name__ == "__main__":
    main()
