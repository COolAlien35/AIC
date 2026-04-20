#!/usr/bin/env python3
# scripts/pre_cache_demo.py
"""
Master pre-caching script. Runs both trained (trust-updating) and
untrained (frozen-trust) agents, saves all .pkl files needed by
the dashboard, and prints a comparison table.

Usage:
    python scripts/pre_cache_demo.py
    python scripts/pre_cache_demo.py --episodes 10
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from aic.training.config import TrainingConfig
from aic.training.train import train
from scripts.benchmark_untrained import benchmark_untrained

console = Console()


def pre_cache(num_episodes: int = 10, seed: int = 42):
    """Run trained + untrained, save all demo data."""

    assets_dir = Path("dashboard/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # ── Trained (trust-updating) ────────────────────────────────────────
    console.print("\n[bold cyan]═══ Running TRAINED agent ═══[/bold cyan]\n")
    config = TrainingConfig(
        num_episodes=num_episodes,
        checkpoint_interval=max(5, num_episodes // 4),
        base_seed=seed,
    )
    trained_results = train(config)

    # ── Untrained (frozen trust) ────────────────────────────────────────
    console.print("\n[bold yellow]═══ Running UNTRAINED agent ═══[/bold yellow]\n")
    untrained_results = benchmark_untrained(num_episodes, seed)

    # ── Comparison table ────────────────────────────────────────────────
    trained_rewards = [r["total_reward"] for r in trained_results]
    untrained_rewards = [
        untrained_results[ep]["total_reward"]
        for ep in sorted(untrained_results.keys())
    ]

    # Save comparison CSV
    comparison_rows = []
    for i in range(num_episodes):
        comparison_rows.append({
            "episode": i,
            "trained_reward": trained_rewards[i],
            "untrained_reward": untrained_rewards[i],
            "delta": trained_rewards[i] - untrained_rewards[i],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = assets_dir / "reward_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    # Print rich comparison table
    table = Table(
        title="Trained vs Untrained Comparison",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Episode", justify="center", width=10)
    table.add_column("Trained", justify="right", width=15)
    table.add_column("Untrained", justify="right", width=15)
    table.add_column("Delta", justify="right", width=12)

    for row in comparison_rows:
        delta = row["delta"]
        delta_color = "green" if delta > 0 else "red"
        delta_str = f"{delta:+.2f}"
        table.add_row(
            str(row["episode"]),
            f"{row['trained_reward']:+.2f}",
            f"{row['untrained_reward']:+.2f}",
            f"[{delta_color}]{delta_str}[/{delta_color}]",
        )

    # Summary row
    avg_trained = sum(trained_rewards) / len(trained_rewards)
    avg_untrained = sum(untrained_rewards) / len(untrained_rewards)
    avg_delta = avg_trained - avg_untrained
    delta_color = "green" if avg_delta > 0 else "red"
    avg_d_str = f"{avg_delta:+.2f}"
    avg_t_str = f"{avg_trained:+.2f}"
    avg_u_str = f"{avg_untrained:+.2f}"
    table.add_section()
    table.add_row(
        "[bold]AVG[/bold]",
        f"[bold]{avg_t_str}[/bold]",
        f"[bold]{avg_u_str}[/bold]",
        f"[bold {delta_color}]{avg_d_str}[/bold {delta_color}]",
    )

    console.print()
    console.print(table)

    # Summary
    console.print(f"\n[bold]Files saved:[/bold]")
    console.print(f"  ✅ dashboard/assets/trained_trajectories.pkl")
    console.print(f"  ✅ dashboard/assets/untrained_trajectories.pkl")
    console.print(f"  ✅ {csv_path}")
    console.print(f"  ✅ logs/reward_curve.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC Pre-Cache Demo Data")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pre_cache(args.episodes, args.seed)
