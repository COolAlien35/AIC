# aic/evals/leaderboard.py
"""
Leaderboard — loads Arena results and provides display-ready structures
for the dashboard and CLI output.

Provides:
  - load_leaderboard(path) → list[LeaderboardEntry]
  - format_leaderboard_table(entries) → str  (rich ASCII table)
  - get_aic_vs_best_baseline(entries) → dict  (advantage deltas)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_ARENA_PATH = "logs/arena_results.json"

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
POLICY_IS_AIC = {"AIC (Trained)", "AIC (Untrained)"}


@dataclass
class LeaderboardEntry:
    rank: int
    policy: str
    composite_score: float
    avg_mttr: float
    sla_success_rate: float          # percentage 0–100
    adversary_suppression_rate: float  # percentage 0–100
    unsafe_action_rate: float         # percentage 0–100
    total_revenue_saved_usd: float
    avg_final_health: float
    scenario_wins: int
    is_aic: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_aic = self.policy in POLICY_IS_AIC


def load_leaderboard(path: str = DEFAULT_ARENA_PATH) -> list[LeaderboardEntry]:
    """Load leaderboard entries from arena JSON output."""
    p = Path(path)
    if not p.exists():
        return []

    with open(p) as f:
        data = json.load(f)

    entries = []
    for item in data.get("leaderboard", []):
        entries.append(LeaderboardEntry(
            rank=item["rank"],
            policy=item["policy"],
            composite_score=item["composite_score"],
            avg_mttr=item["avg_mttr"],
            sla_success_rate=item["sla_success_rate"],
            adversary_suppression_rate=item["adversary_suppression_rate"],
            unsafe_action_rate=item["unsafe_action_rate"],
            total_revenue_saved_usd=item["total_revenue_saved_usd"],
            avg_final_health=item["avg_final_health"],
            scenario_wins=item["scenario_wins"],
        ))
    return sorted(entries, key=lambda e: e.rank)


def get_aic_vs_best_baseline(entries: list[LeaderboardEntry]) -> dict:
    """
    Return delta metrics: AIC Trained vs the best non-AIC policy.

    Useful for the "AIC is X% better" headline claim.
    """
    aic_entry = next((e for e in entries if e.policy == "AIC (Trained)"), None)
    baselines = [e for e in entries if not e.is_aic]

    if not aic_entry or not baselines:
        return {}

    best_baseline = min(baselines, key=lambda e: e.rank)

    return {
        "aic_policy": aic_entry.policy,
        "aic_rank": aic_entry.rank,
        "baseline_policy": best_baseline.policy,
        "baseline_rank": best_baseline.rank,
        "composite_advantage": round(aic_entry.composite_score - best_baseline.composite_score, 4),
        "mttr_improvement_steps": round(best_baseline.avg_mttr - aic_entry.avg_mttr, 1),
        "sla_improvement_pct": round(aic_entry.sla_success_rate - best_baseline.sla_success_rate, 1),
        "adv_suppression_delta_pct": round(
            aic_entry.adversary_suppression_rate - best_baseline.adversary_suppression_rate, 1
        ),
        "unsafe_rate_delta_pct": round(
            best_baseline.unsafe_action_rate - aic_entry.unsafe_action_rate, 1
        ),
        "revenue_delta_usd": round(
            aic_entry.total_revenue_saved_usd - best_baseline.total_revenue_saved_usd, 0
        ),
        "scenario_wins_aic": aic_entry.scenario_wins,
        "scenario_wins_best_baseline": best_baseline.scenario_wins,
    }


def get_scenario_results(path: str = DEFAULT_ARENA_PATH) -> list[dict]:
    """Load per-scenario per-policy results for radar/heatmap charts."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("results", [])


def format_leaderboard_table(entries: list[LeaderboardEntry]) -> str:
    """Return a pretty ASCII leaderboard table for CLI output."""
    header = (
        f"{'#':>2}  {'Medal':<6} {'Policy':<30} {'Score':>6} "
        f"{'MTTR':>6} {'SLA%':>5} {'AdvSup%':>8} {'Wins':>5}"
    )
    sep = "─" * len(header)
    lines = [sep, header, sep]

    for e in entries:
        medal = MEDALS.get(e.rank, "  ")
        tag = " ← AIC" if e.is_aic else ""
        lines.append(
            f"{e.rank:>2}  {medal:<6} {e.policy:<30} {e.composite_score:>6.3f} "
            f"{e.avg_mttr:>6.1f} {e.sla_success_rate:>5.1f} "
            f"{e.adversary_suppression_rate:>8.1f} {e.scenario_wins:>5}{tag}"
        )

    lines.append(sep)
    return "\n".join(lines)
