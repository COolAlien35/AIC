# aic/evals/arena.py
"""
Benchmark Arena — extends benchmark_suite with 2 more baselines and
composite scoring for the Leaderboard.

Additional Baselines:
  RandomRecovery    — picks a random agent recommendation each step
  OraclePolicy      — cheats: knows actual outcome, picks best (upper bound)

Composite Score formula:  0.30 × (1 - normalized_mttr)
  0.25 × sla_success_rate
  0.20 × adversary_suppression_rate
  0.15 × (1 - unsafe_action_rate)
  0.10 × (revenue / MAX_REVENUE)
"""
from __future__ import annotations

import copy
import json
import random as stdlib_random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and booleans."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from aic.utils.constants import (
    SLA_STEPS, METRIC_TARGETS, ALL_AGENTS, AGENT_ADV, INITIAL_TRUST,
)
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.env.world_state import WorldState
from aic.env.scenario_registry import ScenarioEngine, SCENARIO_REGISTRY
from aic.env.reward_engine import RewardEngine
from aic.env.business_impact import compute_business_impact
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.evals.benchmark_suite import (
    BenchmarkResult,
    HighestConfidencePolicy,
    MajorityVotePolicy,
    NoTrustOrchestratorPolicy,
    _run_baseline_episode,
    _run_aic_episode,
    REVENUE_PER_STEP_USD,
)
from aic.schemas.traces import SubAgentRecommendation


# ── Maximum theoretical revenue per episode (used for normalisation) ─────
MAX_REVENUE = SLA_STEPS * REVENUE_PER_STEP_USD


# ─────────────────────────────────────────────────────────────────────────
# Extra baseline policies
# ─────────────────────────────────────────────────────────────────────────

class RandomRecoveryPolicy:
    """Picks a completely random recommendation each step."""
    name = "RandomRecovery"

    def __init__(self, seed: int = 99):
        self._rng = stdlib_random.Random(seed)

    def select(self, recommendations: list[SubAgentRecommendation]) -> SubAgentRecommendation:
        return self._rng.choice(recommendations)


class OraclePolicy:
    """
    Upper-bound oracle — cheats by checking which agent's predicted
    expected_impact would result in the best health score.

    Not achievable in practice; sets the theoretical ceiling.
    """
    name = "Oracle (Upper Bound)"

    def select(self, recommendations: list[SubAgentRecommendation]) -> SubAgentRecommendation:
        # Score by sum of negative expected_impact values (lower = more improvement)
        def oracle_score(r: SubAgentRecommendation) -> float:
            if not r.expected_impact:
                return float("inf")
            # Sum of absolute improvements expected
            return -sum(abs(v) for v in r.expected_impact.values())

        return min(recommendations, key=oracle_score)


# ─────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────

def compute_composite_score(
    results: list[BenchmarkResult],
    max_revenue: float = MAX_REVENUE,
) -> float:
    """
    Compute a composite score across all scenarios for a policy.

    Score ∈ [0, 1], higher = better.

    Weights:
      30% — MTTR (lower steps = better)
      25% — SLA success rate
      20% — Adversary suppression rate
      15% — (1 - unsafe action rate)
      10% — Revenue saved (normalised)
    """
    if not results:
        return 0.0

    n = len(results)
    avg_mttr = sum(r.mttr_steps for r in results) / n
    norm_mttr = 1.0 - (avg_mttr / SLA_STEPS)                            # higher = faster
    sla_rate = sum(1 for r in results if r.sla_met) / n
    adv_sup = sum(r.adversary_suppression_rate for r in results) / n
    safe_rate = 1.0 - sum(r.unsafe_action_rate for r in results) / n
    rev_norm = min(1.0, sum(r.revenue_saved_usd for r in results) / (n * max_revenue))

    return round(
        0.30 * norm_mttr
        + 0.25 * sla_rate
        + 0.20 * adv_sup
        + 0.15 * safe_rate
        + 0.10 * rev_norm,
        4,
    )


# ─────────────────────────────────────────────────────────────────────────
# Full Arena run
# ─────────────────────────────────────────────────────────────────────────

ALL_BASELINES = [
    HighestConfidencePolicy(),
    MajorityVotePolicy(),
    NoTrustOrchestratorPolicy(),
    RandomRecoveryPolicy(),
    OraclePolicy(),
]

POLICY_ORDER = [
    "AIC (Trained)",
    "AIC (Untrained)",
    "Oracle (Upper Bound)",
    "HighestConfidenceOnly",
    "MajorityVote",
    "NoTrustOrchestrator",
    "RandomRecovery",
]


def run_arena(
    output_path: str = "logs/arena_results.json",
    episode_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the full Arena: AIC (trained + untrained) vs 5 baselines
    across all 6 brutal scenarios.

    Returns a dict with:
      - "results": list of BenchmarkResult dicts
      - "leaderboard": ranked list of {policy, composite_score, metrics}
      - "scenario_wins": {policy: number of scenarios where policy ranked #1}
    """
    all_results: dict[str, list[BenchmarkResult]] = {p: [] for p in POLICY_ORDER}

    scenario_ids = sorted(SCENARIO_REGISTRY.keys())

    for scenario_id in scenario_ids:
        scenario_name = SCENARIO_REGISTRY[scenario_id].name
        if verbose:
            print(f"  ▶ Scenario {scenario_id}: {scenario_name}")

        # AIC policies
        for mode in ("trained", "untrained"):
            key = f"AIC ({mode.title()})"
            r = _run_aic_episode(scenario_id, mode, episode_seed)
            all_results[key].append(r)

        # Baseline policies
        for baseline in ALL_BASELINES:
            r = _run_baseline_episode(baseline, scenario_id, episode_seed)
            all_results[baseline.name].append(r)

    # Compute composite scores
    composites = {
        policy: compute_composite_score(runs)
        for policy, runs in all_results.items()
    }

    # Rank
    ranked = sorted(composites.items(), key=lambda x: x[1], reverse=True)

    # Scenario wins (which policy was best on each scenario?)
    scenario_wins: dict[str, int] = {p: 0 for p in POLICY_ORDER}
    for i, scenario_id in enumerate(scenario_ids):
        step_results = {
            policy: runs[i]
            for policy, runs in all_results.items()
            if i < len(runs)
        }
        # Winner = highest sla_met + lowest mttr
        winner = max(
            step_results.items(),
            key=lambda x: (x[1].sla_met, -x[1].mttr_steps, x[1].final_health),
        )[0]
        scenario_wins[winner] = scenario_wins.get(winner, 0) + 1

    # Build leaderboard entries
    leaderboard = []
    for rank, (policy, score) in enumerate(ranked, 1):
        runs = all_results[policy]
        n = len(runs)
        entry = {
            "rank": rank,
            "policy": policy,
            "composite_score": score,
            "avg_mttr": round(sum(r.mttr_steps for r in runs) / n, 1) if n else SLA_STEPS,
            "sla_success_rate": round(sum(1 for r in runs if r.sla_met) / n * 100, 1) if n else 0,
            "adversary_suppression_rate": round(
                sum(r.adversary_suppression_rate for r in runs) / n * 100, 1
            ) if n else 0,
            "unsafe_action_rate": round(
                sum(r.unsafe_action_rate for r in runs) / n * 100, 1
            ) if n else 0,
            "total_revenue_saved_usd": round(sum(r.revenue_saved_usd for r in runs), 0) if runs else 0,
            "avg_final_health": round(sum(r.final_health for r in runs) / n, 3) if n else 0,
            "scenario_wins": scenario_wins.get(policy, 0),
        }
        leaderboard.append(entry)

    # Flatten results for JSON serialisation
    flat_results = []
    for policy, runs in all_results.items():
        for r in runs:
            d = asdict(r)
            flat_results.append(d)

    output = {
        "results": flat_results,
        "leaderboard": leaderboard,
        "scenario_wins": scenario_wins,
        "composite_scores": {p: s for p, s in ranked},
    }

    # Write JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpySafeEncoder)

    if verbose:
        print("\n🏆 ARENA LEADERBOARD:")
        for entry in leaderboard:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry["rank"], "  ")
            print(
                f"  {medal} #{entry['rank']} {entry['policy']:<30} "
                f"score={entry['composite_score']:.3f}  "
                f"MTTR={entry['avg_mttr']:.1f}  "
                f"SLA={entry['sla_success_rate']}%  "
                f"wins={entry['scenario_wins']}/6"
            )

    return output
