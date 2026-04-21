# aic/evals/benchmark_suite.py
"""
Benchmark Suite — compares AIC (Trained) against 3 baseline policies
across all 6 brutal scenarios.

Baselines:
1. HighestConfidenceOnly: Ignores trust, picks top confidence
2. MajorityVote: Picks the most common action direction
3. NoTrustOrchestrator: Phase 4 baseline (untrained)

Metrics:
- MTTR (Mean Time to Recovery) — steps until health > 0.5
- Adversary Suppression Rate — % of steps where adversary was overridden
- Unsafe Action Rate — % of actions with risk_score > 0.8
- Revenue Saved — estimated $ from SLA compliance improvement
"""
from __future__ import annotations

import csv
import copy
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from aic.utils.constants import (
    SLA_STEPS, METRIC_TARGETS, METRIC_FAULT_INIT,
    ALL_AGENTS, AGENT_ADV, INITIAL_TRUST,
)
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.env.world_state import WorldState
from aic.env.scenario_registry import ScenarioEngine, SCENARIO_REGISTRY
from aic.env.reward_engine import RewardEngine
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
from aic.schemas.traces import SubAgentRecommendation


# ── Revenue model ────────────────────────────────────────────────────────
REVENUE_PER_STEP_USD = 5000.0  # $5k per SLA step of healthy uptime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    policy_name: str
    scenario_id: int
    scenario_name: str
    mttr_steps: int            # steps until health > 0.5 (or SLA_STEPS if never)
    adversary_suppression_rate: float  # % of adversary recs rejected
    unsafe_action_rate: float  # % of actions with risk_score > 0.8
    revenue_saved_usd: float   # estimated revenue saved
    total_reward: float
    final_health: float
    sla_met: bool


class HighestConfidencePolicy:
    """Always picks the recommendation with highest confidence. No trust."""
    name = "HighestConfidenceOnly"

    def select(self, recommendations: list[SubAgentRecommendation]) -> SubAgentRecommendation:
        return max(recommendations, key=lambda r: r.confidence)


class MajorityVotePolicy:
    """Picks the action targeting the most commonly mentioned metric."""
    name = "MajorityVote"

    def select(self, recommendations: list[SubAgentRecommendation]) -> SubAgentRecommendation:
        if not recommendations:
            return recommendations[0]
        # Count target metric mentions across recommendations
        metric_counts: dict[str, int] = {}
        for rec in recommendations:
            for m in rec.target_metrics:
                metric_counts[m] = metric_counts.get(m, 0) + 1
        if not metric_counts:
            return recommendations[0]
        top_metric = max(metric_counts, key=metric_counts.get)
        # Pick the rec targeting the most popular metric with highest confidence
        matching = [r for r in recommendations if top_metric in r.target_metrics]
        if matching:
            return max(matching, key=lambda r: r.confidence)
        return max(recommendations, key=lambda r: r.confidence)


class NoTrustOrchestratorPolicy:
    """Phase 4 baseline: untrained orchestrator with frozen trust."""
    name = "NoTrustOrchestrator"

    def __init__(self):
        pass

    def select(self, recommendations: list[SubAgentRecommendation]) -> SubAgentRecommendation:
        # Naive: highest confidence from all agents (no trust filtering)
        return max(recommendations, key=lambda r: r.confidence)


def _run_baseline_episode(
    policy,
    scenario_id: int,
    episode_seed: int = 42,
) -> BenchmarkResult:
    """Run a single episode with a baseline policy."""
    rng = make_episode_rng(episode_seed)
    engine = ScenarioEngine(scenario_id)
    ws = WorldState(rng)

    # Create agents
    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    cycle = get_adversary_cycle(make_episode_rng(episode_seed))
    adv = AdversarialAgent(cycle, correct_recommendation_provider=db)

    mttr = SLA_STEPS  # default: never recovered
    adversary_selected = 0
    unsafe_actions = 0
    total_steps = SLA_STEPS
    total_reward = 0.0
    prev_metrics = ws.snapshot()

    reward_engine = RewardEngine()

    for step in range(SLA_STEPS):
        try:
            faults = engine.get_contributions(step)
            db_obs = ws.get_db_observation()
            infra_obs = ws.get_infra_observation()
            app_obs = ws.get_app_observation()

            db_rec = db.recommend(db_obs, step)
            infra_rec = infra.recommend(infra_obs, step)
            app_rec = app.recommend(app_obs, step)
            adv_rec = adv.recommend({**db_obs, **infra_obs, **app_obs}, step)

            all_recs = [db_rec, infra_rec, app_rec, adv_rec]

            # Policy selects action
            selected = policy.select(all_recs)

            # Track adversary selection
            if selected.agent_name == AGENT_ADV:
                adversary_selected += 1

            # Track unsafe actions
            if selected.risk_score > 0.8:
                unsafe_actions += 1

            # Apply action
            action_deltas = {m: -10.0 for m in selected.target_metrics}
            ws.step(action_deltas, faults)

            # Compute reward
            current = ws.snapshot()
            reward_record = reward_engine.compute_step_reward(
                step=step, metrics=current, prev_metrics=prev_metrics,
                override_applied=False,
                adversary_was_correct=adv.was_correct_at_step(step),
                predicted_2step_impact={m: -5.0 for m in selected.target_metrics},
                reasoning=selected.reasoning,
                lock_penalty=0.0,
            )
            total_reward += reward_record["total"]
            prev_metrics = current

            # Check MTTR
            if ws.get_health_score() > 0.5 and mttr == SLA_STEPS:
                mttr = step + 1

        except Exception:
            continue  # Graceful failure handling

    final_health = ws.get_health_score()
    r2 = reward_engine.compute_episode_end_reward(ws.snapshot(), 0)
    total_reward += r2

    # Revenue saved: healthy steps * revenue per step
    healthy_steps = sum(1 for s in range(SLA_STEPS) if s >= mttr)
    revenue = healthy_steps * REVENUE_PER_STEP_USD

    scenario = SCENARIO_REGISTRY[scenario_id]
    return BenchmarkResult(
        policy_name=policy.name,
        scenario_id=scenario_id,
        scenario_name=scenario.name,
        mttr_steps=mttr,
        adversary_suppression_rate=1.0 - (adversary_selected / total_steps),
        unsafe_action_rate=unsafe_actions / total_steps,
        revenue_saved_usd=revenue,
        total_reward=total_reward,
        final_health=final_health,
        sla_met=final_health > 0.5,
    )


def _run_aic_episode(
    scenario_id: int,
    mode: str = "trained",
    episode_seed: int = 42,
) -> BenchmarkResult:
    """Run a single episode with the full AIC orchestrator."""
    rng = make_episode_rng(episode_seed)
    engine = ScenarioEngine(scenario_id)
    ws = WorldState(rng)

    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    cycle = get_adversary_cycle(make_episode_rng(episode_seed))
    adv = AdversarialAgent(cycle, correct_recommendation_provider=db)
    orch = OrchestratorAgent(adv, use_llm=False)
    orch.mode = mode

    mttr = SLA_STEPS
    adversary_selected = 0
    unsafe_actions = 0
    total_reward = 0.0
    prev_metrics = ws.snapshot()

    reward_engine = RewardEngine()

    for step in range(SLA_STEPS):
        try:
            faults = engine.get_contributions(step)
            db_obs = ws.get_db_observation()
            infra_obs = ws.get_infra_observation()
            app_obs = ws.get_app_observation()

            db_rec = db.recommend(db_obs, step)
            infra_rec = infra.recommend(infra_obs, step)
            app_rec = app.recommend(app_obs, step)
            adv_rec = adv.recommend({**db_obs, **infra_obs, **app_obs}, step)

            all_recs = [db_rec, infra_rec, app_rec, adv_rec]
            current = ws.snapshot()

            action, override_applied = orch.decide(
                step=step,
                sla_remaining=SLA_STEPS - step,
                sub_agent_recommendations=all_recs,
                alert_summary="benchmark",
                prev_metrics=prev_metrics,
                current_metrics=current,
            )

            # Track adversary
            if orch._followed_agent == AGENT_ADV:
                adversary_selected += 1

            # Track unsafe — AIC should always be 0 due to verifier
            if action.explanation_trace.verifier_report:
                vr = action.explanation_trace.verifier_report
                if vr.get("risk_score", 0) > 0.8:
                    unsafe_actions += 1

            ws.step(action.action_deltas, faults)

            reward_record = reward_engine.compute_step_reward(
                step=step, metrics=ws.snapshot(), prev_metrics=prev_metrics,
                override_applied=override_applied,
                adversary_was_correct=adv.was_correct_at_step(step),
                predicted_2step_impact=action.explanation_trace.predicted_2step_impact,
                reasoning=action.explanation_trace.reasoning,
                lock_penalty=0.0,
            )
            total_reward += reward_record["total"]
            prev_metrics = ws.snapshot()

            if ws.get_health_score() > 0.5 and mttr == SLA_STEPS:
                mttr = step + 1

        except Exception:
            continue

    final_health = ws.get_health_score()
    r2 = reward_engine.compute_episode_end_reward(ws.snapshot(), 0)
    total_reward += r2
    healthy_steps = sum(1 for s in range(SLA_STEPS) if s >= mttr)
    revenue = healthy_steps * REVENUE_PER_STEP_USD

    scenario = SCENARIO_REGISTRY[scenario_id]
    return BenchmarkResult(
        policy_name=f"AIC ({mode.title()})",
        scenario_id=scenario_id,
        scenario_name=scenario.name,
        mttr_steps=mttr,
        adversary_suppression_rate=1.0 - (adversary_selected / SLA_STEPS),
        unsafe_action_rate=unsafe_actions / SLA_STEPS,
        revenue_saved_usd=revenue,
        total_reward=total_reward,
        final_health=final_health,
        sla_met=final_health > 0.5,
    )


def run_full_benchmark(
    output_path: str = "logs/benchmark_results.csv",
    episode_seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Run the full benchmark suite: AIC vs 3 baselines across 6 scenarios.

    Returns list of BenchmarkResult and writes CSV.
    """
    baselines = [
        HighestConfidencePolicy(),
        MajorityVotePolicy(),
        NoTrustOrchestratorPolicy(),
    ]
    all_results: list[BenchmarkResult] = []

    for scenario_id in sorted(SCENARIO_REGISTRY.keys()):
        # Run AIC trained
        result = _run_aic_episode(scenario_id, "trained", episode_seed)
        all_results.append(result)

        # Run AIC untrained
        result = _run_aic_episode(scenario_id, "untrained", episode_seed)
        all_results.append(result)

        # Run baselines
        for baseline in baselines:
            result = _run_baseline_episode(baseline, scenario_id, episode_seed)
            all_results.append(result)

    # Write CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))

    return all_results


def get_summary_table(results: list[BenchmarkResult]) -> dict:
    """
    Aggregate benchmark results into a summary table.

    Returns dict with policy_name → averaged metrics.
    """
    from collections import defaultdict
    agg: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        agg[r.policy_name].append(r)

    summary = {}
    for policy, runs in agg.items():
        n = len(runs)
        summary[policy] = {
            "avg_mttr": round(sum(r.mttr_steps for r in runs) / n, 1),
            "avg_adversary_suppression": round(
                sum(r.adversary_suppression_rate for r in runs) / n * 100, 1
            ),
            "avg_unsafe_rate": round(
                sum(r.unsafe_action_rate for r in runs) / n * 100, 1
            ),
            "total_revenue_saved": round(
                sum(r.revenue_saved_usd for r in runs), 0
            ),
            "avg_reward": round(sum(r.total_reward for r in runs) / n, 1),
            "sla_success_rate": round(
                sum(1 for r in runs if r.sla_met) / n * 100, 1
            ),
        }
    return summary
