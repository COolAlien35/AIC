#!/usr/bin/env python3
# scripts/run_episode.py
"""
End-to-end episode runner for the Adaptive Incident Choreographer.
Uses rich for live-updating output showing metrics, actions, trust, and rewards.

Usage:
    python scripts/run_episode.py              # LLM mode (requires ANTHROPIC_API_KEY)
    python scripts/run_episode.py --no-llm     # Rule-based fallback only
    python scripts/run_episode.py --episode 5  # Run specific episode
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from aic.utils.constants import (
    SLA_STEPS, METRIC_TARGETS, METRIC_FAULT_INIT, ALL_AGENTS, AGENT_ADV,
)
from aic.utils.seeding import make_episode_rng, get_adversary_cycle, get_t_drift
from aic.utils.logging_utils import EpisodeLogger, StepRecord
from aic.env.world_state import WorldState
from aic.env.fault_injector import FaultInjector
from aic.env.schema_drift import SchemaDriftInjector
from aic.env.lock_manager import ResourceLockManager
from aic.env.reward_engine import RewardEngine
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent


console = Console()


def run_episode(episode_id: int = 0, use_llm: bool = True, fault_mode: str = "cascading_failure"):
    """Run a single 20-step episode with all agents."""

    console.print(Panel(
        f"[bold cyan]Adaptive Incident Choreographer[/bold cyan]\n"
        f"Episode {episode_id} | Fault: {fault_mode} | LLM: {'ON' if use_llm else 'OFF (rule-based)'}",
        box=box.DOUBLE,
    ))

    # ── Setup ───────────────────────────────────────────────────────────
    rng = make_episode_rng(episode_id)
    t_drift = get_t_drift(rng)
    adv_cycle = get_adversary_cycle(make_episode_rng(episode_id))

    ws = WorldState(make_episode_rng(episode_id))
    fi = FaultInjector(fault_mode)
    drift = SchemaDriftInjector(t_drift=t_drift, drift_type="field_rename")
    locks = ResourceLockManager()
    reward_engine = RewardEngine()
    logger = EpisodeLogger(log_dir="logs", episode_id=episode_id)

    # Create agents
    db_agent = DBAgent(use_llm=use_llm)
    infra_agent = InfraAgent(use_llm=use_llm)
    app_agent = AppAgent(use_llm=use_llm)
    adv_agent = AdversarialAgent(adv_cycle, correct_recommendation_provider=db_agent)
    orchestrator = OrchestratorAgent(adv_agent, use_llm=use_llm)

    console.print(f"[dim]Schema drift at step {t_drift} (field_rename on p95_latency_ms)[/dim]")
    console.print(f"[dim]Adversary correct steps: {sum(adv_cycle)}/20[/dim]\n")

    prev_metrics = ws.snapshot()
    total_reward = 0.0

    # ── Episode loop ────────────────────────────────────────────────────
    for step in range(SLA_STEPS):
        # 1. Get fault contributions
        faults = fi.get_contributions(step)

        # 2. Get observations (possibly drifted)
        db_obs = ws.get_db_observation()
        infra_obs = ws.get_infra_observation()
        app_obs_raw = ws.get_app_observation()
        app_obs = drift.inject(step, "app", app_obs_raw)

        # 3. Collect agent recommendations
        db_rec = db_agent.recommend(db_obs, step)
        infra_rec = infra_agent.recommend(infra_obs, step)
        app_rec = app_agent.recommend(app_obs, step)
        adv_rec = adv_agent.recommend({**db_obs, **infra_obs, **app_obs}, step)

        all_recs = [db_rec, infra_rec, app_rec, adv_rec]

        # 4. Orchestrator decides
        alert_summary = _build_alert_summary(ws.snapshot())
        action, override_applied = orchestrator.decide(
            step=step,
            sla_remaining=SLA_STEPS - step,
            sub_agent_recommendations=all_recs,
            alert_summary=alert_summary,
            prev_metrics=prev_metrics,
            current_metrics=ws.snapshot(),
        )

        # 5. Apply action to world state
        ws.step(action.action_deltas, faults)

        # 6. Lock management
        lock_penalty = locks.detect_and_resolve_deadlocks()

        # 7. Compute reward
        reward_record = reward_engine.compute_step_reward(
            step=step,
            metrics=ws.snapshot(),
            prev_metrics=prev_metrics,
            override_applied=override_applied,
            adversary_was_correct=adv_agent.was_correct_at_step(step),
            predicted_2step_impact=action.explanation_trace.predicted_2step_impact,
            reasoning=action.explanation_trace.reasoning,
            lock_penalty=lock_penalty,
        )
        total_reward += reward_record["total"]

        # 8. Log step
        record = StepRecord(
            episode_id=episode_id,
            step=step,
            timestamp=time.time(),
            world_state=ws.snapshot(),
            agent_recommendations={
                r.agent_name: r.action for r in all_recs
            },
            orchestrator_action=action.action_description,
            reward_components={
                "r1": reward_record["r1"],
                "r3": reward_record["r3"],
                "r4": reward_record["r4"],
            },
            reward_total=reward_record["total"],
            trust_scores=orchestrator.trust_scores.copy(),
            schema_drift_active=drift.was_active_at(step),
            schema_drift_type="field_rename" if drift.was_active_at(step) else None,
            deadlock_detected=lock_penalty < 0,
        )
        logger.log_step(record)

        # 9. Print step summary
        _print_step(step, ws, orchestrator, reward_record, action, override_applied, adv_cycle[step])

        prev_metrics = ws.snapshot()

    # ── Episode end ─────────────────────────────────────────────────────
    r2 = reward_engine.compute_episode_end_reward(ws.snapshot(), steps_remaining=0)
    total_reward += r2

    summary = logger.finalize(total_reward=total_reward, success=ws.is_within_sla())

    console.print()
    console.print(Panel(
        f"[bold]Episode {episode_id} Complete[/bold]\n"
        f"Total Reward: [{'green' if total_reward > 0 else 'red'}]{total_reward:.2f}[/]\n"
        f"R2 (SLA Bonus): {r2:.2f}\n"
        f"Final Health: {ws.get_health_score():.3f}\n"
        f"SLA Met: {'✅ Yes' if ws.is_within_sla() else '❌ No'}\n"
        f"Log: logs/episode_{episode_id:04d}.jsonl",
        box=box.DOUBLE,
        style="bold",
    ))

    return summary


def _build_alert_summary(metrics: dict[str, float]) -> str:
    """Build alert text from current metrics."""
    alerts = []
    for name, value in sorted(metrics.items()):
        target = METRIC_TARGETS.get(name, 0.0)
        if target == 0.0:
            if value > 0.5:
                alerts.append(f"{name}={value:.1f} (target={target:.1f})")
        else:
            pct_off = abs(value - target) / target * 100
            if pct_off > 10:
                alerts.append(f"{name}={value:.1f} ({pct_off:.0f}% off)")
    return "; ".join(alerts[:5]) if alerts else "All nominal"


def _print_step(step, ws, orchestrator, reward, action, override, adv_correct):
    """Print a compact step summary."""
    health = ws.get_health_score()
    health_color = "green" if health > 0.5 else "yellow" if health > 0.3 else "red"

    table = Table(
        title=f"Step {step:02d}/{SLA_STEPS}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        width=100,
    )
    table.add_column("Item", width=25)
    table.add_column("Value", width=70)

    table.add_row("Health", f"[{health_color}]{health:.3f}[/]")
    table.add_row("Action", action.action_description[:70])
    table.add_row("Target", action.target_service)
    table.add_row(
        "Reward",
        f"R1={reward['r1']:+.2f}  R3={reward['r3']:+.2f}  R4={reward['r4']:+.2f}  "
        f"Total={reward['total']:+.2f}",
    )
    table.add_row(
        "Trust",
        "  ".join(
            f"{a.split('_')[0]}={v:.2f}"
            for a, v in orchestrator.trust_scores.items()
        ),
    )
    table.add_row(
        "Adversary",
        f"{'✅ Correct' if adv_correct else '❌ Wrong'} | "
        f"Override: {'Yes' if override else 'No'}",
    )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC Episode Runner")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based fallbacks only")
    parser.add_argument("--fault", default="cascading_failure", help="Fault mode")
    args = parser.parse_args()

    run_episode(
        episode_id=args.episode,
        use_llm=not args.no_llm,
        fault_mode=args.fault,
    )
