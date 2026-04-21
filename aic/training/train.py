#!/usr/bin/env python3
# aic/training/train.py
"""
Training loop for the AIC orchestrator agent.

Runs episodic rollouts with the rule-based multi-agent system,
logging per-component rewards (R1/R2/R3/R4), caching trajectories
for the demo dashboard, and saving checkpoints periodically.

Usage:
    python -m aic.training.train
    python -m aic.training.train --num_episodes 5
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from aic.training.config import TrainingConfig
from aic.utils.seeding import make_episode_rng, get_t_drift, get_adversary_cycle
from aic.utils.constants import SLA_STEPS
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


DRIFT_TYPES = ["field_rename", "unit_shift", "silent_null"]
CACHE_EPISODES = {0, 25, 50, 75, 99}


def run_episode(
    episode_id: int,
    config: TrainingConfig,
    orchestrator: OrchestratorAgent,
    db: DBAgent,
    infra: InfraAgent,
    app: AppAgent,
) -> dict:
    """
    Run a single 20-step episode and return trajectory + reward summary.

    Returns dict with: episode_id, total_reward, r2_bonus,
    reward_history, trust_evolution, trajectory, final_health.
    """
    rng = make_episode_rng(episode_id, config.base_seed)
    cycle = get_adversary_cycle(make_episode_rng(episode_id, config.base_seed))
    t_drift = get_t_drift(make_episode_rng(episode_id, config.base_seed))
    drift_type = DRIFT_TYPES[episode_id % 3]

    ws = WorldState(rng)
    fi = FaultInjector(config.fault_mode)
    drift = SchemaDriftInjector(t_drift, drift_type)
    locks = ResourceLockManager()
    reward_eng = RewardEngine()

    adv = AdversarialAgent(cycle, db)
    orchestrator.adversarial_agent = adv
    orchestrator.reset()

    trajectory = []
    prev_metrics = ws.snapshot()
    trust_evolution = []

    early_termination = False
    r2 = 0.0

    for step in range(SLA_STEPS):
        # Get sliced observations (with possible drift injection)
        db_obs_raw = ws.get_db_observation()
        infra_obs_raw = ws.get_infra_observation()
        app_obs_raw = ws.get_app_observation()

        db_obs = drift.inject(step, "db", db_obs_raw)
        app_obs = drift.inject(step, "app", app_obs_raw)

        # Sub-agent recommendations
        recs = [
            db.recommend(db_obs, step),
            infra.recommend(infra_obs_raw, step),
            app.recommend(app_obs, step),
            adv.recommend(db_obs, step),
        ]

        # Alert summary
        health = ws.get_health_score()
        alert = (
            f"Step {step}: Health={health:.2f}, "
            f"SLA remaining={SLA_STEPS - step} steps. Critical metrics degraded."
        )

        # Orchestrator decision
        action, override_applied = orchestrator.decide(
            step=step,
            sla_remaining=SLA_STEPS - step,
            sub_agent_recommendations=recs,
            alert_summary=alert,
            prev_metrics=prev_metrics,
            current_metrics=ws.snapshot(),
        )

        # Apply action and fault to world state
        faults = fi.get_contributions(step)
        ws.step(action.action_deltas, faults)
        lock_penalty = locks.detect_and_resolve_deadlocks()

        # Compute reward
        adv_was_correct = adv.was_correct_at_step(step)
        r = reward_eng.compute_step_reward(
            step=step,
            metrics=ws.snapshot(),
            prev_metrics=prev_metrics,
            override_applied=override_applied,
            adversary_was_correct=adv_was_correct,
            predicted_2step_impact=action.explanation_trace.predicted_2step_impact,
            reasoning=action.explanation_trace.reasoning,
            lock_penalty=lock_penalty,
        )

        # Track trust evolution
        trust_evolution.append({
            "step": step,
            **orchestrator.trust_scores.copy(),
        })

        # Record trajectory step
        trajectory.append({
            "step": step,
            "metrics": ws.snapshot(),
            "health": ws.get_health_score(),
            "action": action.action_description,
            "override_applied": override_applied,
            "adv_was_correct": adv_was_correct,
            "trust_scores": orchestrator.trust_scores.copy(),
            "reward": r,
            "trace": action.explanation_trace.model_dump(),
            "drift_active": drift.was_active_at(step),
        })

        prev_metrics = ws.snapshot()

        # SLA early termination: check if all metrics are within SLA threshold
        if ws.is_within_sla():
            steps_remaining = SLA_STEPS - step - 1
            r2 = reward_eng.compute_episode_end_reward(ws.snapshot(), steps_remaining)
            early_termination = True
            break

    # Episode end reward (only if no early termination)
    if not early_termination:
        r2 = reward_eng.compute_episode_end_reward(ws.snapshot(), steps_remaining=0)

    total_reward = reward_eng.get_total_episode_reward()

    return {
        "episode_id": episode_id,
        "total_reward": total_reward,
        "r2_bonus": r2,
        "reward_history": reward_eng.get_reward_history(),
        "trust_evolution": trust_evolution,
        "trajectory": trajectory,
        "final_health": ws.get_health_score(),
    }


def train(config: TrainingConfig = None) -> list[dict]:
    """
    Main training loop. Runs num_episodes episodes, caches trajectories,
    saves checkpoints, and writes reward_curve.csv.
    """
    if config is None:
        config = TrainingConfig()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.trajectories_dir).mkdir(parents=True, exist_ok=True)

    # Initialize agents (rule-based for training speed)
    db = DBAgent(use_llm=config.use_llm_agents)
    infra_agent = InfraAgent(use_llm=config.use_llm_agents)
    app_agent = AppAgent(use_llm=config.use_llm_agents)

    # Orchestrator with trust updating
    dummy_cycle = get_adversary_cycle(make_episode_rng(0, config.base_seed))
    adv_agent = AdversarialAgent(dummy_cycle, db)
    orchestrator = OrchestratorAgent(adv_agent, use_llm=config.use_llm_agents)
    orchestrator.mode = "trained"

    all_episode_results = []
    cached_trajectories = {}

    # Determine which episodes to cache
    cache_set = set()
    for ep in CACHE_EPISODES:
        if ep < config.num_episodes:
            cache_set.add(ep)
    # Always cache first and last
    cache_set.add(0)
    cache_set.add(config.num_episodes - 1)

    print(f"Starting training: {config.num_episodes} episodes")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"Caching episodes: {sorted(cache_set)}")

    for episode_id in range(config.num_episodes):
        result = run_episode(
            episode_id, config, orchestrator, db, infra_agent, app_agent,
        )
        all_episode_results.append(result)

        print(
            f"Episode {episode_id:03d}: "
            f"reward={result['total_reward']:+.2f}, "
            f"health={result['final_health']:.3f}, "
            f"r2={result['r2_bonus']:.1f}"
        )

        # Cache specific episodes for demo
        if episode_id in cache_set:
            cached_trajectories[episode_id] = result
            print(f"  → Cached trajectory for episode {episode_id}")

        # Save checkpoint
        if (episode_id + 1) % config.checkpoint_interval == 0:
            checkpoint = {
                "episode_id": episode_id,
                "trust_scores": orchestrator.trust_scores,
                "episode_results_so_far": [
                    {
                        "episode_id": r["episode_id"],
                        "total_reward": r["total_reward"],
                    }
                    for r in all_episode_results
                ],
            }
            cp_path = Path(config.output_dir) / f"checkpoint_ep{episode_id:03d}.json"
            with open(cp_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  → Checkpoint saved to {cp_path}")

    # Save all cached trajectories for dashboard
    traj_path = Path(config.trajectories_dir) / "trained_trajectories.pkl"
    with open(traj_path, "wb") as f:
        pickle.dump(cached_trajectories, f)
    print(f"Trajectories saved to {traj_path}")

    # Save reward curve data with per-component breakdown
    reward_rows = []
    for r in all_episode_results:
        row = {
            "episode": r["episode_id"],
            "total_reward": r["total_reward"],
            "final_health": r["final_health"],
            "r2_bonus": r["r2_bonus"],
        }
        # Aggregate per-component averages from step rewards
        if r["reward_history"]:
            row["avg_r1"] = sum(s["r1"] for s in r["reward_history"]) / len(r["reward_history"])
            row["avg_r3"] = sum(s["r3"] for s in r["reward_history"]) / len(r["reward_history"])
            row["avg_r4"] = sum(s["r4"] for s in r["reward_history"]) / len(r["reward_history"])
            row["sum_r1"] = sum(s["r1"] for s in r["reward_history"])
            row["sum_r3"] = sum(s["r3"] for s in r["reward_history"])
            row["sum_r4"] = sum(s["r4"] for s in r["reward_history"])
        reward_rows.append(row)

    reward_curve = pd.DataFrame(reward_rows)
    csv_path = Path(config.log_dir) / "reward_curve.csv"
    reward_curve.to_csv(csv_path, index=False)
    print(f"Reward curve saved to {csv_path}")

    return all_episode_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC Training Loop")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--fault_mode", default="cascading_failure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        num_episodes=args.num_episodes,
        checkpoint_interval=args.checkpoint_interval,
        fault_mode=args.fault_mode,
        base_seed=args.seed,
    )
    train(config)
