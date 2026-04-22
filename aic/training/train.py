#!/usr/bin/env python3
# aic/training/train.py
"""
Baseline rollout and pseudo-training loop for the AIC orchestrator.

All baseline rollouts execute through `AICEnvironment`, making the OpenEnv
environment the single source of truth for state transitions, rewards, and
logging. The heuristic `OrchestratorAgent` remains the teacher/baseline policy
until GRPO training is enabled.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.network_agent import NetworkAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.agents.security_agent import SecurityAgent
from aic.env.aic_environment import AICEnvironment
from aic.schemas.traces import SubAgentRecommendation
from aic.training.config import TrainingConfig
from aic.utils.constants import SLA_STEPS
from aic.utils.seeding import get_adversary_cycle, make_episode_rng


CACHE_EPISODES = {0, 25, 50, 75, 99}


def _materialize_recommendations(obs: dict) -> list[SubAgentRecommendation]:
    return [
        SubAgentRecommendation.model_validate(rec)
        for rec in obs.get("sub_agent_recommendations", [])
    ]


def _select_candidate_id(
    obs: dict,
    action_description: str,
    followed_agent: str | None,
) -> int:
    candidates = obs.get("candidate_recommendations", [])
    for candidate in candidates:
        if (
            candidate.get("agent_name") == followed_agent
            and candidate.get("action") == action_description
        ):
            return int(candidate["recommendation_id"])
    for candidate in candidates:
        if candidate.get("action") == action_description:
            return int(candidate["recommendation_id"])
    for candidate in candidates:
        if candidate.get("agent_name") == "recovery_verifier":
            return int(candidate["recommendation_id"])
    return 0


def run_episode(
    episode_id: int,
    config: TrainingConfig,
    orchestrator: OrchestratorAgent,
    db: DBAgent,
    infra: InfraAgent,
    app: AppAgent,
    net: NetworkAgent = None,
    sec: SecurityAgent = None,
) -> dict:
    """Run a single episode via the environment and return trajectory + summary."""
    env = AICEnvironment(
        episode_id=episode_id,
        base_seed=config.base_seed,
        fault_mode=config.fault_mode,
        log_dir=config.log_dir,
        use_llm_agents=config.use_llm_agents,
        db_agent=db,
        infra_agent=infra,
        app_agent=app,
        net_agent=net,
        sec_agent=sec,
        include_network=net is not None,
        include_security=sec is not None,
        manage_trust_scores=False,
    )
    obs = env.reset()

    cycle = get_adversary_cycle(make_episode_rng(episode_id, config.base_seed))
    adv = AdversarialAgent(cycle, db)
    orchestrator.adversarial_agent = adv
    orchestrator.reset()

    trajectory = []
    trust_evolution = []
    prev_metrics = obs["current_metrics"]
    done = False

    while not done:
        recommendations = _materialize_recommendations(obs)
        action, override_applied = orchestrator.decide(
            step=obs["step"],
            sla_remaining=obs["sla_remaining_steps"],
            sub_agent_recommendations=recommendations,
            alert_summary=obs["alert_summary_text"],
            prev_metrics=prev_metrics,
            current_metrics=obs["current_metrics"],
        )

        candidate_id = _select_candidate_id(
            obs,
            action.action_description,
            getattr(orchestrator, "_followed_agent", None),
        )
        structured_action = {
            "selected_recommendation_id": candidate_id,
            "override_adversary": override_applied,
            "reasoning": action.explanation_trace.reasoning,
            "predicted_2step_impact": action.explanation_trace.predicted_2step_impact,
            "schema_drift_detected": action.explanation_trace.schema_drift_detected,
            "schema_drift_field": action.explanation_trace.schema_drift_field,
        }

        env.trust_scores = orchestrator.trust_scores.copy()
        next_obs, _reward, done, info = env.step(structured_action)

        trust_evolution.append({
            "step": obs["step"],
            **orchestrator.trust_scores.copy(),
        })
        trajectory.append({
            "step": obs["step"],
            "metrics": info["current_metrics"],
            "health": info["health"],
            "action": action.action_description,
            "override_applied": override_applied,
            "adv_was_correct": adv.was_correct_at_step(obs["step"]),
            "trust_scores": orchestrator.trust_scores.copy(),
            "reward": info["reward_record"],
            "trace": action.explanation_trace.model_dump(),
            "env_trace": info["trace"],
            "selected_candidate_id": candidate_id,
            "drift_active": info["schema_drift_active"],
        })

        prev_metrics = next_obs["current_metrics"]
        obs = next_obs

    total_reward = env.reward_engine.get_total_episode_reward()
    reward_history = env.reward_engine.get_reward_history()
    r2_bonus = getattr(env.reward_engine, "_r2_bonus", 0.0)

    scenario_name = "Unknown Scenario"
    for t in reversed(trajectory):
        hyp = t.get("trace", {}).get("root_cause_hypothesis")
        if hyp and isinstance(hyp, dict) and hyp.get("scenario_name"):
            scenario_name = hyp["scenario_name"]
            break

    mttr_steps = SLA_STEPS
    for t in trajectory:
        if t["health"] > 0.5:
            mttr_steps = t["step"] + 1
            break

    return {
        "episode_id": episode_id,
        "total_reward": total_reward,
        "r2_bonus": r2_bonus,
        "reward_history": reward_history,
        "trust_evolution": trust_evolution,
        "trajectory": trajectory,
        "final_health": env.world_state.get_health_score(),
        "scenario_name": scenario_name,
        "mttr": mttr_steps,
    }


def train(config: TrainingConfig = None) -> list[dict]:
    """Run heuristic-baseline episodes, cache trajectories, and write metrics."""
    if config is None:
        config = TrainingConfig()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.trajectories_dir).mkdir(parents=True, exist_ok=True)

    db = DBAgent(use_llm=config.use_llm_agents)
    infra_agent = InfraAgent(use_llm=config.use_llm_agents)
    app_agent = AppAgent(use_llm=config.use_llm_agents)
    net_agent = NetworkAgent(use_llm=config.use_llm_agents)
    sec_agent = SecurityAgent(use_llm=config.use_llm_agents)

    dummy_cycle = get_adversary_cycle(make_episode_rng(0, config.base_seed))
    adv_agent = AdversarialAgent(dummy_cycle, db)
    orchestrator = OrchestratorAgent(adv_agent, use_llm=config.use_llm_agents)
    orchestrator.mode = "trained"

    all_episode_results = []
    cached_trajectories = {}

    cache_set = {ep for ep in CACHE_EPISODES if ep < config.num_episodes}
    cache_set.add(0)
    cache_set.add(config.num_episodes - 1)

    print(f"Starting training: {config.num_episodes} episodes")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"Caching episodes: {sorted(cache_set)}")

    for episode_id in range(config.num_episodes):
        result = run_episode(
            episode_id,
            config,
            orchestrator,
            db,
            infra_agent,
            app_agent,
            net=net_agent,
            sec=sec_agent,
        )
        all_episode_results.append(result)

        print(
            f"Episode {episode_id:03d}: "
            f"reward={result['total_reward']:+.2f}, "
            f"health={result['final_health']:.3f}, "
            f"r2={result['r2_bonus']:.1f}"
        )

        if episode_id in cache_set:
            cached_trajectories[episode_id] = result
            print(f"  → Cached trajectory for episode {episode_id}")

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

    traj_path = Path(config.trajectories_dir) / "trained_trajectories.pkl"
    with open(traj_path, "wb") as f:
        pickle.dump(cached_trajectories, f)
    print(f"Trajectories saved to {traj_path}")

    reward_rows = []
    for r in all_episode_results:
        row = {
            "episode": r["episode_id"],
            "total_reward": r["total_reward"],
            "final_health": r["final_health"],
            "r2_bonus": r["r2_bonus"],
        }
        if r["reward_history"]:
            for key in ("r1", "r3", "r4", "r5", "r6"):
                row[f"avg_{key}"] = sum(s.get(key, 0.0) for s in r["reward_history"]) / len(r["reward_history"])
                row[f"sum_{key}"] = sum(s.get(key, 0.0) for s in r["reward_history"])
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