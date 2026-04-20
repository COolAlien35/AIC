#!/usr/bin/env python3
# scripts/benchmark_untrained.py
"""
Benchmark script for the untrained (naive) orchestrator.

Runs episodes with trust scores FROZEN at INITIAL_TRUST (0.5) —
the orchestrator never learns to distrust the adversary.

Saves results to dashboard/assets/untrained_trajectories.pkl.

Usage:
    python scripts/benchmark_untrained.py
    python scripts/benchmark_untrained.py --episodes 20
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.training.config import TrainingConfig
from aic.training.train import run_episode
from aic.utils.constants import ALL_AGENTS, INITIAL_TRUST, SLA_STEPS
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent


class FrozenTrustOrchestrator(OrchestratorAgent):
    """
    Orchestrator that never updates trust scores.
    Trust stays frozen at INITIAL_TRUST (0.5) for all agents.
    This simulates an untrained agent that cannot learn to distrust.
    """

    def _update_trust_scores(self, step, prev_metrics, current_metrics):
        """Override: freeze trust at initial values."""
        self.trust_scores = {a: INITIAL_TRUST for a in ALL_AGENTS}


def benchmark_untrained(num_episodes: int = 20, seed: int = 42):
    """Run episodes with frozen trust and cache results."""
    config = TrainingConfig(
        num_episodes=num_episodes,
        base_seed=seed,
        use_llm_agents=False,
    )
    Path(config.trajectories_dir).mkdir(parents=True, exist_ok=True)

    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)

    # Frozen-trust orchestrator
    dummy_cycle = get_adversary_cycle(make_episode_rng(0, seed))
    adv = AdversarialAgent(dummy_cycle, db)
    orchestrator = FrozenTrustOrchestrator(adv, use_llm=False)

    all_results = {}
    rewards = []

    print(f"Benchmarking untrained agent: {num_episodes} episodes")
    print("Trust scores FROZEN at 0.5 (no learning)\n")

    for ep_id in range(num_episodes):
        result = run_episode(ep_id, config, orchestrator, db, infra, app)
        all_results[ep_id] = result
        rewards.append(result["total_reward"])

        print(
            f"Episode {ep_id:03d}: "
            f"reward={result['total_reward']:+.2f}, "
            f"health={result['final_health']:.3f}"
        )

    # Save
    out_path = Path(config.trajectories_dir) / "untrained_trajectories.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {len(all_results)} episodes to {out_path}")

    avg = sum(rewards) / len(rewards)
    print(f"Average reward: {avg:+.2f}")
    print(f"Min: {min(rewards):+.2f}, Max: {max(rewards):+.2f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC Untrained Benchmark")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark_untrained(args.episodes, args.seed)
