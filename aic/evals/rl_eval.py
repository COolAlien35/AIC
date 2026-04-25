"""Evaluation hooks for heuristic, SFT, and RL policies on AICEnvironment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from aic.env.aic_environment import AICEnvironment
from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.prompting import build_orchestrator_prompt, parse_decision


@dataclass
class RLEvalResult:
    episode_id: int
    total_reward: float
    final_health: float
    success: bool
    steps: int


def evaluate_policy_fn(
    policy_fn: Callable[[dict], dict | OrchestratorDecision | str],
    config: TrainingConfig | None = None,
    num_episodes: int = 5,
) -> list[RLEvalResult]:
    """Evaluate a callable policy against held-out AIC episodes."""
    if config is None:
        config = TrainingConfig()

    results: list[RLEvalResult] = []
    heldout_offset = 10_000
    for episode_id in range(num_episodes):
        env = AICEnvironment(
            episode_id=heldout_offset + episode_id,
            base_seed=config.base_seed,
            fault_mode=config.fault_mode,
            use_llm_agents=False,
            include_network=True,
            include_security=True,
            manage_trust_scores=False,
        )
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = policy_fn(obs)
            obs, _reward, done, _info = env.step(action)
            steps += 1
        results.append(
            RLEvalResult(
                episode_id=heldout_offset + episode_id,
                total_reward=env.reward_engine.get_total_episode_reward(),
                final_health=env.world_state.get_health_score(),
                success=env.world_state.is_within_sla(),
                steps=steps,
            )
        )
    return results


def build_model_policy(
    generate_fn: Callable[[str], str],
) -> Callable[[dict], dict | str]:
    """Wrap a text generation function into an environment policy callable."""

    def _policy(obs: dict) -> dict | str:
        prompt = build_orchestrator_prompt(obs)
        completion = generate_fn(prompt)
        try:
            return parse_decision(completion).model_dump()
        except Exception:
            return completion

    return _policy
