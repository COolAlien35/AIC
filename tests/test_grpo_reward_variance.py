"""Regression test: GRPO reward function must produce non-zero variance
across a group of completions (otherwise GRPO advantage is zero and the
gradient vanishes, which is exactly what caused the original outage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aic.training.config import TrainingConfig  # noqa: E402
from aic.training.train_grpo import (  # noqa: E402
    _shape_reward,
    generate_grpo_prompt_dataset,
)


def test_shape_reward_distinguishes_garbage_from_partial():
    """The graded shaping should give different scores to qualitatively
    different completions (no JSON, partial JSON, full JSON-ish)."""
    completions = [
        "",  # empty
        "ok",  # short non-JSON
        "{ partial",  # has brace but no schema key
        '{"selected_recommendation_id": 0, "reasoning": "x"}',  # full
    ]
    scores = [_shape_reward(c) for c in completions]
    assert len(set(scores)) >= 3, f"Expected diverse shaping scores, got {scores}"
    assert scores[0] < scores[-1], (
        f"Empty string should score worse than valid JSON; got {scores}"
    )


def test_grpo_reward_func_has_variance(tmp_path):
    """End-to-end: invoke the GRPO reward_func with 4 completions and assert
    np.std(rewards) > 0. Locks in the regression that caused reward_std=0
    in the original outage."""
    config = TrainingConfig(
        sft_num_episodes=12,
        grpo_dataset_path=str(tmp_path / "prompts.jsonl"),
        grpo_output_dir=str(tmp_path / "grpo"),
    )
    dataset_path = generate_grpo_prompt_dataset(config)
    assert dataset_path.exists()

    import json

    with open(dataset_path) as f:
        first = json.loads(f.readline())

    # Replicate the inner reward_func without instantiating GRPOTrainer.
    from aic.env.aic_environment import AICEnvironment

    def _reward(completion: str) -> float:
        env = AICEnvironment(
            episode_id=int(first["episode_id"]),
            base_seed=int(first["base_seed"]),
            fault_mode=first["fault_mode"],
            use_llm_agents=False,
            manage_trust_scores=False,
            scenario_id=int(first["scenario_id"]),
        )
        env.reset()
        try:
            _obs, env_reward, _done, _info = env.step(completion)
        except Exception:
            env_reward = -8.0
        return float(env_reward) + _shape_reward(completion)

    completions = [
        "",  # empty
        "garbage no json",  # text
        '{"selected_recommendation_id": 0}',  # minimal JSON
        '{"selected_recommendation_id": 1, "reasoning": "fix db"}',  # different id
    ]
    rewards = np.array([_reward(c) for c in completions])
    assert rewards.std() > 0.0, f"Expected non-zero reward variance, got {rewards}"
