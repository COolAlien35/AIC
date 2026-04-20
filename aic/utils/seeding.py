# aic/utils/seeding.py
"""
Reproducible seed management for the AIC project.
Uses np.random.default_rng for isolated, episode-level RNG state.
"""
import random
from typing import Optional

import numpy as np


_global_seed: Optional[int] = None


def set_global_seed(seed: int) -> None:
    """Set seed globally for all random operations."""
    global _global_seed
    _global_seed = seed
    random.seed(seed)
    np.random.seed(seed)


def make_episode_rng(episode_id: int, base_seed: int = 42) -> np.random.Generator:
    """
    Return a seeded RNG for a specific episode.
    This ensures episode-level reproducibility independent of global state.
    Use this for: t_drift sampling, adversarial cycle selection, fault injection.
    """
    episode_seed = base_seed + episode_id * 1000
    return np.random.default_rng(episode_seed)


def get_t_drift(episode_rng: np.random.Generator, t_min: int = 8, t_max: int = 15) -> int:
    """Sample schema drift injection step. Fixed at episode start."""
    return int(episode_rng.integers(t_min, t_max + 1))


def get_adversary_cycle(episode_rng: np.random.Generator, n_steps: int = 20) -> list[bool]:
    """
    Generate a deterministic per-step correct/incorrect schedule for adversarial agent.
    Returns list of booleans: True = adversary is correct this step.
    Long-run accuracy = exactly 0.5 (n_steps//2 True values out of n_steps).
    """
    # Create a balanced schedule with exactly n_steps//2 True values
    schedule = [True] * (n_steps // 2) + [False] * (n_steps - n_steps // 2)
    # Use episode RNG for deterministic in-place shuffle
    episode_rng.shuffle(schedule)
    return schedule
