"""Curriculum scheduler for progressive difficulty training.

Defines task difficulty tiers (easy → medium → hard) and manages
automatic advancement based on rolling mean reward thresholds.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DifficultyTier(IntEnum):
    """Ordered difficulty levels for curriculum learning."""
    EASY = 0
    MEDIUM = 1
    HARD = 2


@dataclass
class TierConfig:
    """Configuration for a single difficulty tier."""
    tier: DifficultyTier
    fault_modes: list[str]
    sla_steps: int
    description: str
    include_network: bool = False
    include_security: bool = False
    schema_drift_enabled: bool = False
    adversary_accuracy: float = 0.5


# Default tier definitions — easy tasks use simple faults and generous SLA,
# hard tasks add more agents, schema drift, and tighter SLA.
DEFAULT_TIERS: dict[DifficultyTier, TierConfig] = {
    DifficultyTier.EASY: TierConfig(
        tier=DifficultyTier.EASY,
        fault_modes=["cascading_failure"],
        sla_steps=30,
        description="Single fault mode, generous SLA, no drift, basic agents only",
        include_network=False,
        include_security=False,
        schema_drift_enabled=False,
        adversary_accuracy=0.3,
    ),
    DifficultyTier.MEDIUM: TierConfig(
        tier=DifficultyTier.MEDIUM,
        fault_modes=["cascading_failure", "memory_leak", "db_connection_saturation"],
        sla_steps=20,
        description="Multiple fault modes, standard SLA, no drift, all agents",
        include_network=True,
        include_security=False,
        schema_drift_enabled=False,
        adversary_accuracy=0.5,
    ),
    DifficultyTier.HARD: TierConfig(
        tier=DifficultyTier.HARD,
        fault_modes=[
            "cascading_failure", "memory_leak",
            "db_connection_saturation", "network_storm",
        ],
        sla_steps=15,
        description="All fault modes, tight SLA, schema drift, all agents, tough adversary",
        include_network=True,
        include_security=True,
        schema_drift_enabled=True,
        adversary_accuracy=0.6,
    ),
}


class CurriculumScheduler:
    """Manages progressive difficulty advancement during training.

    Tracks rolling reward mean per tier and advances to the next tier
    when the agent consistently exceeds a configurable threshold.

    Usage::

        scheduler = CurriculumScheduler()
        for episode in range(num_episodes):
            tier_config = scheduler.current_tier_config()
            env_kwargs = scheduler.get_env_kwargs(episode)
            env = AICEnvironment(**env_kwargs)
            ...
            scheduler.record_episode(reward)
            scheduler.maybe_advance()
    """

    def __init__(
        self,
        tiers: dict[DifficultyTier, TierConfig] | None = None,
        advancement_threshold: float = -200.0,
        rolling_window: int = 10,
        min_episodes_per_tier: int = 5,
        log_path: str | None = None,
    ):
        self.tiers = tiers or DEFAULT_TIERS
        self._current_tier = DifficultyTier.EASY
        self._advancement_threshold = advancement_threshold
        self._rolling_window = rolling_window
        self._min_episodes_per_tier = min_episodes_per_tier

        self._reward_buffer: deque[float] = deque(maxlen=rolling_window)
        self._episodes_in_tier: int = 0
        self._history: list[dict[str, Any]] = []

        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def current_tier(self) -> DifficultyTier:
        return self._current_tier

    def current_tier_config(self) -> TierConfig:
        return self.tiers[self._current_tier]

    def rolling_mean_reward(self) -> float | None:
        if not self._reward_buffer:
            return None
        return sum(self._reward_buffer) / len(self._reward_buffer)

    def record_episode(self, reward: float, episode_id: int = -1) -> None:
        """Record an episode reward for the current tier."""
        self._reward_buffer.append(reward)
        self._episodes_in_tier += 1
        record = {
            "episode_id": episode_id,
            "tier": self._current_tier.name,
            "reward": reward,
            "rolling_mean": self.rolling_mean_reward(),
            "episodes_in_tier": self._episodes_in_tier,
        }
        self._history.append(record)
        logger.info(
            "Curriculum | tier=%s ep=%d reward=%.2f rolling_mean=%.2f",
            self._current_tier.name, episode_id, reward,
            self.rolling_mean_reward() or 0.0,
        )

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def maybe_advance(self) -> bool:
        """Check if the agent should advance to the next tier.

        Returns True if advancement occurred.
        """
        if self._current_tier == DifficultyTier.HARD:
            return False  # Already at max difficulty

        if self._episodes_in_tier < self._min_episodes_per_tier:
            return False  # Not enough data yet

        mean = self.rolling_mean_reward()
        if mean is None:
            return False

        if mean >= self._advancement_threshold:
            old_tier = self._current_tier
            self._current_tier = DifficultyTier(self._current_tier + 1)
            self._reward_buffer.clear()
            self._episodes_in_tier = 0
            logger.info(
                "Curriculum ADVANCE: %s → %s (rolling_mean=%.2f >= threshold=%.2f)",
                old_tier.name, self._current_tier.name,
                mean, self._advancement_threshold,
            )
            return True
        return False

    def get_env_kwargs(self, episode_id: int, base_seed: int = 42) -> dict[str, Any]:
        """Generate AICEnvironment constructor kwargs for the current tier."""
        config = self.current_tier_config()
        # Rotate through available fault modes for this tier
        fault_mode = config.fault_modes[episode_id % len(config.fault_modes)]
        return {
            "episode_id": episode_id,
            "base_seed": base_seed,
            "fault_mode": fault_mode,
            "include_network": config.include_network,
            "include_security": config.include_security,
            "manage_trust_scores": False,
            "use_llm_agents": False,
        }

    def get_reset_options(self) -> dict[str, Any]:
        """Generate env.reset() options for the current tier."""
        config = self.current_tier_config()
        options: dict[str, Any] = {}
        if not config.schema_drift_enabled:
            options["drift_type"] = None
        return options

    def get_history(self) -> list[dict[str, Any]]:
        return self._history.copy()

    def summary(self) -> str:
        """Human-readable summary of curriculum state."""
        config = self.current_tier_config()
        mean = self.rolling_mean_reward()
        return (
            f"Tier: {self._current_tier.name} | "
            f"Episodes: {self._episodes_in_tier} | "
            f"Rolling Mean: {mean:.2f if mean else 'N/A'} | "
            f"Threshold: {self._advancement_threshold:.2f} | "
            f"Desc: {config.description}"
        )
