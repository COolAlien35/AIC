# aic/env/aic_environment.py
"""
AIC Gymnasium environment.

Wraps WorldState, FaultInjector, and EpisodeLogger into a standard
gymnasium.Env interface. Actions are natural-language strings (for LLM agents).
"""
import json
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aic.utils.constants import (
    SLA_STEPS, INITIAL_TRUST, ALL_AGENTS,
    TRACE_HISTORY_WINDOW, SERVICES,
)
from aic.utils.seeding import make_episode_rng
from aic.utils.logging_utils import EpisodeLogger, StepRecord
from aic.env.world_state import WorldState
from aic.env.fault_injector import FaultInjector


class AICEnvironment(gym.Env):
    """
    Adaptive Incident Choreographer environment.

    Observation: dict matching OrchestratorObservation fields
    Action: natural-language string (up to 2000 chars)
    Reward: float (stub — full reward engine in Phase 3)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        episode_id: int = 0,
        base_seed: int = 42,
        fault_mode: str = "cascading_failure",
        render_mode: Optional[str] = None,
        log_dir: str = "logs",
    ):
        super().__init__()

        self.episode_id = episode_id
        self.base_seed = base_seed
        self.fault_mode = fault_mode
        self.render_mode = render_mode
        self.log_dir = log_dir

        # Create episode-level RNG
        self._episode_rng = make_episode_rng(episode_id, base_seed)

        # Core components
        self.world_state = WorldState(self._episode_rng)
        self.fault_injector = FaultInjector(fault_mode)
        self.logger = EpisodeLogger(log_dir=log_dir, episode_id=episode_id)

        # Agent trust scores
        self.trust_scores: dict[str, float] = {
            agent: INITIAL_TRUST for agent in ALL_AGENTS
        }

        # Episode state
        self.step_count: int = 0
        self.done: bool = False
        self.trace_history: deque = deque(maxlen=TRACE_HISTORY_WINDOW)

        # Gymnasium spaces
        self.action_space = spaces.Text(max_length=2000)
        self.observation_space = spaces.Dict({
            "alert_summary_text": spaces.Text(max_length=5000),
            "sla_remaining_steps": spaces.Discrete(SLA_STEPS + 1),
            "step": spaces.Discrete(SLA_STEPS + 1),
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment for a new episode.

        Returns:
            (observation, info) tuple per Gymnasium API.
        """
        super().reset(seed=seed)

        # Re-create RNG if seed provided
        if seed is not None:
            self._episode_rng = make_episode_rng(self.episode_id, seed)
        else:
            self._episode_rng = make_episode_rng(self.episode_id, self.base_seed)

        self.world_state.reset(self._episode_rng)
        self.fault_injector = FaultInjector(self.fault_mode)
        self.logger = EpisodeLogger(log_dir=self.log_dir, episode_id=self.episode_id)

        self.trust_scores = {agent: INITIAL_TRUST for agent in ALL_AGENTS}
        self.step_count = 0
        self.done = False
        self.trace_history = deque(maxlen=TRACE_HISTORY_WINDOW)

        return self._get_orchestrator_obs(), {}

    def step(
        self, action: str
    ) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """
        Execute one step of the environment.

        Args:
            action: Natural-language action string from the orchestrator.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError(
                "Episode already done. Call reset() before stepping."
            )

        # Parse action into deltas (stub — full parsing in Phase 4)
        action_deltas = self._parse_action(action)

        # Get fault contributions for this step
        fault_contributions = self.fault_injector.get_contributions(self.step_count)

        # Evolve world state
        self.world_state.step(action_deltas, fault_contributions)

        # Advance step counter
        self.step_count += 1
        terminated = self.step_count >= SLA_STEPS
        self.done = terminated

        # Build observation
        obs = self._get_orchestrator_obs()

        # Reward stub (Phase 3 will implement full reward engine)
        reward = 0.0
        health = self.world_state.get_health_score()

        # Log step
        import time
        record = StepRecord(
            episode_id=self.episode_id,
            step=self.step_count,
            timestamp=time.time(),
            world_state=self.world_state.snapshot(),
            agent_recommendations={},
            orchestrator_action=action,
            reward_components={},
            reward_total=reward,
            trust_scores=self.trust_scores.copy(),
            schema_drift_active=False,
            schema_drift_type=None,
            deadlock_detected=False,
        )
        self.logger.log_step(record)

        info = {
            "step": self.step_count,
            "health": health,
            "is_within_sla": self.world_state.is_within_sla(),
        }

        if terminated:
            self.logger.finalize(
                total_reward=reward,
                success=self.world_state.is_within_sla(),
            )

        return obs, reward, terminated, False, info

    def _parse_action(self, action: str) -> dict[str, float]:
        """
        Parse an action string into metric deltas.
        Stub implementation — returns empty deltas.
        Full implementation in Phase 4 with LLM parsing.
        """
        return {}

    def _get_orchestrator_obs(self) -> dict:
        """
        Build the observation dict for the orchestrator agent.
        Matches OrchestratorObservation schema fields.
        """
        metrics = self.world_state.snapshot()

        # Build alert summary text
        alerts = []
        for name, value in sorted(metrics.items()):
            target = self.world_state.targets.get(name, 0.0)
            if target == 0.0:
                if value > 0.5:
                    alerts.append(f"ALERT: {name}={value:.1f} (target={target:.1f})")
            else:
                pct_off = abs(value - target) / target * 100
                if pct_off > 10:
                    alerts.append(
                        f"ALERT: {name}={value:.1f} "
                        f"({pct_off:.0f}% from target {target:.1f})"
                    )

        alert_text = "\n".join(alerts) if alerts else "All metrics nominal."

        return {
            "alert_summary_text": alert_text,
            "sla_remaining_steps": SLA_STEPS - self.step_count,
            "sub_agent_recommendations": [],
            "trace_history": list(self.trace_history),
            "current_trust_scores": self.trust_scores.copy(),
            "step": self.step_count,
        }

    def render(self) -> Optional[str]:
        """Render the current environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Build an ANSI text representation of the current state."""
        lines = [
            f"=== AIC Environment | Episode {self.episode_id} | "
            f"Step {self.step_count}/{SLA_STEPS} ===",
            f"Health: {self.world_state.get_health_score():.3f}  "
            f"SLA: {'OK' if self.world_state.is_within_sla() else 'BREACH'}",
            "",
            "Metrics:",
        ]
        for name in sorted(self.world_state.metrics.keys()):
            current = self.world_state.metrics[name]
            target = self.world_state.targets[name]
            lines.append(f"  {name:25s}  {current:10.2f}  (target: {target:.1f})")

        lines.append("")
        lines.append("Trust Scores:")
        for agent, score in self.trust_scores.items():
            lines.append(f"  {agent:25s}  {score:.3f}")

        return "\n".join(lines)
