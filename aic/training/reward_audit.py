"""Reward audit loop for detecting and mitigating reward hacking during RL training.

Logs every (action, reward) pair, flags suspicious patterns, enforces timeouts,
and clamps rewards for flagged episodes before they enter the training batch.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditFlag:
    """A single suspicious-pattern flag raised by the audit loop."""
    flag_type: str
    step: int
    description: str
    severity: float  # 0.0–1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag_type": self.flag_type,
            "step": self.step,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class EpisodeAuditResult:
    """Audit result for a single episode."""
    episode_id: int
    flags: list[AuditFlag] = field(default_factory=list)
    total_steps: int = 0
    wall_clock_seconds: float = 0.0
    reward_clamped: bool = False
    original_reward: float = 0.0
    adjusted_reward: float = 0.0

    @property
    def is_flagged(self) -> bool:
        return len(self.flags) > 0

    @property
    def max_severity(self) -> float:
        if not self.flags:
            return 0.0
        return max(f.severity for f in self.flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "flags": [f.to_dict() for f in self.flags],
            "total_steps": self.total_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "reward_clamped": self.reward_clamped,
            "original_reward": self.original_reward,
            "adjusted_reward": self.adjusted_reward,
            "is_flagged": self.is_flagged,
            "max_severity": self.max_severity,
        }


class RewardAuditLoop:
    """Runtime audit loop for detecting reward-hacking patterns during training.

    Features:
    - Logs every (action, reward) pair to JSONL audit file
    - Flags repeated identical actions receiving high reward
    - Flags reward spikes with no meaningful state change
    - Enforces per-episode wall-clock and step-count timeouts
    - Post-episode: clamps rewards if audit flags exceed severity threshold

    Usage::

        audit = RewardAuditLoop(log_dir="logs/audit")

        audit.begin_episode(episode_id=0)
        for step in episode:
            action, reward, metrics = ...
            audit.record_step(step, action, reward, metrics)
            if audit.should_terminate_episode():
                break
        result = audit.end_episode(total_reward)
        adjusted_reward = result.adjusted_reward
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        max_wall_clock_seconds: float = 120.0,
        max_steps_per_episode: int = 50,
        repeat_action_threshold: int = 3,
        reward_spike_threshold: float = 20.0,
        state_change_epsilon: float = 0.01,
        severity_clamp_threshold: float = 0.5,
        reward_clamp_value: float = 0.0,
    ):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._max_wall_clock = max_wall_clock_seconds
        self._max_steps = max_steps_per_episode
        self._repeat_threshold = repeat_action_threshold
        self._reward_spike_threshold = reward_spike_threshold
        self._state_change_epsilon = state_change_epsilon
        self._severity_clamp_threshold = severity_clamp_threshold
        self._reward_clamp_value = reward_clamp_value

        # Per-episode state
        self._current_episode_id: int = -1
        self._episode_start_time: float = 0.0
        self._step_log: list[dict[str, Any]] = []
        self._action_counter: Counter = Counter()
        self._flags: list[AuditFlag] = []
        self._prev_metrics: dict[str, float] | None = None

        # Cross-episode statistics
        self._all_results: list[EpisodeAuditResult] = []

    def begin_episode(self, episode_id: int) -> None:
        """Start auditing a new episode."""
        self._current_episode_id = episode_id
        self._episode_start_time = time.time()
        self._step_log = []
        self._action_counter = Counter()
        self._flags = []
        self._prev_metrics = None

    def record_step(
        self,
        step: int,
        action: str | dict,
        reward: float,
        metrics: dict[str, float],
        info: dict[str, Any] | None = None,
    ) -> list[AuditFlag]:
        """Record one step and check for suspicious patterns.

        Returns any new flags raised this step.
        """
        action_key = json.dumps(action, sort_keys=True) if isinstance(action, dict) else str(action)

        record = {
            "episode_id": self._current_episode_id,
            "step": step,
            "action": action_key[:200],
            "reward": reward,
            "timestamp": time.time(),
        }
        self._step_log.append(record)

        # Write to audit JSONL
        audit_path = self._log_dir / f"audit_ep{self._current_episode_id:04d}.jsonl"
        with open(audit_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        new_flags: list[AuditFlag] = []

        # Check 1: Repeated identical actions getting high reward
        self._action_counter[action_key] += 1
        if (
            self._action_counter[action_key] >= self._repeat_threshold
            and reward > 0
        ):
            flag = AuditFlag(
                flag_type="repeated_action_high_reward",
                step=step,
                description=(
                    f"Action repeated {self._action_counter[action_key]}x with positive reward "
                    f"({reward:.2f}). Possible exploitation."
                ),
                severity=min(1.0, self._action_counter[action_key] / (self._repeat_threshold * 2)),
            )
            new_flags.append(flag)

        # Check 2: Reward spike with no meaningful state change
        if self._prev_metrics is not None:
            total_change = sum(
                abs(metrics.get(k, 0) - self._prev_metrics.get(k, 0))
                for k in set(metrics) | set(self._prev_metrics)
            )
            if reward > self._reward_spike_threshold and total_change < self._state_change_epsilon:
                flag = AuditFlag(
                    flag_type="reward_spike_no_state_change",
                    step=step,
                    description=(
                        f"Reward spike ({reward:.2f}) with negligible state change "
                        f"(delta={total_change:.4f}). Possible reward hacking."
                    ),
                    severity=0.8,
                )
                new_flags.append(flag)

        # Check 3: Wall-clock timeout approaching
        elapsed = time.time() - self._episode_start_time
        if elapsed > self._max_wall_clock * 0.9:
            flag = AuditFlag(
                flag_type="wall_clock_warning",
                step=step,
                description=f"Episode approaching wall-clock limit ({elapsed:.1f}s / {self._max_wall_clock}s).",
                severity=0.4,
            )
            new_flags.append(flag)

        self._flags.extend(new_flags)
        self._prev_metrics = metrics.copy()

        for f in new_flags:
            logger.warning("AuditFlag [ep=%d step=%d]: %s", self._current_episode_id, step, f.description)

        return new_flags

    def should_terminate_episode(self) -> bool:
        """Check if the episode should be forcibly terminated."""
        # Wall-clock timeout
        elapsed = time.time() - self._episode_start_time
        if elapsed > self._max_wall_clock:
            self._flags.append(AuditFlag(
                flag_type="wall_clock_timeout",
                step=len(self._step_log),
                description=f"Episode exceeded wall-clock limit ({elapsed:.1f}s > {self._max_wall_clock}s).",
                severity=1.0,
            ))
            return True

        # Step count timeout
        if len(self._step_log) >= self._max_steps:
            self._flags.append(AuditFlag(
                flag_type="step_count_timeout",
                step=len(self._step_log),
                description=f"Episode exceeded step limit ({len(self._step_log)} >= {self._max_steps}).",
                severity=0.7,
            ))
            return True

        return False

    def end_episode(self, total_reward: float) -> EpisodeAuditResult:
        """Finalize episode audit — clamp reward if flags are too severe."""
        elapsed = time.time() - self._episode_start_time

        result = EpisodeAuditResult(
            episode_id=self._current_episode_id,
            flags=self._flags.copy(),
            total_steps=len(self._step_log),
            wall_clock_seconds=elapsed,
            original_reward=total_reward,
        )

        # Clamp reward if severity exceeds threshold
        if result.max_severity >= self._severity_clamp_threshold:
            result.reward_clamped = True
            result.adjusted_reward = self._reward_clamp_value
            logger.warning(
                "Audit CLAMPED reward for ep=%d: %.2f → %.2f (severity=%.2f)",
                self._current_episode_id, total_reward,
                self._reward_clamp_value, result.max_severity,
            )
        else:
            result.adjusted_reward = total_reward

        self._all_results.append(result)

        # Write episode summary
        summary_path = self._log_dir / f"audit_ep{self._current_episode_id:04d}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def get_all_results(self) -> list[EpisodeAuditResult]:
        return self._all_results.copy()

    def get_flagged_episodes(self) -> list[EpisodeAuditResult]:
        return [r for r in self._all_results if r.is_flagged]

    def summary_stats(self) -> dict[str, Any]:
        """Cross-episode audit statistics."""
        if not self._all_results:
            return {"total_episodes": 0}

        flagged = self.get_flagged_episodes()
        clamped = [r for r in self._all_results if r.reward_clamped]
        flag_types = Counter()
        for r in self._all_results:
            for f in r.flags:
                flag_types[f.flag_type] += 1

        return {
            "total_episodes": len(self._all_results),
            "flagged_episodes": len(flagged),
            "clamped_episodes": len(clamped),
            "flag_type_counts": dict(flag_types),
            "avg_severity": (
                sum(r.max_severity for r in flagged) / len(flagged)
                if flagged else 0.0
            ),
        }
