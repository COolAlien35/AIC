# aic/env/lock_manager.py
"""
Resource lock manager with deadlock detection.

Each service (db, infra, app) has one mutex.
Agents must acquire the lock for a service before executing remediation actions.
Deadlock: an agent waiting for ≥2 consecutive steps → forced release + penalty.

Design: LLM agents cannot block — they operate in a single-threaded step loop.
"Locking" is simulated: when an agent requests a lock already held, it is
recorded as "waiting". If it waits for DEADLOCK_TIMEOUT_STEPS, deadlock is declared.
"""
from typing import Optional

from aic.utils.constants import (
    SERVICES, DEADLOCK_TIMEOUT_STEPS, DEADLOCK_PENALTY, LOCK_HANDOFF_BONUS,
)


class ResourceLockManager:
    """
    Non-blocking lock manager for the AIC environment.

    acquire() returns immediately with True/False.
    Deadlock is detected via wait-step counting, not actual thread blocking.
    """

    def __init__(self):
        # Current holder of each service lock
        self._holders: dict[str, Optional[str]] = {s: None for s in SERVICES}
        # Agents currently waiting for a lock: {agent_name: service_name}
        self._waiting: dict[str, Optional[str]] = {}
        # Track consecutive waiting steps per agent
        self._wait_steps: dict[str, int] = {}
        # Accumulated penalties and bonuses this episode
        self._total_penalty: float = 0.0
        self._total_bonus: float = 0.0
        # History of lock events for logging
        self.event_log: list[dict] = []

    def reset(self) -> None:
        """Clear all lock state for a new episode."""
        self._holders = {s: None for s in SERVICES}
        self._waiting = {}
        self._wait_steps = {}
        self._total_penalty = 0.0
        self._total_bonus = 0.0
        self.event_log = []

    def request_lock(self, agent: str, service: str) -> bool:
        """
        Request lock for a service.
        Returns True if lock acquired, False if service is already locked.
        """
        if service not in SERVICES:
            raise ValueError(f"Unknown service: {service}. Valid: {SERVICES}")

        if self._holders[service] is None:
            # Lock is free — acquire it
            self._holders[service] = agent
            self._waiting.pop(agent, None)
            self._wait_steps.pop(agent, None)
            self.event_log.append({
                "event": "lock_acquired",
                "agent": agent,
                "service": service,
            })
            return True

        elif self._holders[service] == agent:
            # Agent already holds this lock — idempotent
            return True

        else:
            # Lock held by another agent — record waiting
            self._waiting[agent] = service
            self._wait_steps[agent] = self._wait_steps.get(agent, 0) + 1
            self.event_log.append({
                "event": "lock_waiting",
                "agent": agent,
                "service": service,
                "holder": self._holders[service],
                "wait_steps": self._wait_steps[agent],
            })
            return False

    def release_lock(self, agent: str, service: str) -> float:
        """
        Release a lock held by agent.
        Returns LOCK_HANDOFF_BONUS if another agent was waiting, else 0.0.
        """
        if self._holders.get(service) != agent:
            return 0.0  # Agent doesn't hold this lock — no-op

        self._holders[service] = None
        bonus = 0.0

        # Check if any agent was waiting for this lock
        waiting_agents = [a for a, s in self._waiting.items() if s == service]
        if waiting_agents:
            # Clean handoff to first waiting agent
            next_agent = waiting_agents[0]
            self._holders[service] = next_agent
            self._waiting.pop(next_agent)
            self._wait_steps.pop(next_agent, None)
            bonus = LOCK_HANDOFF_BONUS
            self._total_bonus += bonus
            self.event_log.append({
                "event": "lock_handoff",
                "from_agent": agent,
                "to_agent": next_agent,
                "service": service,
            })

        return bonus

    def detect_and_resolve_deadlocks(self) -> float:
        """
        Check for deadlock: any agent waiting ≥ DEADLOCK_TIMEOUT_STEPS.

        On deadlock: force-release the held lock, clear all waits, apply penalty.
        Returns total penalty from deadlocks this call (negative float or 0.0).
        """
        penalty = 0.0
        deadlocked_agents = [
            agent for agent, steps in self._wait_steps.items()
            if steps >= DEADLOCK_TIMEOUT_STEPS
        ]

        for agent in deadlocked_agents:
            # Find what service they were waiting for
            service = self._waiting.get(agent)
            if service is None:
                continue

            # Force-release the lock from the current holder
            current_holder = self._holders.get(service)
            if current_holder:
                self._holders[service] = None
                self.event_log.append({
                    "event": "deadlock_force_release",
                    "victim_agent": current_holder,
                    "waiting_agent": agent,
                    "service": service,
                })

            # Clear waiting state for this agent
            self._waiting.pop(agent, None)
            self._wait_steps.pop(agent, None)

            penalty += DEADLOCK_PENALTY
            self._total_penalty += DEADLOCK_PENALTY

        return penalty

    def get_status(self) -> dict:
        """Return a snapshot of the current lock state."""
        return {
            "holders": self._holders.copy(),
            "waiting": self._waiting.copy(),
            "wait_steps": self._wait_steps.copy(),
            "total_penalty": self._total_penalty,
            "total_bonus": self._total_bonus,
        }

    def is_locked_by(self, agent: str, service: str) -> bool:
        """Check if a specific agent holds a specific service lock."""
        return self._holders.get(service) == agent
