"""AIC tasks - structured, graded sub-problems on top of the AIC environment.

Three deterministic tasks with 0.0-1.0 graders, spanning easy / medium / hard
difficulty. Each task pins a specific scenario from
``aic.env.scenario_registry`` and exposes a pure ``grade(trace)`` function so
any policy can be scored by the same metric judges use.

Usage::

    from aic.tasks import TASKS, grade_episode
    score = grade_episode("db_pool_recovery", episode_trace)

The registry is the source of truth referenced from ``openenv.yaml``.
"""
from __future__ import annotations

from .registry import (
    Task,
    TASKS,
    DIFFICULTY_ORDER,
    grade_episode,
    get_task,
    get_task_for_scenario,
)
from . import task_db_pool_recovery  # noqa: F401  (registers task)
from . import task_canary_blackout  # noqa: F401
from . import task_adversarial_misroute  # noqa: F401

__all__ = [
    "Task",
    "TASKS",
    "DIFFICULTY_ORDER",
    "grade_episode",
    "get_task",
    "get_task_for_scenario",
]
