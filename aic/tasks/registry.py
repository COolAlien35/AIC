"""Task registry + standardized episode-trace grading interface.

A "task" pins a specific scenario from ``aic.env.scenario_registry`` and a
pure grader function ``trace -> float`` returning a score in ``[0.0, 1.0]``.
This is the rubric-mandated 0.0-1.0 grader that judges use to compare
policies on equal footing - independent of the shaping reward used during
training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Trace = list of step dicts emitted by AICEnvironment / scripts/run_final_benchmark.py.
# Each step dict is expected to contain at least:
#   step (int), reward (float), info.current_metrics (dict), info.health, info.is_within_sla
EpisodeTrace = list[dict[str, Any]]
GraderFn = Callable[[EpisodeTrace], float]

DIFFICULTY_ORDER = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Task:
    """A graded task on top of the AIC environment."""

    task_id: str
    title: str
    difficulty: str          # one of DIFFICULTY_ORDER
    scenario_id: int         # references aic.env.scenario_registry.SCENARIO_REGISTRY
    description: str
    grader: GraderFn
    target_metrics: tuple[str, ...] = field(default_factory=tuple)
    success_threshold: float = 0.6  # score >= threshold counts as task_success


TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a task in the global registry. Idempotent for hot-reloads."""
    if task.difficulty not in DIFFICULTY_ORDER:
        raise ValueError(
            f"Task {task.task_id} has invalid difficulty {task.difficulty!r}; "
            f"must be one of {DIFFICULTY_ORDER}"
        )
    TASKS[task.task_id] = task
    return task


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id {task_id!r}; known: {sorted(TASKS.keys())}")
    return TASKS[task_id]


def get_task_for_scenario(scenario_id: int) -> Task | None:
    """Return the canonical task pinned to a given scenario_id, if any."""
    for t in TASKS.values():
        if t.scenario_id == scenario_id:
            return t
    return None


def grade_episode(task_id: str, trace: EpisodeTrace) -> float:
    """Run the task's grader against an episode trace; clipped to [0, 1]."""
    task = get_task(task_id)
    score = float(task.grader(trace))
    if score != score:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, score))


def _final_metrics(trace: EpisodeTrace) -> dict[str, float]:
    """Pull the last step's `current_metrics` (or empty dict)."""
    if not trace:
        return {}
    last = trace[-1]
    info = last.get("info") if isinstance(last.get("info"), dict) else last
    cm = info.get("current_metrics") if isinstance(info, dict) else {}
    if not isinstance(cm, dict):
        return {}
    return cm


def _initial_metrics(trace: EpisodeTrace) -> dict[str, float]:
    """Pull the first step's `current_metrics` (or empty dict)."""
    if not trace:
        return {}
    first = trace[0]
    info = first.get("info") if isinstance(first.get("info"), dict) else first
    cm = info.get("current_metrics") if isinstance(info, dict) else {}
    if not isinstance(cm, dict):
        return {}
    return cm


def _final_within_sla(trace: EpisodeTrace) -> bool:
    if not trace:
        return False
    last = trace[-1]
    info = last.get("info") if isinstance(last.get("info"), dict) else last
    val = info.get("is_within_sla") if isinstance(info, dict) else False
    return bool(val)


def _final_health(trace: EpisodeTrace) -> float:
    if not trace:
        return 0.0
    last = trace[-1]
    info = last.get("info") if isinstance(last.get("info"), dict) else last
    val = info.get("health") if isinstance(info, dict) else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _verifier_pass_rate(trace: EpisodeTrace) -> float:
    """Fraction of steps where the verifier approved the selected action."""
    approved = 0
    total = 0
    for step in trace:
        info = step.get("info") if isinstance(step.get("info"), dict) else step
        report = info.get("verifier_report") if isinstance(info, dict) else None
        if isinstance(report, dict):
            total += 1
            if bool(report.get("approved", False)):
                approved += 1
    return (approved / total) if total else 0.0


__all__ = [
    "EpisodeTrace",
    "GraderFn",
    "Task",
    "TASKS",
    "DIFFICULTY_ORDER",
    "register_task",
    "get_task",
    "get_task_for_scenario",
    "grade_episode",
    "_final_metrics",
    "_initial_metrics",
    "_final_within_sla",
    "_final_health",
    "_verifier_pass_rate",
]
