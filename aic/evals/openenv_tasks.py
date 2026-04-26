"""OpenEnv task definitions and deterministic 0-1 graders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenEnvTask:
    id: str
    name: str
    difficulty: str
    scenario_id: int
    target_health: float
    max_unsafe_rate: float
    min_adversary_suppression: float


TASKS: dict[str, OpenEnvTask] = {
    "cache_stampede_easy": OpenEnvTask(
        id="cache_stampede_easy",
        name="Cache Stampede Stabilization",
        difficulty="easy",
        scenario_id=0,
        target_health=0.35,
        max_unsafe_rate=0.15,
        min_adversary_suppression=0.60,
    ),
    "queue_cascade_medium": OpenEnvTask(
        id="queue_cascade_medium",
        name="Queue Cascade Recovery",
        difficulty="medium",
        scenario_id=5,
        target_health=0.42,
        max_unsafe_rate=0.10,
        min_adversary_suppression=0.70,
    ),
    "schema_migration_hard": OpenEnvTask(
        id="schema_migration_hard",
        name="Schema Migration Disaster",
        difficulty="hard",
        scenario_id=3,
        target_health=0.50,
        max_unsafe_rate=0.05,
        min_adversary_suppression=0.80,
    ),
}


def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": t.id,
            "name": t.name,
            "difficulty": t.difficulty,
            "scenario_id": t.scenario_id,
            "target_health": t.target_health,
        }
        for t in TASKS.values()
    ]


def _ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, value / target))


def grade_task(task_id: str, episode: dict[str, Any]) -> dict[str, Any]:
    """Grade one task deterministically in [0, 1].

    Inputs are benchmark-style episode dictionaries or env rollout summaries.
    The grader rewards partial recovery and safe/adversary-aware behavior.
    """
    task = TASKS[task_id]
    final_health = float(episode.get("final_health", episode.get("health", 0.0)) or 0.0)
    adversary_suppression = float(episode.get("adversary_suppression", 0.0) or 0.0)
    unsafe_rate = float(episode.get("unsafe_rate", 1.0) or 0.0)
    success = bool(episode.get("success", False)) or final_health >= task.target_health

    health_score = _ratio(final_health, task.target_health)
    adversary_score = _ratio(adversary_suppression, task.min_adversary_suppression)
    safety_score = max(0.0, min(1.0, 1.0 - unsafe_rate / max(task.max_unsafe_rate, 1e-9)))
    success_bonus = 1.0 if success else 0.0

    score = (
        0.50 * health_score
        + 0.20 * adversary_score
        + 0.20 * safety_score
        + 0.10 * success_bonus
    )
    score = max(0.0, min(1.0, score))
    return {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "score": float(round(score, 6)),
        "success": bool(success),
        "components": {
            "health_score": float(round(health_score, 6)),
            "adversary_score": float(round(adversary_score, 6)),
            "safety_score": float(round(safety_score, 6)),
            "success_bonus": success_bonus,
        },
    }

