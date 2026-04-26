"""Task: db_pool_recovery (EASY).

Pinned to scenario 0 (Cache Stampede). The cache eviction storm drives DB
latency from ~50ms to 850ms+ and saturates the connection pool to 98%. The
agent must propose actions that bring DB latency back toward target (50ms)
and pull the connection pool below 80% within the SLA window.

Grader (deterministic, in [0, 1]):
    score = 0.55 * latency_recovery
          + 0.30 * pool_recovery
          + 0.10 * sla_met_bonus
          + 0.05 * verifier_pass_rate
"""
from __future__ import annotations

from .registry import (
    EpisodeTrace,
    Task,
    _final_metrics,
    _final_within_sla,
    _initial_metrics,
    _verifier_pass_rate,
    register_task,
)

TASK_ID = "db_pool_recovery"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def grade(trace: EpisodeTrace) -> float:
    final = _final_metrics(trace)
    initial = _initial_metrics(trace)

    init_latency = float(initial.get("db_latency_ms", 850.0))
    final_latency = float(final.get("db_latency_ms", init_latency))
    target_latency = 50.0
    if init_latency > target_latency:
        latency_recovery = (init_latency - final_latency) / (init_latency - target_latency)
    else:
        latency_recovery = 1.0
    latency_recovery = _clip01(latency_recovery)

    init_pool = float(initial.get("conn_pool_pct", 98.0))
    final_pool = float(final.get("conn_pool_pct", init_pool))
    target_pool = 60.0
    if init_pool > target_pool:
        pool_recovery = (init_pool - final_pool) / (init_pool - target_pool)
    else:
        pool_recovery = 1.0
    pool_recovery = _clip01(pool_recovery)

    sla_bonus = 1.0 if _final_within_sla(trace) else 0.0
    verifier = _verifier_pass_rate(trace)

    score = (
        0.55 * latency_recovery
        + 0.30 * pool_recovery
        + 0.10 * sla_bonus
        + 0.05 * verifier
    )
    return _clip01(score)


TASK = Task(
    task_id=TASK_ID,
    title="DB pool recovery from cache stampede",
    difficulty="easy",
    scenario_id=0,
    description=(
        "Cache cluster eviction storm has driven DB latency to 850ms and "
        "connection pool utilization to 98%. Recover both metrics toward "
        "their targets (50ms latency, 60% pool) before the SLA timer expires."
    ),
    grader=grade,
    target_metrics=("db_latency_ms", "conn_pool_pct"),
    success_threshold=0.6,
)

register_task(TASK)
