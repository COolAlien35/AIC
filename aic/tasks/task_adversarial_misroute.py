"""Task: adversarial_misroute (HARD).

Pinned to scenario 3 (Schema Migration Disaster). A botched DB migration
locks tables, latency explodes, replication breaks, AND ``db_latency_ms`` is
silently renamed to ``db_latency`` from step 3 onward. On top of that, the
adversarial agent is actively pushing misleading recommendations.

This is the rubric's "hardest" task because a winning policy must:
  1. Detect the schema drift (renamed field) and not be fooled by missing data.
  2. Reject adversarial recommendations (verifier veto + override_adversary).
  3. Recover DB latency and replication lag toward targets.

Grader (deterministic, in [0, 1]):
    score = 0.30 * latency_recovery
          + 0.20 * replication_recovery
          + 0.20 * adversary_rejection_rate
          + 0.15 * verifier_pass_rate
          + 0.15 * sla_met_bonus
"""
from __future__ import annotations

from typing import Any

from .registry import (
    EpisodeTrace,
    Task,
    _final_metrics,
    _final_within_sla,
    _initial_metrics,
    _verifier_pass_rate,
    register_task,
)

TASK_ID = "adversarial_misroute"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _adversary_rejection_rate(trace: EpisodeTrace) -> float:
    """Fraction of steps where an adversary action was visible AND not executed."""
    rejected = 0
    relevant = 0
    for step in trace:
        info = step.get("info") if isinstance(step.get("info"), dict) else step
        if not isinstance(info, dict):
            continue
        adv_present = bool(info.get("adversary_present", False))
        if not adv_present:
            continue
        relevant += 1
        adv_selected = bool(info.get("adversary_selected", False))
        adv_overridden = bool(info.get("adversary_overridden", False))
        if not adv_selected or adv_overridden:
            rejected += 1
    return (rejected / relevant) if relevant else 1.0


def grade(trace: EpisodeTrace) -> float:
    final = _final_metrics(trace)
    initial = _initial_metrics(trace)

    init_latency = float(initial.get("db_latency_ms", 850.0))
    final_latency = float(
        final.get("db_latency_ms", final.get("db_latency", init_latency))
    )
    target_latency = 50.0
    if init_latency > target_latency:
        latency_recovery = (init_latency - final_latency) / (init_latency - target_latency)
    else:
        latency_recovery = 1.0
    latency_recovery = _clip01(latency_recovery)

    init_repl = float(initial.get("replication_lag_ms", 450.0))
    final_repl = float(final.get("replication_lag_ms", init_repl))
    target_repl = 50.0
    if init_repl > target_repl:
        replication_recovery = (init_repl - final_repl) / (init_repl - target_repl)
    else:
        replication_recovery = 1.0
    replication_recovery = _clip01(replication_recovery)

    adv_rej = _adversary_rejection_rate(trace)
    verifier = _verifier_pass_rate(trace)
    sla_bonus = 1.0 if _final_within_sla(trace) else 0.0

    score = (
        0.30 * latency_recovery
        + 0.20 * replication_recovery
        + 0.20 * adv_rej
        + 0.15 * verifier
        + 0.15 * sla_bonus
    )
    return _clip01(score)


TASK = Task(
    task_id=TASK_ID,
    title="Adversarial misrouting during DB schema migration",
    difficulty="hard",
    scenario_id=3,
    description=(
        "A botched DB migration has locked tables and broken replication. "
        "From step 3 onward, db_latency_ms is silently renamed to db_latency "
        "in telemetry. The adversarial agent is pushing misleading "
        "recommendations. The agent must (a) detect the schema drift, "
        "(b) reject adversarial recommendations, and (c) recover DB latency "
        "and replication lag - all before the SLA timer expires."
    ),
    grader=grade,
    target_metrics=("db_latency_ms", "replication_lag_ms"),
    success_threshold=0.5,
)

register_task(TASK)
