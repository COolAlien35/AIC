"""Task: canary_blackout (MEDIUM).

Pinned to scenario 1 (Canary Failure). A buggy canary deployment is taking
10% of traffic; error_rate_pct spikes from ~0.5% to 2%+ and observability
goes dark on ``error_rate_pct`` for steps 5..8 (NaN blackout). The agent
must recover error rate AND latency despite the blacked-out signal.

Grader (deterministic, in [0, 1]):
    score = 0.40 * error_rate_recovery
          + 0.25 * p95_latency_recovery
          + 0.15 * throughput_recovery
          + 0.10 * verifier_pass_rate
          + 0.10 * sla_met_bonus
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

TASK_ID = "canary_blackout"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _rel_recovery(initial: float, final: float, target: float, *, lower_is_better: bool) -> float:
    if lower_is_better:
        if initial <= target:
            return 1.0
        return _clip01((initial - final) / (initial - target))
    if initial >= target:
        return 1.0
    return _clip01((final - initial) / (target - initial))


def grade(trace: EpisodeTrace) -> float:
    final = _final_metrics(trace)
    initial = _initial_metrics(trace)

    err_recovery = _rel_recovery(
        float(initial.get("error_rate_pct", 18.5)),
        float(final.get("error_rate_pct", initial.get("error_rate_pct", 18.5))),
        target=0.5,
        lower_is_better=True,
    )
    p95_recovery = _rel_recovery(
        float(initial.get("p95_latency_ms", 3200.0)),
        float(final.get("p95_latency_ms", initial.get("p95_latency_ms", 3200.0))),
        target=200.0,
        lower_is_better=True,
    )
    tput_recovery = _rel_recovery(
        float(initial.get("throughput_rps", 180.0)),
        float(final.get("throughput_rps", initial.get("throughput_rps", 180.0))),
        target=1000.0,
        lower_is_better=False,
    )
    sla_bonus = 1.0 if _final_within_sla(trace) else 0.0
    verifier = _verifier_pass_rate(trace)

    score = (
        0.40 * err_recovery
        + 0.25 * p95_recovery
        + 0.15 * tput_recovery
        + 0.10 * verifier
        + 0.10 * sla_bonus
    )
    return _clip01(score)


TASK = Task(
    task_id=TASK_ID,
    title="Canary failure recovery during telemetry blackout",
    difficulty="medium",
    scenario_id=1,
    description=(
        "A buggy canary deployment is corrupting 10% of traffic, driving "
        "error rate to 18.5% and p95 latency to 3.2s. Observability is "
        "partially blinded - error_rate_pct goes NaN for steps 5..8. "
        "Recover error rate, p95 latency, and throughput before the SLA "
        "deadline despite the blacked-out signal."
    ),
    grader=grade,
    target_metrics=("error_rate_pct", "p95_latency_ms", "throughput_rps"),
    success_threshold=0.55,
)

register_task(TASK)
