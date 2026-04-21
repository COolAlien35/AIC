# aic/env/counterfactual_simulator.py
"""
Counterfactual Simulator — "sandbox" for previewing action outcomes.

Runs a lightweight simulation of the world state evolution formula:
    metric(t+1) = metric(t) + action_delta + fault_contribution + noise

Mastermind Refinement: Includes simplified propagation logic from Phase 8
so that DB actions correctly reflect downstream App impact in simulation.

Includes 10% Gaussian noise for realism (90% accuracy to real world).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from aic.utils.constants import METRIC_TARGETS, NOISE_STD


# Hard physical bounds (same as WorldState)
_METRIC_CLIPS: dict[str, tuple[float, float]] = {
    "db_latency_ms": (1.0, 10000.0),
    "conn_pool_pct": (0.0, 100.0),
    "replication_lag_ms": (0.0, 5000.0),
    "cpu_pct": (0.0, 100.0),
    "mem_pct": (0.0, 100.0),
    "pod_restarts": (0.0, 100.0),
    "net_io_mbps": (0.0, 10000.0),
    "error_rate_pct": (0.0, 100.0),
    "p95_latency_ms": (1.0, 30000.0),
    "queue_depth": (0.0, 10000.0),
    "throughput_rps": (0.0, 100000.0),
    "sla_compliance_pct": (0.0, 100.0),
}

# Simulation noise factor (10% of base noise = 90% accuracy)
SIM_NOISE_FACTOR: float = 0.10

# Default coupling map: maps metric changes to downstream metric effects.
# Format: {source_metric: [(target_metric, coupling_coefficient)]}
# This mirrors the ServiceTopology DAG at the metric level.
DEFAULT_COUPLING_MAP: dict[str, list[tuple[str, float]]] = {
    "db_latency_ms": [
        ("p95_latency_ms", 0.6),   # DB slowness → App latency
        ("error_rate_pct", 0.15),   # DB slowness → error rate increase
        ("queue_depth", 0.3),       # DB slowness → queue backlog
    ],
    "conn_pool_pct": [
        ("db_latency_ms", 0.4),    # Pool saturation → DB latency
        ("p95_latency_ms", 0.3),   # Pool saturation → App latency
    ],
    "queue_depth": [
        ("p95_latency_ms", 0.2),   # Queue backlog → App latency
        ("error_rate_pct", 0.1),   # Queue backlog → error rate
    ],
    "cpu_pct": [
        ("p95_latency_ms", 0.3),   # CPU pressure → latency
        ("error_rate_pct", 0.05),  # CPU pressure → errors
    ],
    "error_rate_pct": [
        ("sla_compliance_pct", -1.5),  # Errors → SLA degradation
        ("throughput_rps", -5.0),      # Errors → throughput drop
    ],
}


@dataclass
class SimulationResult:
    """Result of a counterfactual simulation."""
    predicted_metrics: dict[str, float]
    predicted_health: float
    impact_score: float      # lower = better (distance from targets)
    steps_simulated: int
    action_deltas: dict[str, float]


def _compute_health(metrics: dict[str, float]) -> float:
    """Compute health score from metrics (same formula as WorldState)."""
    total = 0.0
    count = 0
    for name, target in METRIC_TARGETS.items():
        current = metrics.get(name, target)
        if target == 0.0:
            score = 1.0 if current <= 0.5 else max(0.0, 1.0 - current / 10.0)
        else:
            normalized_dist = abs(current - target) / target
            score = max(0.0, 1.0 - normalized_dist)
        total += score
        count += 1
    return total / count if count > 0 else 0.0


def _compute_impact_score(metrics: dict[str, float]) -> float:
    """
    Compute distance from target state (lower = better).
    Sum of normalized distances across all metrics.
    """
    total = 0.0
    for name, target in METRIC_TARGETS.items():
        current = metrics.get(name, target)
        if target == 0.0:
            total += abs(current)
        else:
            total += abs(current - target) / target
    return total


def _apply_coupling(
    sim_metrics: dict[str, float],
    action_deltas: dict[str, float],
    coupling_map: dict[str, list[tuple[str, float]]],
) -> None:
    """
    Apply simplified propagation: for each metric in action_deltas,
    propagate the effect to downstream metrics via the coupling map.
    """
    for source_metric, delta in action_deltas.items():
        if source_metric not in coupling_map:
            continue
        for target_metric, coefficient in coupling_map[source_metric]:
            if target_metric in sim_metrics:
                sim_metrics[target_metric] += delta * coefficient


def simulate_action(
    current_metrics: dict[str, float],
    action_deltas: dict[str, float],
    fault_contributions: Optional[dict[str, float]] = None,
    coupling_map: Optional[dict[str, list[tuple[str, float]]]] = None,
    steps: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """
    Simulate the effect of action_deltas on the current world state.

    Runs `steps` iterations of the evolution formula in a sandbox.
    Includes simplified propagation logic so DB actions correctly
    reflect downstream App impact.

    Args:
        current_metrics: Current world state snapshot.
        action_deltas: Proposed metric changes from the action.
        fault_contributions: Optional ongoing fault pressure per step.
        coupling_map: Metric coupling map for propagation. Uses default if None.
        steps: Number of steps to simulate forward.
        rng: Optional numpy RNG for reproducibility.

    Returns:
        SimulationResult with predicted metrics and impact score.
    """
    if rng is None:
        rng = np.random.default_rng()

    cmap = coupling_map if coupling_map is not None else DEFAULT_COUPLING_MAP
    faults = fault_contributions or {}
    sim_metrics = copy.deepcopy(current_metrics)

    for t in range(steps):
        if t == 0:
            # Apply action deltas on first step
            for name, delta in action_deltas.items():
                if name in sim_metrics:
                    sim_metrics[name] += delta

            # Apply coupling propagation from the action
            _apply_coupling(sim_metrics, action_deltas, cmap)

        # Apply ongoing fault contributions
        for name, delta in faults.items():
            if name in sim_metrics:
                sim_metrics[name] += delta

        # Add simulation noise (10% of base noise)
        for name in list(sim_metrics.keys()):
            noise = rng.normal(0, NOISE_STD * SIM_NOISE_FACTOR)
            sim_metrics[name] += noise

            # Clip to physical bounds
            if name in _METRIC_CLIPS:
                lo, hi = _METRIC_CLIPS[name]
                sim_metrics[name] = max(lo, min(hi, sim_metrics[name]))

    predicted_health = _compute_health(sim_metrics)
    impact_score = _compute_impact_score(sim_metrics)

    return SimulationResult(
        predicted_metrics=sim_metrics,
        predicted_health=predicted_health,
        impact_score=impact_score,
        steps_simulated=steps,
        action_deltas=action_deltas,
    )


def compare_actions(
    current_metrics: dict[str, float],
    candidates: list[dict[str, float]],
    fault_contributions: Optional[dict[str, float]] = None,
    coupling_map: Optional[dict[str, list[tuple[str, float]]]] = None,
    steps: int = 2,
    seed: int = 42,
) -> list[SimulationResult]:
    """
    Simulate multiple candidate actions and return results sorted best first.

    Args:
        current_metrics: Current world state.
        candidates: List of action_deltas dicts to compare.
        fault_contributions: Ongoing fault pressure.
        coupling_map: Metric coupling map for propagation.
        steps: Simulation steps.
        seed: RNG seed for reproducibility.

    Returns:
        List of SimulationResult, sorted by impact_score (best=lowest first).
    """
    rng = np.random.default_rng(seed)
    results = []
    for action_deltas in candidates:
        result = simulate_action(
            current_metrics, action_deltas, fault_contributions,
            coupling_map, steps, rng,
        )
        results.append(result)
    results.sort(key=lambda r: r.impact_score)
    return results
