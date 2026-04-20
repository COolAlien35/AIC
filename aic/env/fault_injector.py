# aic/env/fault_injector.py
"""
Fault injection system for the AIC environment.

4 fault modes:
  - memory_leak:              Gradual memory/CPU exhaustion + latency creep
  - db_connection_saturation: Connection pool saturation + DB degradation
  - network_storm:            Network I/O spike + error rate + queue backup
  - cascading_failure:        All three above at 0.6x intensity simultaneously

Drift decays by 0.95^step so faults slightly fade unless the orchestrator
fails to act. After step 15, drift is halved to model natural stabilisation.
"""
from typing import Optional


# Per-mode drift rates (metric → per-step contribution)
_MEMORY_LEAK: dict[str, float] = {
    "mem_pct": +2.0,
    "cpu_pct": +1.5,
    "pod_restarts": +0.3,
    "p95_latency_ms": +50.0,
}

_DB_CONNECTION_SATURATION: dict[str, float] = {
    "conn_pool_pct": +1.5,
    "db_latency_ms": +80.0,
    "replication_lag_ms": +20.0,
}

_NETWORK_STORM: dict[str, float] = {
    "net_io_mbps": +30.0,
    "error_rate_pct": +1.0,
    "queue_depth": +60.0,
}

# Cascading failure is all three at 60% strength
_CASCADING_FACTOR: float = 0.6


def _build_cascading() -> dict[str, float]:
    merged: dict[str, float] = {}
    for base in (_MEMORY_LEAK, _DB_CONNECTION_SATURATION, _NETWORK_STORM):
        for metric, rate in base.items():
            merged[metric] = merged.get(metric, 0.0) + rate * _CASCADING_FACTOR
    return merged


FAULT_MODES: dict[str, dict[str, float]] = {
    "memory_leak": _MEMORY_LEAK,
    "db_connection_saturation": _DB_CONNECTION_SATURATION,
    "network_storm": _NETWORK_STORM,
    "cascading_failure": _build_cascading(),
}


class FaultInjector:
    """
    Produces per-step fault contributions for a chosen fault mode.

    Usage:
        fi = FaultInjector("cascading_failure")
        contributions = fi.get_contributions(step=3)
        world_state.step(action_deltas={}, fault_contributions=contributions)
    """

    DECAY_BASE: float = 0.95
    LATE_STEP_THRESHOLD: int = 15

    def __init__(self, fault_mode: str = "cascading_failure"):
        if fault_mode not in FAULT_MODES:
            raise ValueError(
                f"Unknown fault_mode '{fault_mode}'. "
                f"Valid modes: {list(FAULT_MODES.keys())}"
            )
        self.fault_mode = fault_mode
        self._base_rates: dict[str, float] = FAULT_MODES[fault_mode]

    def get_contributions(self, step: int) -> dict[str, float]:
        """
        Return per-metric fault drift for the given step.

        Drift decays exponentially: rate × 0.95^step.
        After step 15, drift is further halved.
        """
        decay = self.DECAY_BASE ** step
        late_factor = 0.5 if step > self.LATE_STEP_THRESHOLD else 1.0

        contributions: dict[str, float] = {}
        for metric, base_rate in self._base_rates.items():
            contributions[metric] = base_rate * decay * late_factor
        return contributions

    def is_recoverable(self, world_state_metrics: dict[str, float]) -> bool:
        """
        Returns True if the system health is above 0.3 (system can still recover).
        Uses a simple heuristic based on the metric values.
        """
        from aic.env.world_state import WorldState
        from aic.utils.constants import METRIC_TARGETS

        # Quick health estimate without needing a full WorldState instance
        total = 0.0
        for name, target in METRIC_TARGETS.items():
            current = world_state_metrics.get(name, target)
            if target == 0.0:
                score = 1.0 if current <= 0.5 else max(0.0, 1.0 - current / 10.0)
            else:
                normalized_dist = abs(current - target) / target
                score = max(0.0, 1.0 - normalized_dist)
            total += score
        health = total / len(METRIC_TARGETS)
        return health > 0.3

    def get_fault_mode(self) -> str:
        """Return the active fault mode name."""
        return self.fault_mode

    def get_base_rates(self) -> dict[str, float]:
        """Return the base (un-decayed) drift rates for the active fault mode."""
        return self._base_rates.copy()
