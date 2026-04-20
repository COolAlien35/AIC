# aic/env/world_state.py
"""
Manages the 12-metric production environment state.

Evolution formula per step:
  metric(t+1) = metric(t) + action_delta + fault_contribution + noise
  noise ~ N(0, NOISE_STD)

DB→App causal coupling:
  If conn_pool_pct increased by delta at step t,
  then p95_latency_ms increases by ALPHA_DB_APP * delta at step t+DB_APP_LAG_STEPS.
"""
import copy
from typing import Optional

import numpy as np

from aic.utils.constants import (
    METRIC_TARGETS, METRIC_FAULT_INIT, NOISE_STD,
    ALPHA_DB_APP, DB_APP_LAG_STEPS, SLA_HEALTH_THRESHOLD,
    OBS_DB, OBS_INFRA, OBS_APP,
)


# Hard physical bounds for each metric
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


class WorldState:
    """
    Manages the 12-metric production environment state.

    Evolution formula per step:
      metric(t+1) = metric(t) + action_delta + fault_contribution + noise
      noise ~ N(0, NOISE_STD)

    DB→App causal coupling:
      If conn_pool_pct increased by delta at step t,
      then p95_latency_ms increases by ALPHA_DB_APP * delta at step t+DB_APP_LAG_STEPS.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.metrics: dict[str, float] = copy.deepcopy(METRIC_FAULT_INIT)
        self.targets: dict[str, float] = copy.deepcopy(METRIC_TARGETS)
        # Ring buffer for DB→App lag: stores conn_pool_pct deltas
        self._db_delta_buffer: list[float] = [0.0] * DB_APP_LAG_STEPS
        # Track applied actions for causal attribution
        self._last_action_deltas: dict[str, float] = {}

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset to fault initial state."""
        if rng is not None:
            self.rng = rng
        self.metrics = copy.deepcopy(METRIC_FAULT_INIT)
        self._db_delta_buffer = [0.0] * DB_APP_LAG_STEPS
        self._last_action_deltas = {}

    def step(
        self,
        action_deltas: dict[str, float],
        fault_contributions: dict[str, float],
    ) -> dict[str, float]:
        """
        Advance world state by one step.

        Args:
            action_deltas: Changes from orchestrator action. Keys are metric names.
                           Positive = metric increases, negative = metric decreases.
                           Only metrics the action targets are included.
            fault_contributions: Ongoing fault drift per metric. Keys are metric names.
                                 Positive = metric getting worse, negative = recovering.

        Returns:
            Updated metrics dict (same object, mutated in place, also returned).
        """
        # 1. Record DB pool delta for lag buffer
        db_pool_delta = (
            action_deltas.get("conn_pool_pct", 0.0)
            + fault_contributions.get("conn_pool_pct", 0.0)
        )

        # 2. Apply lagged DB→App coupling from buffer
        app_lag_effect = self._db_delta_buffer.pop(0) * ALPHA_DB_APP
        self._db_delta_buffer.append(db_pool_delta)

        # 3. Apply all updates
        noise = self.rng.normal(0, NOISE_STD, size=len(self.metrics))

        for i, metric_name in enumerate(sorted(self.metrics.keys())):
            delta = action_deltas.get(metric_name, 0.0)
            fault = fault_contributions.get(metric_name, 0.0)
            noise_val = noise[i]

            # Apply lag effect only to p95_latency_ms
            lag = app_lag_effect if metric_name == "p95_latency_ms" else 0.0

            new_value = self.metrics[metric_name] + delta + fault + noise_val + lag

            # Clip to physically valid ranges
            new_value = self._clip_metric(metric_name, new_value)
            self.metrics[metric_name] = new_value

        self._last_action_deltas = action_deltas
        return self.metrics

    def _clip_metric(self, name: str, value: float) -> float:
        """Clip metric to physically valid range."""
        lo, hi = _METRIC_CLIPS.get(name, (-float("inf"), float("inf")))
        return max(lo, min(hi, value))

    def get_health_score(self) -> float:
        """
        Compute normalized health score across all metrics.
        Returns 0.0 (all at fault init) to 1.0 (all at target).
        """
        total = 0.0
        for name, target in self.targets.items():
            current = self.metrics[name]
            if target == 0.0:
                # For pod_restarts: score = 1 if current <= 0.5
                score = 1.0 if current <= 0.5 else max(0.0, 1.0 - current / 10.0)
            else:
                # Normalized distance from target
                normalized_dist = abs(current - target) / target
                score = max(0.0, 1.0 - normalized_dist)
            total += score
        return total / len(self.targets)

    def is_within_sla(self) -> bool:
        """
        Returns True if all metrics are within SLA_HEALTH_THRESHOLD of target.
        """
        for name, target in self.targets.items():
            current = self.metrics[name]
            if target == 0.0:
                if current > 0.5:
                    return False
            else:
                if abs(current - target) / target > SLA_HEALTH_THRESHOLD:
                    return False
        return True

    def get_db_observation(self) -> dict[str, float]:
        """Return only the DB agent's observable metrics."""
        return {k: self.metrics[k] for k in OBS_DB}

    def get_infra_observation(self) -> dict[str, float]:
        """Return only the Infra agent's observable metrics."""
        return {k: self.metrics[k] for k in OBS_INFRA}

    def get_app_observation(self) -> dict[str, float]:
        """Return only the App agent's observable metrics."""
        return {k: self.metrics[k] for k in OBS_APP}

    def snapshot(self) -> dict[str, float]:
        """Return a copy of current metrics for logging."""
        return copy.deepcopy(self.metrics)
