# aic/env/schema_drift.py
"""
Schema drift injector. Simulates silent API contract changes mid-episode.

Three drift types:
1. field_rename: p95_latency_ms → p95_latency (key changed, value unchanged)
2. unit_shift: replication_lag_ms → replication_lag_ms (key same, value × 0.001)
3. silent_null: conn_pool_pct returns None for NULL_DRIFT_DURATION consecutive steps

The injector is initialized at episode start with a fixed t_drift.
It operates transparently — the orchestrator must detect the anomaly.
"""
from typing import Any

from aic.utils.constants import (
    DRIFT_FIELD_RENAME, DRIFT_UNIT_SHIFT, DRIFT_SILENT_NULL,
    DRIFT_TYPES, NULL_DRIFT_DURATION,
)


DRIFT_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    DRIFT_FIELD_RENAME: {
        "affected_service": "app",
        "original_field": "p95_latency_ms",
        "drifted_field": "p95_latency",       # Key renamed, value unchanged
        "description": "Field 'p95_latency_ms' renamed to 'p95_latency' without notification",
    },
    DRIFT_UNIT_SHIFT: {
        "affected_service": "db",
        "original_field": "replication_lag_ms",
        "drifted_field": "replication_lag_ms",  # Key same, value ÷ 1000
        "scale_factor": 0.001,
        "description": "Field 'replication_lag_ms' now returns value in seconds (÷1000)",
    },
    DRIFT_SILENT_NULL: {
        "affected_service": "db",
        "original_field": "conn_pool_pct",
        "drifted_field": "conn_pool_pct",       # Key same, value None
        "null_duration": NULL_DRIFT_DURATION,
        "description": f"Field 'conn_pool_pct' returns null for {NULL_DRIFT_DURATION} steps",
    },
}


class SchemaDriftInjector:
    """
    Manages schema drift injection for one episode.

    Usage:
        injector = SchemaDriftInjector(t_drift=11, drift_type="field_rename")
        raw_response = injector.inject(step=11, service="app",
                                       raw_response={"p95_latency_ms": 3200.0, ...})
        # Returns {"p95_latency": 3200.0, ...}  — field renamed
    """

    def __init__(self, t_drift: int, drift_type: str):
        if drift_type not in DRIFT_TYPES:
            raise ValueError(
                f"Unknown drift type: {drift_type}. Valid: {DRIFT_TYPES}"
            )
        if not 0 <= t_drift <= 20:
            raise ValueError(f"t_drift must be in [0, 20], got {t_drift}")

        self.t_drift = t_drift
        self.drift_type = drift_type
        self.spec = DRIFT_SPECIFICATIONS[drift_type]
        self._null_steps_elapsed = 0
        self.active = False
        self.drift_ended = False

    def inject(
        self, step: int, service: str, raw_response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply drift to a service API response if drift is active for this step.

        Args:
            step: Current episode step (0-indexed).
            service: "db", "infra", or "app".
            raw_response: Dict of metric values from that service's API.

        Returns:
            Modified response dict (copy, not in-place).
        """
        # Only affect the target service
        if service != self.spec["affected_service"]:
            return raw_response.copy()

        # Drift begins at t_drift
        if step < self.t_drift:
            return raw_response.copy()

        # Silent null has a duration limit
        if self.drift_type == DRIFT_SILENT_NULL and self.drift_ended:
            return raw_response.copy()

        self.active = True
        result = raw_response.copy()
        original_field = self.spec["original_field"]

        if self.drift_type == DRIFT_FIELD_RENAME:
            drifted_field = self.spec["drifted_field"]
            if original_field in result:
                value = result.pop(original_field)
                result[drifted_field] = value

        elif self.drift_type == DRIFT_UNIT_SHIFT:
            if original_field in result and result[original_field] is not None:
                result[original_field] = (
                    result[original_field] * self.spec["scale_factor"]
                )

        elif self.drift_type == DRIFT_SILENT_NULL:
            if self._null_steps_elapsed < self.spec["null_duration"]:
                result[original_field] = None
                self._null_steps_elapsed += 1
            else:
                self.drift_ended = True
                self.active = False

        return result

    def was_active_at(self, step: int) -> bool:
        """Returns True if drift was active during the given step."""
        if self.drift_type == DRIFT_SILENT_NULL:
            return self.t_drift <= step < self.t_drift + NULL_DRIFT_DURATION
        return step >= self.t_drift

    def get_drift_description(self) -> str:
        """Return human-readable description of the drift."""
        return self.spec["description"]

    def get_affected_field(self) -> str:
        """Return the original field name affected by drift."""
        return self.spec["original_field"]
