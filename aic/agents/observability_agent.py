# aic/agents/observability_agent.py
"""
Observability Integrity Agent — reasons about whether telemetry is trustworthy.

Detects:
  1. Stale metrics    — same value repeated for 3+ consecutive steps
  2. Suspicious spikes — single-step jump > 3σ of rolling window
  3. Cross-metric inconsistencies — throughput up but error_rate also up
  4. Partial dropout  — metrics returning None/NaN unexpectedly
  5. Unit anomalies   — values outside physical plausible range
  6. Schema drift     — key disappears from metric dict mid-episode

Outputs an ObservabilityReport every step with:
  - overall_trustworthy: bool
  - suspect_metrics: list[str]
  - corruption_type: str (dominant type if detected)
  - trust_adjustment: float (negative penalty to apply to simulation)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from aic.utils.constants import METRIC_TARGETS


# Physical plausible bounds for sanity checking
_PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "db_latency_ms": (1.0, 60_000.0),
    "conn_pool_pct": (0.0, 100.0),
    "replication_lag_ms": (0.0, 30_000.0),
    "cpu_pct": (0.0, 100.0),
    "mem_pct": (0.0, 100.0),
    "pod_restarts": (0.0, 1000.0),
    "net_io_mbps": (0.0, 100_000.0),
    "error_rate_pct": (0.0, 100.0),
    "p95_latency_ms": (1.0, 300_000.0),
    "queue_depth": (0.0, 1_000_000.0),
    "throughput_rps": (0.0, 1_000_000.0),
    "sla_compliance_pct": (0.0, 100.0),
}

_STALE_WINDOW = 3       # steps before declaring a metric stale
_SPIKE_SIGMA = 3.0      # sigma threshold for spike detection
_ROLLING_WINDOW = 5     # steps used for rolling mean/std


@dataclass
class ObservabilityReport:
    """Telemetry trustworthiness assessment for one step."""
    step: int
    overall_trustworthy: bool
    suspect_metrics: list[str]
    corruption_types: list[str]       # e.g. ["stale_metric", "suspicious_spike"]
    trust_adjustment: float           # penalty to simulation confidence: 0.0 (clean) to -1.0 (fully corrupted)
    details: list[str]                # human-readable explanations

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "overall_trustworthy": self.overall_trustworthy,
            "suspect_metrics": self.suspect_metrics,
            "corruption_types": self.corruption_types,
            "trust_adjustment": self.trust_adjustment,
            "details": self.details,
        }


class ObservabilityAgent:
    """
    Monitors telemetry quality step-by-step.

    Does not propose actions. Outputs ObservabilityReport consumed by
    the orchestrator to adjust simulation confidence.
    """

    def __init__(self, stale_window: int = _STALE_WINDOW, spike_sigma: float = _SPIKE_SIGMA):
        self._stale_window = stale_window
        self._spike_sigma = spike_sigma
        self._history: dict[str, deque] = {}   # metric → deque of recent values
        self._known_keys: Optional[set] = None  # first-seen metric keys
        self._step = 0

    def reset(self) -> None:
        self._history.clear()
        self._known_keys = None
        self._step = 0

    def assess(self, metrics: dict[str, float]) -> ObservabilityReport:
        """
        Assess the trustworthiness of the current metric snapshot.

        Args:
            metrics: Raw metric dict for current step.

        Returns:
            ObservabilityReport with findings.
        """
        step = self._step
        suspect: list[str] = []
        corruption_types: list[str] = []
        details: list[str] = []

        # ── 1. Schema drift: missing keys ─────────────────────────────────
        if self._known_keys is None:
            self._known_keys = set(metrics.keys())
        else:
            missing = self._known_keys - set(metrics.keys())
            for key in missing:
                suspect.append(key)
                if "schema_drift" not in corruption_types:
                    corruption_types.append("schema_drift")
                details.append(f"Key '{key}' disappeared from metric snapshot at step {step}.")

        # Update known keys (allow additions)
        self._known_keys.update(metrics.keys())

        for metric, value in metrics.items():
            # Update rolling history
            if metric not in self._history:
                self._history[metric] = deque(maxlen=_ROLLING_WINDOW)
            self._history[metric].append(value)
            history = list(self._history[metric])

            # ── 2. NaN / None dropout ──────────────────────────────────────
            if value is None or (isinstance(value, float) and math.isnan(value)):
                suspect.append(metric)
                if "partial_dropout" not in corruption_types:
                    corruption_types.append("partial_dropout")
                details.append(f"Metric '{metric}' returned NaN/None at step {step}.")
                continue

            # ── 3. Physical bounds check ───────────────────────────────────
            if metric in _PLAUSIBLE_BOUNDS:
                lo, hi = _PLAUSIBLE_BOUNDS[metric]
                if not (lo <= value <= hi):
                    suspect.append(metric)
                    if "unit_anomaly" not in corruption_types:
                        corruption_types.append("unit_anomaly")
                    details.append(
                        f"'{metric}'={value:.2f} is outside plausible bounds [{lo}, {hi}]. "
                        f"Possible unit shift."
                    )

            # ── 4. Stale metric ────────────────────────────────────────────
            if len(history) >= self._stale_window:
                recent = history[-self._stale_window:]
                if len(set(f"{v:.4f}" for v in recent)) == 1:
                    suspect.append(metric)
                    if "stale_metric" not in corruption_types:
                        corruption_types.append("stale_metric")
                    details.append(
                        f"'{metric}' has not changed for {self._stale_window} steps "
                        f"(value={value:.2f}). Possible telemetry freeze."
                    )

            # ── 5. Suspicious spike ────────────────────────────────────────
            if len(history) >= 3:
                window = history[:-1]  # all but latest
                mean = sum(window) / len(window)
                variance = sum((x - mean) ** 2 for x in window) / len(window)
                std = math.sqrt(variance) if variance > 0 else 0.0
                if std > 0 and abs(value - mean) > self._spike_sigma * std:
                    suspect.append(metric)
                    if "suspicious_spike" not in corruption_types:
                        corruption_types.append("suspicious_spike")
                    sigma_dist = abs(value - mean) / std
                    details.append(
                        f"'{metric}'={value:.1f} is {sigma_dist:.1f}σ from recent mean={mean:.1f}. "
                        f"Possible synthetic injection or measurement error."
                    )

        # ── 6. Cross-metric consistency check ─────────────────────────────
        throughput = metrics.get("throughput_rps", None)
        error_rate = metrics.get("error_rate_pct", None)
        if (
            throughput is not None and error_rate is not None
            and throughput > METRIC_TARGETS.get("throughput_rps", 1000) * 1.2
            and error_rate > METRIC_TARGETS.get("error_rate_pct", 0.5) * 5
        ):
            if "cross_metric_inconsistency" not in corruption_types:
                corruption_types.append("cross_metric_inconsistency")
            details.append(
                f"Inconsistency: throughput={throughput:.0f}rps (elevated) while "
                f"error_rate={error_rate:.1f}% (also elevated). "
                f"This combination is suspicious — possible synthetic data injection."
            )

        # Deduplicate suspect list
        suspect = list(dict.fromkeys(suspect))

        # Compute trust adjustment
        num_suspect = len(suspect)
        if num_suspect == 0:
            trust_adjustment = 0.0
        elif num_suspect <= 2:
            trust_adjustment = -0.1 * num_suspect
        else:
            trust_adjustment = max(-0.8, -0.1 * num_suspect)

        overall_trustworthy = len(corruption_types) == 0 or (
            len(corruption_types) == 1
            and corruption_types[0] in ("suspicious_spike",)
            and num_suspect <= 1
        )

        self._step += 1

        return ObservabilityReport(
            step=step,
            overall_trustworthy=overall_trustworthy,
            suspect_metrics=suspect,
            corruption_types=corruption_types,
            trust_adjustment=trust_adjustment,
            details=details,
        )

    def get_cumulative_corruption_score(self) -> float:
        """Returns fraction of steps with any corruption detected."""
        if self._step == 0:
            return 0.0
        # We track this implicitly via step count
        return 0.0  # placeholder; dashboard can compute from per-step reports
