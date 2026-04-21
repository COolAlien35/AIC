# aic/agents/root_cause_analyst_agent.py
"""
Root Cause Analyst — Bayesian hypothesis ranking over Phase 8 scenarios.

Maintains a probability distribution over the 6 brutal scenarios.
Updates beliefs based on incoming telemetry by comparing observed metric
spikes against scenario fingerprints (characteristic metric signatures).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aic.utils.constants import METRIC_TARGETS


@dataclass
class RootCauseHypothesis:
    """Top root cause hypothesis with confidence."""
    scenario_name: str
    scenario_id: int
    confidence: float
    evidence: list[str]  # metrics that support this hypothesis


# Scenario fingerprints: which metrics spike characteristically for each scenario
# Values represent expected direction: positive = metric increases, negative = decreases
SCENARIO_FINGERPRINTS: dict[int, dict] = {
    0: {
        "name": "Cache Stampede",
        "signature": {
            "db_latency_ms": 1.0,
            "queue_depth": 1.0,
            "conn_pool_pct": 0.8,
            "error_rate_pct": 0.5,
        },
    },
    1: {
        "name": "Canary Failure",
        "signature": {
            "error_rate_pct": 1.0,
            "p95_latency_ms": 0.8,
            "throughput_rps": -0.6,
        },
    },
    2: {
        "name": "Regional Outage",
        "signature": {
            "net_io_mbps": 0.7,
            "error_rate_pct": 0.6,
            "p95_latency_ms": 0.5,
            "cpu_pct": 0.4,
            "mem_pct": 0.4,
            "db_latency_ms": 0.5,
            "queue_depth": 0.4,
        },
    },
    3: {
        "name": "Schema Migration Disaster",
        "signature": {
            "db_latency_ms": 1.0,
            "replication_lag_ms": 0.9,
            "conn_pool_pct": 0.7,
        },
    },
    4: {
        "name": "Credential Compromise",
        "signature": {
            "error_rate_pct": 1.0,
            "throughput_rps": -0.8,
            "p95_latency_ms": 0.5,
        },
    },
    5: {
        "name": "Queue Cascade",
        "signature": {
            "queue_depth": 1.0,
            "p95_latency_ms": 0.7,
            "error_rate_pct": 0.6,
            "cpu_pct": 0.3,
        },
    },
}


class RootCauseAnalyst:
    """
    Bayesian root cause analyst that ranks failure hypotheses.

    Maintains a probability distribution over the 6 scenarios and
    updates beliefs based on incoming telemetry data.

    Mastermind Refinement: includes time-decay to prevent getting stuck
    on early hypotheses when secondary symptoms mask the original cause.
    """

    # Decay rate toward uniform prior per update (evidence window effect)
    DECAY_RATE: float = 0.05

    def __init__(self, decay_rate: float = 0.05):
        # Uniform prior
        n = len(SCENARIO_FINGERPRINTS)
        self._beliefs: dict[int, float] = {
            sid: 1.0 / n for sid in SCENARIO_FINGERPRINTS
        }
        self._update_count: int = 0
        self.DECAY_RATE = decay_rate

    def reset(self) -> None:
        """Reset to uniform prior."""
        n = len(SCENARIO_FINGERPRINTS)
        self._beliefs = {sid: 1.0 / n for sid in SCENARIO_FINGERPRINTS}
        self._update_count = 0

    def update(self, metrics: dict[str, float]) -> None:
        """
        Bayesian update with time-decay: increase belief in scenarios whose
        fingerprints match the observed metric deviations.

        Time-decay prevents the analyst from getting "stuck" on an early
        hypothesis that has been superseded by secondary symptoms.
        """
        n = len(SCENARIO_FINGERPRINTS)
        uniform = 1.0 / n

        # Time-decay: blend current beliefs toward uniform prior
        for sid in self._beliefs:
            self._beliefs[sid] = (
                (1.0 - self.DECAY_RATE) * self._beliefs[sid]
                + self.DECAY_RATE * uniform
            )

        likelihoods: dict[int, float] = {}

        for sid, fingerprint in SCENARIO_FINGERPRINTS.items():
            signature = fingerprint["signature"]
            likelihood = 1.0

            for metric_name, expected_direction in signature.items():
                target = METRIC_TARGETS.get(metric_name)
                current = metrics.get(metric_name)

                if target is None or current is None:
                    continue

                # Compute how much this metric deviates from target
                if target == 0.0:
                    deviation = current / 10.0  # normalize
                else:
                    deviation = (current - target) / target

                # Does the deviation direction match the fingerprint?
                if expected_direction > 0:
                    # Expect increase: higher deviation = more evidence
                    match_score = max(0.0, deviation) * abs(expected_direction)
                else:
                    # Expect decrease: lower deviation = more evidence
                    match_score = max(0.0, -deviation) * abs(expected_direction)

                # Convert to likelihood multiplier
                likelihood *= (1.0 + min(match_score, 2.0))

            likelihoods[sid] = likelihood

        # Bayesian update: posterior ∝ prior × likelihood
        total = 0.0
        for sid in self._beliefs:
            self._beliefs[sid] *= likelihoods.get(sid, 1.0)
            total += self._beliefs[sid]

        # Normalize
        if total > 0:
            for sid in self._beliefs:
                self._beliefs[sid] /= total

        self._update_count += 1

    def get_top_hypothesis(self) -> RootCauseHypothesis:
        """Return the most likely root cause hypothesis."""
        best_sid = max(self._beliefs, key=self._beliefs.get)
        fingerprint = SCENARIO_FINGERPRINTS[best_sid]

        # Collect evidence: which metrics support this hypothesis
        evidence = list(fingerprint["signature"].keys())

        return RootCauseHypothesis(
            scenario_name=fingerprint["name"],
            scenario_id=best_sid,
            confidence=round(self._beliefs[best_sid], 4),
            evidence=evidence,
        )

    def get_all_hypotheses(self) -> list[RootCauseHypothesis]:
        """Return all hypotheses sorted by confidence."""
        results = []
        for sid, confidence in sorted(
            self._beliefs.items(), key=lambda x: x[1], reverse=True
        ):
            fp = SCENARIO_FINGERPRINTS[sid]
            results.append(RootCauseHypothesis(
                scenario_name=fp["name"],
                scenario_id=sid,
                confidence=round(confidence, 4),
                evidence=list(fp["signature"].keys()),
            ))
        return results

    def get_beliefs(self) -> dict[str, float]:
        """Return current belief distribution as name→confidence dict."""
        return {
            SCENARIO_FINGERPRINTS[sid]["name"]: round(conf, 4)
            for sid, conf in self._beliefs.items()
        }
