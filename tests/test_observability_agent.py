# tests/test_observability_agent.py
"""Tests for the Observability Integrity Agent."""
import math
import pytest
from aic.agents.observability_agent import ObservabilityAgent, ObservabilityReport
from aic.utils.constants import METRIC_TARGETS


class TestObservabilityAgent:

    def test_clean_metrics_are_trustworthy(self):
        obs = ObservabilityAgent()
        report = obs.assess(METRIC_TARGETS.copy())
        assert isinstance(report, ObservabilityReport)
        assert report.overall_trustworthy is True
        assert report.trust_adjustment == 0.0
        assert len(report.suspect_metrics) == 0

    def test_nan_detected_as_partial_dropout(self):
        obs = ObservabilityAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["db_latency_ms"] = float("nan")
        report = obs.assess(metrics)
        assert "db_latency_ms" in report.suspect_metrics
        assert "partial_dropout" in report.corruption_types
        assert report.overall_trustworthy is False

    def test_none_detected_as_partial_dropout(self):
        obs = ObservabilityAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["cpu_pct"] = None
        report = obs.assess(metrics)
        assert "cpu_pct" in report.suspect_metrics
        assert "partial_dropout" in report.corruption_types

    def test_out_of_bounds_detected_as_unit_anomaly(self):
        obs = ObservabilityAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["cpu_pct"] = 250.0  # above 100% — impossible
        report = obs.assess(metrics)
        assert "cpu_pct" in report.suspect_metrics
        assert "unit_anomaly" in report.corruption_types

    def test_stale_metric_detected_after_window(self):
        obs = ObservabilityAgent(stale_window=3)
        metrics = METRIC_TARGETS.copy()
        # Same value 3 times = stale
        obs.assess(metrics)
        obs.assess(metrics)
        report = obs.assess(metrics)  # 3rd identical reading
        assert "stale_metric" in report.corruption_types

    def test_spike_detected(self):
        obs = ObservabilityAgent(spike_sigma=2.0)
        # Build history with slightly varying values (avoids stale detection)
        for i in range(4):
            m = {k: v + i * 0.01 for k, v in METRIC_TARGETS.items()}
            obs.assess(m)
        # Now spike one metric
        spiked = {k: v + 4 * 0.01 for k, v in METRIC_TARGETS.items()}
        spiked["db_latency_ms"] = 5000.0  # huge spike from ~50
        report = obs.assess(spiked)
        assert "suspicious_spike" in report.corruption_types
        assert "db_latency_ms" in report.suspect_metrics

    def test_schema_drift_missing_key(self):
        obs = ObservabilityAgent()
        obs.assess(METRIC_TARGETS.copy())  # establish baseline
        # Remove a key
        partial = {k: v for k, v in METRIC_TARGETS.items() if k != "net_io_mbps"}
        report = obs.assess(partial)
        assert "schema_drift" in report.corruption_types
        assert "net_io_mbps" in report.suspect_metrics

    def test_trust_adjustment_scales_with_suspects(self):
        obs = ObservabilityAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["db_latency_ms"] = float("nan")
        metrics["cpu_pct"] = float("nan")
        metrics["mem_pct"] = float("nan")
        report = obs.assess(metrics)
        assert report.trust_adjustment < -0.2

    def test_reset_clears_state(self):
        obs = ObservabilityAgent()
        obs.assess(METRIC_TARGETS.copy())
        obs.reset()
        assert obs._step == 0
        assert obs._known_keys is None

    def test_step_increments(self):
        obs = ObservabilityAgent()
        assert obs._step == 0
        obs.assess(METRIC_TARGETS.copy())
        assert obs._step == 1
        obs.assess(METRIC_TARGETS.copy())
        assert obs._step == 2
