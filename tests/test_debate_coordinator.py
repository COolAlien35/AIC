# tests/test_debate_coordinator.py
"""Tests for the Agent Debate Coordinator."""
import pytest
from aic.agents.debate_coordinator import DebateCoordinator, DebateRound
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_DB, AGENT_INFRA, AGENT_APP, AGENT_ADV, METRIC_TARGETS


def _make_rec(agent: str, confidence: float, targets: list[str],
              risk: float = 0.2, blast: str = "low", rollback: str = "undo") -> SubAgentRecommendation:
    return SubAgentRecommendation(
        agent_name=agent,
        action=f"Action from {agent}",
        reasoning=f"Reasoning from {agent} with confidence {confidence}",
        confidence=confidence,
        target_metrics=targets,
        expected_impact={t: -10.0 for t in targets},
        risk_score=risk,
        blast_radius=blast,
        rollback_plan=rollback,
    )


class TestDebateCoordinator:

    def test_empty_recommendations(self):
        dc = DebateCoordinator()
        recs, round_ = dc.run_debate([], {}, None)
        assert recs == []
        assert isinstance(round_, DebateRound)
        assert round_.debate_changed_selection is False

    def test_single_recommendation_no_debate(self):
        dc = DebateCoordinator()
        rec = _make_rec(AGENT_DB, 0.8, ["db_latency_ms"])
        recs, round_ = dc.run_debate([rec], METRIC_TARGETS.copy(), None)
        assert len(recs) == 1
        assert round_.debate_changed_selection is False

    def test_db_criticises_non_db_when_coupling_high(self):
        """DB agent should criticise InfraAgent when conn_pool is critical."""
        dc = DebateCoordinator()
        infra_rec = _make_rec(AGENT_INFRA, 0.9, ["cpu_pct", "mem_pct"])
        db_rec = _make_rec(AGENT_DB, 0.7, ["db_latency_ms", "conn_pool_pct"])

        metrics = METRIC_TARGETS.copy()
        metrics["conn_pool_pct"] = 95.0  # well above 60% target
        metrics["db_latency_ms"] = 500.0  # well above 50ms target

        recs, round_ = dc.run_debate([infra_rec, db_rec], metrics, None)
        # DB agent should have criticised infra
        assert len(round_.criticisms) > 0
        assert any(c["critic"] == AGENT_DB for c in round_.criticisms)
        # The criticised rec should have text in its criticisms list
        assert len(infra_rec.criticisms) > 0

    def test_db_supports_db_targeting_rec(self):
        """DB agent should support the candidate that targets DB metrics."""
        dc = DebateCoordinator()
        infra_rec = _make_rec(AGENT_INFRA, 0.9, ["cpu_pct"])
        db_rec = _make_rec(AGENT_DB, 0.7, ["db_latency_ms", "conn_pool_pct"])

        metrics = METRIC_TARGETS.copy()
        metrics["conn_pool_pct"] = 95.0
        metrics["db_latency_ms"] = 500.0

        _, round_ = dc.run_debate([infra_rec, db_rec], metrics, None)
        assert len(round_.supports) > 0
        assert any(s["target_agent"] == AGENT_DB for s in round_.supports)

    def test_infra_criticises_on_cpu_critical(self):
        """InfraAgent should warn when CPU is critical but action ignores it."""
        dc = DebateCoordinator()
        app_rec = _make_rec(AGENT_APP, 0.9, ["error_rate_pct"])
        infra_rec = _make_rec(AGENT_INFRA, 0.6, ["cpu_pct"])

        metrics = METRIC_TARGETS.copy()
        metrics["cpu_pct"] = 92.0  # above 85% critical

        _, round_ = dc.run_debate([app_rec, infra_rec], metrics, None)
        assert any(c["critic"] == AGENT_INFRA for c in round_.criticisms)

    def test_high_blast_no_rollback_challenge(self):
        dc = DebateCoordinator()
        bad_rec = _make_rec(AGENT_APP, 0.9, ["error_rate_pct"],
                            blast="high", rollback="")  # no rollback
        safe_rec = _make_rec(AGENT_DB, 0.6, ["db_latency_ms"])

        _, round_ = dc.run_debate([bad_rec, safe_rec], METRIC_TARGETS.copy(), None)
        assert any("rollback" in c["text"].lower() for c in round_.criticisms)

    def test_rebuttal_generated_when_criticised(self):
        dc = DebateCoordinator()
        infra_rec = _make_rec(AGENT_INFRA, 0.9, ["cpu_pct"])
        db_rec = _make_rec(AGENT_DB, 0.7, ["db_latency_ms"])

        metrics = METRIC_TARGETS.copy()
        metrics["conn_pool_pct"] = 95.0
        metrics["db_latency_ms"] = 500.0

        _, _ = dc.run_debate([infra_rec, db_rec], metrics, None)
        # Top rec should have rebuttal if it was criticised
        if infra_rec.criticisms:
            assert len(infra_rec.rebuttals) > 0

    def test_adversary_excluded_from_debate(self):
        dc = DebateCoordinator()
        adv_rec = _make_rec(AGENT_ADV, 0.99, ["error_rate_pct"])
        db_rec = _make_rec(AGENT_DB, 0.6, ["db_latency_ms"])

        _, round_ = dc.run_debate([adv_rec, db_rec], METRIC_TARGETS.copy(), None)
        # Adversary should not be a critic or target in debate
        for c in round_.criticisms:
            assert c["critic"] != AGENT_ADV

    def test_reset_clears_rounds(self):
        dc = DebateCoordinator()
        rec = _make_rec(AGENT_DB, 0.8, ["db_latency_ms"])
        dc.run_debate([rec], {}, None)
        assert len(dc.get_rounds()) == 1
        dc.reset()
        assert len(dc.get_rounds()) == 0

    def test_summarise(self):
        dc = DebateCoordinator()
        rec = _make_rec(AGENT_DB, 0.8, ["db_latency_ms"])
        dc.run_debate([rec], {}, None)
        s = dc.summarise()
        assert "total_rounds" in s
        assert s["total_rounds"] == 1
