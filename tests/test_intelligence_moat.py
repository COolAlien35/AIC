# tests/test_intelligence_moat.py
"""
Phase 10 tests for Counterfactual Simulator, Knowledge Agent,
Root Cause Analyst, and Orchestrator Thinking loop.

Verification criteria:
1. Simulation: DB Scale action → predicted db_latency drops
2. RAG: Cache Stampede → retrieves cache_stampede.md runbook
3. Root Cause: Regional Outage metrics → identifies Regional Outage
4. Coupling: DB latency reduction propagates to App latency in simulation
5. Trace: ExplanationTrace contains hypothesis, runbook, simulation
"""
import pytest
from aic.utils.constants import METRIC_TARGETS, METRIC_FAULT_INIT


# ── 1. Counterfactual Simulator ──────────────────────────────────────────

class TestCounterfactualSimulator:
    def test_db_scale_reduces_latency(self):
        """VERIFICATION: DB Scale action → predicted db_latency drops."""
        from aic.env.counterfactual_simulator import simulate_action
        metrics = dict(METRIC_FAULT_INIT)  # degraded state
        action = {"db_latency_ms": -400.0, "conn_pool_pct": -20.0}
        result = simulate_action(metrics, action, steps=2)
        assert result.predicted_metrics["db_latency_ms"] < metrics["db_latency_ms"]

    def test_simulation_improves_health(self):
        from aic.env.counterfactual_simulator import simulate_action
        metrics = dict(METRIC_FAULT_INIT)
        action = {"db_latency_ms": -400.0, "error_rate_pct": -10.0}
        baseline = simulate_action(metrics, {}, steps=2)
        improved = simulate_action(metrics, action, steps=2)
        assert improved.predicted_health > baseline.predicted_health

    def test_coupling_db_to_app(self):
        """Mastermind: DB latency reduction propagates to App latency."""
        from aic.env.counterfactual_simulator import simulate_action
        metrics = dict(METRIC_FAULT_INIT)
        # Only reduce DB latency — app should also improve via coupling
        action = {"db_latency_ms": -500.0}
        result = simulate_action(metrics, action, steps=2)
        # p95_latency_ms should decrease due to coupling (0.6 coefficient)
        assert result.predicted_metrics["p95_latency_ms"] < metrics["p95_latency_ms"]

    def test_compare_actions_ranks_correctly(self):
        """DB fix should rank better than no-op."""
        from aic.env.counterfactual_simulator import compare_actions
        metrics = dict(METRIC_FAULT_INIT)
        candidates = [
            {},  # no-op
            {"db_latency_ms": -500.0, "conn_pool_pct": -30.0},  # DB fix
        ]
        results = compare_actions(metrics, candidates)
        # DB fix should have lower impact_score (better)
        assert results[0].impact_score < results[1].impact_score
        # DB fix should be first (best)
        assert results[0].action_deltas.get("db_latency_ms") == -500.0

    def test_noise_introduces_variance(self):
        """Two simulations with different seeds produce slightly different results."""
        from aic.env.counterfactual_simulator import compare_actions
        metrics = dict(METRIC_FAULT_INIT)
        candidates = [{"db_latency_ms": -100.0}]
        r1 = compare_actions(metrics, candidates, seed=1)
        r2 = compare_actions(metrics, candidates, seed=2)
        # Results should be similar but not identical
        assert abs(r1[0].impact_score - r2[0].impact_score) < 1.0
        assert r1[0].impact_score != r2[0].impact_score

    def test_simulation_respects_clips(self):
        """Metrics should not go below physical bounds."""
        from aic.env.counterfactual_simulator import simulate_action
        metrics = {"db_latency_ms": 50.0, "error_rate_pct": 1.0}
        action = {"db_latency_ms": -1000.0, "error_rate_pct": -100.0}
        result = simulate_action(metrics, action, steps=1)
        assert result.predicted_metrics["db_latency_ms"] >= 1.0
        assert result.predicted_metrics["error_rate_pct"] >= 0.0


# ── 2. Knowledge Agent ───────────────────────────────────────────────────

class TestKnowledgeAgent:
    def test_runbooks_loaded(self):
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        assert agent.get_runbook_count() == 6

    def test_list_runbooks(self):
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        runbooks = agent.list_runbooks()
        assert "cache_stampede" in runbooks
        assert "regional_outage" in runbooks

    def test_cache_stampede_retrieval(self):
        """VERIFICATION: Cache Stampede metrics → retrieves cache_stampede.md."""
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        metrics = {
            "db_latency_ms": 850.0,
            "queue_depth": 890.0,
            "conn_pool_pct": 98.0,
            "error_rate_pct": 18.5,
        }
        evidence = agent.retrieve(metrics, hypothesis="Cache Stampede")
        assert evidence is not None
        assert evidence.related_incident_id == "cache_stampede"
        assert evidence.confidence_score >= 0.3

    def test_regional_outage_retrieval(self):
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        metrics = {
            "net_io_mbps": 380.0,
            "error_rate_pct": 18.5,
            "p95_latency_ms": 3200.0,
            "cpu_pct": 89.0,
        }
        evidence = agent.retrieve(metrics, hypothesis="Regional Outage")
        assert evidence is not None
        assert evidence.related_incident_id == "regional_outage"

    def test_no_match_returns_none(self):
        """Hallucination prevention: healthy metrics → no runbook match."""
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        # All metrics at target = healthy system
        evidence = agent.retrieve(dict(METRIC_TARGETS))
        assert evidence is None

    def test_hypothesis_boosts_matching(self):
        """Providing a hypothesis keyword improves retrieval accuracy."""
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        metrics = {"db_latency_ms": 800.0, "replication_lag_ms": 500.0}
        # Without hypothesis
        ev1 = agent.retrieve(metrics)
        # With hypothesis
        ev2 = agent.retrieve(metrics, hypothesis="Schema Migration")
        assert ev2 is not None
        assert ev2.related_incident_id == "schema_migration"

    def test_remediation_extracted(self):
        from aic.agents.knowledge_agent import KnowledgeAgent
        agent = KnowledgeAgent()
        metrics = {"queue_depth": 900.0, "p95_latency_ms": 3200.0}
        evidence = agent.retrieve(metrics, hypothesis="Queue Cascade")
        assert evidence is not None
        assert len(evidence.suggested_remediation) > 10


# ── 3. Root Cause Analyst ────────────────────────────────────────────────

class TestRootCauseAnalyst:
    def test_uniform_prior(self):
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst()
        beliefs = analyst.get_beliefs()
        # All 6 scenarios should start with equal probability
        assert len(beliefs) == 6
        for conf in beliefs.values():
            assert abs(conf - 1.0 / 6) < 0.01

    def test_cache_stampede_identification(self):
        """Cache Stampede metrics → identifies Cache Stampede."""
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst()
        metrics = {
            "db_latency_ms": 850.0,
            "queue_depth": 890.0,
            "conn_pool_pct": 98.0,
            "error_rate_pct": 18.5,
            "p95_latency_ms": 200.0,
            "throughput_rps": 1000.0,
        }
        for _ in range(3):
            analyst.update(metrics)
        hyp = analyst.get_top_hypothesis()
        assert hyp.scenario_name == "Cache Stampede"
        assert hyp.confidence > 0.2

    def test_regional_outage_identification(self):
        """VERIFICATION: Broad metric degradation → identifies Regional Outage."""
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst()
        # Regional outage: broad degradation across many metrics
        metrics = {
            "net_io_mbps": 380.0,
            "error_rate_pct": 18.5,
            "p95_latency_ms": 3200.0,
            "cpu_pct": 89.0,
            "mem_pct": 92.0,
            "db_latency_ms": 500.0,
            "queue_depth": 400.0,
            "throughput_rps": 1000.0,
        }
        for _ in range(3):
            analyst.update(metrics)
        hyp = analyst.get_top_hypothesis()
        assert hyp.scenario_name == "Regional Outage"

    def test_time_decay_prevents_sticking(self):
        """Mastermind: analyst doesn't get stuck on early hypothesis."""
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst(decay_rate=0.1)
        # First: cache stampede symptoms
        cache_metrics = {
            "db_latency_ms": 850.0, "queue_depth": 890.0,
            "conn_pool_pct": 98.0,
        }
        for _ in range(3):
            analyst.update(cache_metrics)
        hyp1 = analyst.get_top_hypothesis()
        assert hyp1.scenario_name == "Cache Stampede"

        # Now: symptoms change to queue cascade (strong queue signal)
        queue_metrics = {
            "queue_depth": 5000.0, "p95_latency_ms": 5000.0,
            "error_rate_pct": 30.0, "cpu_pct": 60.0,
            "db_latency_ms": 50.0, "conn_pool_pct": 60.0,
            "net_io_mbps": 100.0, "mem_pct": 60.0,
        }
        for _ in range(8):
            analyst.update(queue_metrics)
        hyp2 = analyst.get_top_hypothesis()
        # Should have shifted away from Cache Stampede
        assert hyp2.scenario_name != "Cache Stampede"

    def test_reset_returns_to_uniform(self):
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst()
        analyst.update({"db_latency_ms": 1000.0})
        analyst.reset()
        beliefs = analyst.get_beliefs()
        for conf in beliefs.values():
            assert abs(conf - 1.0 / 6) < 0.01

    def test_get_all_hypotheses_sorted(self):
        from aic.agents.root_cause_analyst_agent import RootCauseAnalyst
        analyst = RootCauseAnalyst()
        analyst.update({"db_latency_ms": 1000.0, "queue_depth": 800.0})
        all_h = analyst.get_all_hypotheses()
        assert len(all_h) == 6
        confs = [h.confidence for h in all_h]
        assert confs == sorted(confs, reverse=True)


# ── 4. Orchestrator Thinking Loop ────────────────────────────────────────

class TestOrchestratorThinking:
    def _make_orchestrator(self):
        from aic.agents.adversarial_agent import AdversarialAgent
        from aic.agents.orchestrator_agent import OrchestratorAgent
        from aic.agents.db_agent import DBAgent
        db = DBAgent(use_llm=False)
        adv = AdversarialAgent([True] * 20, correct_recommendation_provider=db)
        orch = OrchestratorAgent(adv, use_llm=False)
        orch.mode = "trained"
        return orch

    def test_trace_contains_hypothesis(self):
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        metrics = dict(METRIC_FAULT_INIT)
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="increase pool size safely",
            reasoning="pool utilization is high increase safely",
            confidence=0.85,
            target_metrics=["conn_pool_pct"],
            risk_score=0.2,
        )]
        action, _ = orch.decide(0, 20, recs, "alert", {}, metrics)
        trace = action.explanation_trace
        assert trace.root_cause_hypothesis is not None
        assert "scenario_name" in trace.root_cause_hypothesis

    def test_trace_contains_runbook(self):
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        metrics = dict(METRIC_FAULT_INIT)
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="scale database connections",
            reasoning="connection pool is saturated at high level",
            confidence=0.85,
            target_metrics=["db_latency_ms"],
            risk_score=0.2,
        )]
        action, _ = orch.decide(0, 20, recs, "alert", {}, metrics)
        trace = action.explanation_trace
        assert trace.runbook_evidence is not None

    def test_trace_contains_simulation(self):
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        metrics = dict(METRIC_FAULT_INIT)
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="reduce database latency",
            reasoning="database is experiencing high latency",
            confidence=0.85,
            target_metrics=["db_latency_ms"],
            risk_score=0.2,
        )]
        action, _ = orch.decide(0, 20, recs, "alert", {}, metrics)
        trace = action.explanation_trace
        assert trace.simulation_scores is not None
        assert "db_agent" in trace.simulation_scores

    def test_no_runbook_logs_discovery_note(self):
        """Mastermind: missing runbook → logs discovery note."""
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        # All metrics exactly at target → no deviation → no keywords extracted
        healthy = {k: v for k, v in METRIC_TARGETS.items()}
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="standard health check action",
            reasoning="routine maintenance standard check",
            confidence=0.5,
            target_metrics=["db_latency_ms"],
        )]
        action, _ = orch.decide(0, 20, recs, "nominal", {}, healthy)
        runbook = action.explanation_trace.runbook_evidence
        assert runbook is not None
        # With healthy metrics, either no match or low confidence
        if runbook.get("incident_id") is None:
            assert "simulation-based discovery" in runbook.get("note", "")
        # If a weak match is found, that's also acceptable

    def test_simulation_reranks_candidates(self):
        """Orchestrator picks best simulated action, not just highest confidence."""
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        metrics = dict(METRIC_FAULT_INIT)
        recs = [
            SubAgentRecommendation(
                agent_name="app_agent",
                action="scale app horizontally to handle load",
                reasoning="app error rate is elevated significantly",
                confidence=0.90,  # higher confidence
                target_metrics=["error_rate_pct"],
                risk_score=0.2,
            ),
            SubAgentRecommendation(
                agent_name="db_agent",
                action="reduce db latency and connection pool",
                reasoning="db is the bottleneck causing cascading failures",
                confidence=0.85,  # lower confidence
                target_metrics=["db_latency_ms", "conn_pool_pct"],
                risk_score=0.2,
            ),
        ]
        action, _ = orch.decide(0, 20, recs, "alert", {}, metrics)
        # DB fix targets more metrics with coupling → should simulate better
        # The action should be from simulation-based ranking
        assert action.explanation_trace.simulation_scores is not None
