# tests/test_safety_tier.py
"""
Phase 9 tests for Specialist Agents, Recovery Verifier, and Safety Tier.

Verification criteria:
1. Safety: risk_score=0.9 → verifier returns approved=False
2. Cascade: All high-risk recs → orchestrator defaults to "Wait and Observe"
3. Network: Regional Outage metrics → NetworkAgent suggests DNS Failover
4. Security: High auth failures → SecurityAgent suggests token revocation
5. Trace: ExplanationTrace contains verifier_report
"""
import pytest


# ── 1. Schema Upgrade ─────────────────────────────────────────────────────

class TestSchemaUpgrade:
    def test_recommendation_has_new_fields(self):
        from aic.schemas.traces import SubAgentRecommendation
        rec = SubAgentRecommendation(
            agent_name="test",
            action="test action",
            reasoning="test reasoning text",
            confidence=0.8,
            target_metrics=["cpu_pct"],
        )
        # New fields should have defaults
        assert rec.risk_score == 0.0
        assert rec.blast_radius == "low"
        assert rec.rollback_plan == ""
        assert rec.expected_impact == {}

    def test_recommendation_accepts_new_fields(self):
        from aic.schemas.traces import SubAgentRecommendation
        rec = SubAgentRecommendation(
            agent_name="test",
            action="test action",
            reasoning="test reasoning text",
            confidence=0.8,
            target_metrics=["cpu_pct"],
            risk_score=0.9,
            blast_radius="high",
            rollback_plan="revert the change",
            expected_impact={"cpu_pct": -10.0},
        )
        assert rec.risk_score == 0.9
        assert rec.blast_radius == "high"

    def test_explanation_trace_has_verifier_report(self):
        from aic.schemas.traces import ExplanationTrace
        trace = ExplanationTrace(
            step=0,
            action_taken="test action",
            reasoning="detailed test reasoning for the action taken",
            sub_agent_trust_scores={"db_agent": 0.5},
            override_applied=False,
            predicted_2step_impact={"cpu_pct": -5.0},
            schema_drift_detected=False,
            verifier_report={"approved": True, "risk_score": 0.3},
        )
        assert trace.verifier_report["approved"] is True

    def test_explanation_trace_verifier_defaults_none(self):
        from aic.schemas.traces import ExplanationTrace
        trace = ExplanationTrace(
            step=0,
            action_taken="test action",
            reasoning="detailed test reasoning for the action taken",
            sub_agent_trust_scores={"db_agent": 0.5},
            override_applied=False,
            predicted_2step_impact={},
            schema_drift_detected=False,
        )
        assert trace.verifier_report is None


# ── 2. Recovery Verifier ──────────────────────────────────────────────────

class TestRecoveryVerifier:
    def test_high_risk_vetoed(self):
        """VERIFICATION: risk_score=0.9 → approved=False."""
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        from aic.schemas.traces import SubAgentRecommendation
        verifier = RecoveryVerifierAgent()
        rec = SubAgentRecommendation(
            agent_name="test",
            action="dangerous action here",
            reasoning="this is a risky action",
            confidence=0.9,
            target_metrics=["cpu_pct"],
            risk_score=0.9,
            blast_radius="medium",
            rollback_plan="revert everything",
        )
        report = verifier.verify(rec)
        assert report.approved is False
        assert "exceeds threshold" in report.verification_reasoning

    def test_high_blast_no_rollback_vetoed(self):
        """blast_radius=high with empty rollback → vetoed."""
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        from aic.schemas.traces import SubAgentRecommendation
        verifier = RecoveryVerifierAgent()
        rec = SubAgentRecommendation(
            agent_name="test",
            action="isolate entire cluster",
            reasoning="security threat detected now",
            confidence=0.85,
            target_metrics=["auth_failure_rate"],
            risk_score=0.7,
            blast_radius="high",
            rollback_plan="",  # No rollback!
        )
        report = verifier.verify(rec)
        assert report.approved is False
        assert "rollback" in report.verification_reasoning.lower()

    def test_safe_action_approved(self):
        """Low risk, low blast → approved."""
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        from aic.schemas.traces import SubAgentRecommendation
        verifier = RecoveryVerifierAgent()
        rec = SubAgentRecommendation(
            agent_name="test",
            action="increase connection pool",
            reasoning="pool utilization is high",
            confidence=0.8,
            target_metrics=["conn_pool_pct"],
            risk_score=0.3,
            blast_radius="low",
            rollback_plan="decrease pool size",
        )
        report = verifier.verify(rec)
        assert report.approved is True

    def test_high_blast_with_rollback_approved(self):
        """blast_radius=high WITH rollback_plan → approved (if risk < 0.8)."""
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        from aic.schemas.traces import SubAgentRecommendation
        verifier = RecoveryVerifierAgent()
        rec = SubAgentRecommendation(
            agent_name="test",
            action="reroute all traffic",
            reasoning="regional outage detected",
            confidence=0.85,
            target_metrics=["regional_latency_ms"],
            risk_score=0.6,
            blast_radius="high",
            rollback_plan="restore original routing weights",
        )
        report = verifier.verify(rec)
        assert report.approved is True

    def test_veto_log_tracks_vetoes(self):
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        from aic.schemas.traces import SubAgentRecommendation
        verifier = RecoveryVerifierAgent()
        rec = SubAgentRecommendation(
            agent_name="bad_agent",
            action="destroy everything now",
            reasoning="total system reset needed",
            confidence=0.95,
            target_metrics=["cpu_pct"],
            risk_score=0.95,
        )
        verifier.verify(rec)
        assert len(verifier.get_veto_log()) == 1

    def test_safe_minimal_action(self):
        from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
        verifier = RecoveryVerifierAgent()
        safe = verifier.get_safe_minimal_action()
        assert "Wait and Observe" in safe.action
        assert safe.risk_score == 0.0


# ── 3. Network Agent ─────────────────────────────────────────────────────

class TestNetworkAgent:
    def test_dns_failover_on_high_dns_latency(self):
        """VERIFICATION: High DNS latency → suggests DNS Failover."""
        from aic.agents.network_agent import NetworkAgent
        agent = NetworkAgent(use_llm=False)
        obs = {
            "packet_loss_pct": 2.0,
            "dns_latency_ms": 120.0,
            "lb_5xx_count": 5,
            "regional_latency_ms": 40.0,
        }
        rec = agent.recommend(obs, step=0)
        assert "DNS" in rec.action or "dns" in rec.action.lower()
        assert rec.agent_name == "network_agent"
        assert rec.risk_score >= 0.0
        assert rec.rollback_plan != ""

    def test_drain_lb_on_high_5xx(self):
        from aic.agents.network_agent import NetworkAgent
        agent = NetworkAgent(use_llm=False)
        obs = {
            "packet_loss_pct": 8.0,
            "dns_latency_ms": 10.0,
            "lb_5xx_count": 50,
            "regional_latency_ms": 40.0,
        }
        rec = agent.recommend(obs, step=0)
        assert "drain" in rec.action.lower() or "load balancer" in rec.action.lower()

    def test_reroute_on_regional_latency(self):
        from aic.agents.network_agent import NetworkAgent
        agent = NetworkAgent(use_llm=False)
        obs = {
            "packet_loss_pct": 1.0,
            "dns_latency_ms": 8.0,
            "lb_5xx_count": 2,
            "regional_latency_ms": 250.0,
        }
        rec = agent.recommend(obs, step=0)
        assert "reroute" in rec.action.lower() or "availability zone" in rec.action.lower()

    def test_new_fields_populated(self):
        from aic.agents.network_agent import NetworkAgent
        agent = NetworkAgent(use_llm=False)
        obs = {"dns_latency_ms": 100.0}
        rec = agent.recommend(obs, step=0)
        assert rec.blast_radius in ("low", "medium", "high")
        assert 0.0 <= rec.risk_score <= 1.0
        assert len(rec.expected_impact) > 0


# ── 4. Security Agent ────────────────────────────────────────────────────

class TestSecurityAgent:
    def test_revoke_tokens_on_suspicious(self):
        from aic.agents.security_agent import SecurityAgent
        agent = SecurityAgent(use_llm=False)
        obs = {
            "auth_failure_rate": 15.0,
            "suspicious_token_count": 8,
            "compromised_ip_count": 0,
        }
        rec = agent.recommend(obs, step=0)
        assert "revoke" in rec.action.lower() or "token" in rec.action.lower()
        assert rec.agent_name == "security_agent"

    def test_isolate_on_compromised_ips(self):
        from aic.agents.security_agent import SecurityAgent
        agent = SecurityAgent(use_llm=False)
        obs = {
            "auth_failure_rate": 20.0,
            "suspicious_token_count": 3,
            "compromised_ip_count": 5,
        }
        rec = agent.recommend(obs, step=0)
        assert "isolate" in rec.action.lower()
        assert rec.risk_score > 0.5  # Security isolation is high-risk
        assert rec.blast_radius == "high"

    def test_degraded_safe_on_low_threat(self):
        from aic.agents.security_agent import SecurityAgent
        agent = SecurityAgent(use_llm=False)
        obs = {
            "auth_failure_rate": 5.0,
            "suspicious_token_count": 0,
            "compromised_ip_count": 0,
        }
        rec = agent.recommend(obs, step=0)
        assert "degraded" in rec.action.lower() or "safe" in rec.action.lower()

    def test_security_has_rollback(self):
        from aic.agents.security_agent import SecurityAgent
        agent = SecurityAgent(use_llm=False)
        obs = {"compromised_ip_count": 10}
        rec = agent.recommend(obs, step=0)
        assert rec.rollback_plan != ""


# ── 5. Orchestrator Verifier Integration ─────────────────────────────────

class TestOrchestratorVerifierIntegration:
    def _make_orchestrator(self):
        from aic.agents.adversarial_agent import AdversarialAgent
        from aic.agents.orchestrator_agent import OrchestratorAgent
        from aic.agents.db_agent import DBAgent
        db = DBAgent(use_llm=False)
        adv = AdversarialAgent([True] * 20, correct_recommendation_provider=db)
        orch = OrchestratorAgent(adv, use_llm=False)
        orch.mode = "trained"
        return orch

    def test_safe_action_passes_verifier(self):
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="increase pool size safely",
            reasoning="pool utilization is high, increase safely",
            confidence=0.85,
            target_metrics=["conn_pool_pct"],
            risk_score=0.2,
            blast_radius="low",
            rollback_plan="decrease pool size",
        )]
        action, _ = orch.decide(0, 20, recs, "alert", {}, {})
        assert action.explanation_trace.verifier_report is not None
        assert action.explanation_trace.verifier_report["approved"] is True

    def test_veto_cascade_to_safe_action(self):
        """All high-risk recs → orchestrator falls back to Wait and Observe."""
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        # All recommendations are too risky
        recs = [
            SubAgentRecommendation(
                agent_name="security_agent",
                action="isolate everything now",
                reasoning="total security lockdown needed urgently",
                confidence=0.95,
                target_metrics=["auth_failure_rate"],
                risk_score=0.95,
                blast_radius="high",
                rollback_plan="",
            ),
            SubAgentRecommendation(
                agent_name="network_agent",
                action="kill all network connections",
                reasoning="network compromised need full reset",
                confidence=0.90,
                target_metrics=["packet_loss_pct"],
                risk_score=0.9,
                blast_radius="high",
                rollback_plan="",
            ),
            SubAgentRecommendation(
                agent_name="db_agent",
                action="drop and recreate all tables",
                reasoning="corruption detected rebuild needed",
                confidence=0.85,
                target_metrics=["db_latency_ms"],
                risk_score=0.85,
                blast_radius="high",
                rollback_plan="",
            ),
        ]
        action, _ = orch.decide(0, 20, recs, "alert", {}, {})
        assert "Wait and Observe" in action.action_description
        assert len(orch._vetoed_actions) == 3

    def test_verifier_report_in_trace(self):
        """VERIFICATION: ExplanationTrace contains verifier_report."""
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        recs = [SubAgentRecommendation(
            agent_name="db_agent",
            action="standard pool increase",
            reasoning="pool utilization is moderately high",
            confidence=0.80,
            target_metrics=["conn_pool_pct"],
            risk_score=0.3,
        )]
        action, _ = orch.decide(0, 20, recs, "alert", {}, {})
        trace = action.explanation_trace
        assert trace.verifier_report is not None
        assert "risk_score" in trace.verifier_report

    def test_veto_picks_second_best(self):
        """First rec vetoed, second approved → orchestrator picks second."""
        from aic.schemas.traces import SubAgentRecommendation
        orch = self._make_orchestrator()
        recs = [
            SubAgentRecommendation(
                agent_name="security_agent",
                action="dangerous lockdown action",
                reasoning="extreme security measure needed now",
                confidence=0.95,
                target_metrics=["auth_failure_rate"],
                risk_score=0.95,  # Will be vetoed
            ),
            SubAgentRecommendation(
                agent_name="db_agent",
                action="safe pool increase action",
                reasoning="safe standard remediation action",
                confidence=0.80,
                target_metrics=["conn_pool_pct"],
                risk_score=0.2,  # Will pass
            ),
        ]
        action, _ = orch.decide(0, 20, recs, "alert", {}, {})
        assert "safe pool" in action.action_description.lower()
        assert len(orch._vetoed_actions) == 1
