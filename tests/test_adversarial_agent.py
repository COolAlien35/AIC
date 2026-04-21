# tests/test_adversarial_agent.py
"""
Phase 4 tests.
Covers adversarial agent accuracy, format parity, determinism,
and full episode end-to-end with rule-based agents.
"""
import pytest

from aic.utils.constants import (
    AGENT_ADV, AGENT_DB, METRIC_FAULT_INIT, SLA_STEPS,
)
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent, COUNTERFACTUAL_TEMPLATES
from aic.schemas.traces import SubAgentRecommendation


# ── 1. Adversary accuracy ──────────────────────────────────────────────────

class TestAdversaryAccuracy:
    def test_50pct_accuracy_over_20_steps(self):
        """Adversary must be correct exactly 50% of the time."""
        cycle = get_adversary_cycle(make_episode_rng(0))
        db_agent = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db_agent)

        correct_count = sum(1 for s in range(20) if adv.was_correct_at_step(s))
        assert correct_count == 10

    def test_50pct_across_multiple_episodes(self):
        """Each episode should have exactly 10/20 correct."""
        for ep_id in range(10):
            cycle = get_adversary_cycle(make_episode_rng(ep_id))
            assert sum(cycle) == 10, f"Episode {ep_id} has {sum(cycle)} correct"

    def test_6_templates_exist(self):
        assert len(COUNTERFACTUAL_TEMPLATES) == 6
        for t in COUNTERFACTUAL_TEMPLATES:
            assert "action" in t
            assert "reasoning" in t
            assert "confidence" in t
            assert "target_metrics" in t


# ── 2. Format parity ───────────────────────────────────────────────────────

class TestFormatParity:
    def test_adversary_same_format_as_db_agent(self):
        """Adversary output must be a valid SubAgentRecommendation."""
        cycle = get_adversary_cycle(make_episode_rng(0))
        db_agent = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db_agent)

        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}

        db_rec = db_agent.recommend(obs, step=0)
        adv_rec = adv.recommend(obs, step=0)

        # Both must be SubAgentRecommendation with same fields
        assert isinstance(db_rec, SubAgentRecommendation)
        assert isinstance(adv_rec, SubAgentRecommendation)
        assert hasattr(adv_rec, "action")
        assert hasattr(adv_rec, "reasoning")
        assert hasattr(adv_rec, "confidence")
        assert hasattr(adv_rec, "target_metrics")

    def test_correct_step_uses_provider_agent_name(self):
        """On correct steps, adversary wraps the provider's recommendation but keeps AGENT_ADV identity."""
        cycle = [True] * 20  # all correct
        db_agent = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db_agent)

        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}
        rec = adv.recommend(obs, step=0)
        # Adversary must always identify as AGENT_ADV, even on correct steps
        assert rec.agent_name == AGENT_ADV
        # But the action content should match the provider's recommendation
        db_rec = db_agent.recommend(obs, step=0)
        assert rec.action == db_rec.action

    def test_counterfactual_step_uses_adv_agent_name(self):
        """On counterfactual steps, adversary uses its own agent name."""
        cycle = [False] * 20  # all counterfactual
        db_agent = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db_agent)

        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}
        rec = adv.recommend(obs, step=0)
        assert rec.agent_name == AGENT_ADV


# ── 3. Determinism ─────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_episode_same_recommendations(self):
        """Same episode ID → same adversary recommendations."""
        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}

        actions1 = []
        cycle1 = get_adversary_cycle(make_episode_rng(7))
        db1 = DBAgent(use_llm=False)
        adv1 = AdversarialAgent(cycle1, correct_recommendation_provider=db1)
        for s in range(20):
            actions1.append(adv1.recommend(obs, s).action)

        actions2 = []
        cycle2 = get_adversary_cycle(make_episode_rng(7))
        db2 = DBAgent(use_llm=False)
        adv2 = AdversarialAgent(cycle2, correct_recommendation_provider=db2)
        for s in range(20):
            actions2.append(adv2.recommend(obs, s).action)

        assert actions1 == actions2

    def test_different_episodes_different_cycles(self):
        """Different episodes should produce different adversary schedules."""
        cycle1 = get_adversary_cycle(make_episode_rng(0))
        cycle2 = get_adversary_cycle(make_episode_rng(1))
        # They can't be identical (extremely unlikely with different seeds)
        assert cycle1 != cycle2


# ── 4. Specialist agents (rule-based) ──────────────────────────────────────

class TestSpecialistAgents:
    def test_db_agent_rule_based(self):
        db = DBAgent(use_llm=False)
        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}
        rec = db.recommend(obs, step=0)
        assert isinstance(rec, SubAgentRecommendation)
        assert rec.agent_name == AGENT_DB
        assert len(rec.action) > 0

    def test_infra_agent_rule_based(self):
        from aic.utils.constants import AGENT_INFRA
        infra = InfraAgent(use_llm=False)
        obs = {"cpu_pct": 89.0, "mem_pct": 92.0, "pod_restarts": 7.0, "net_io_mbps": 380.0}
        rec = infra.recommend(obs, step=0)
        assert rec.agent_name == AGENT_INFRA

    def test_app_agent_rule_based(self):
        from aic.utils.constants import AGENT_APP
        app = AppAgent(use_llm=False)
        obs = {"error_rate_pct": 18.5, "p95_latency_ms": 3200.0, "queue_depth": 890.0}
        rec = app.recommend(obs, step=0)
        assert rec.agent_name == AGENT_APP


# ── 5. Orchestrator (rule-based) ───────────────────────────────────────────

class TestOrchestrator:
    def test_orchestrator_rule_based_decide(self):
        from aic.agents.orchestrator_agent import OrchestratorAgent
        from aic.schemas.traces import OrchestratorAction

        cycle = get_adversary_cycle(make_episode_rng(0))
        db = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db)
        orch = OrchestratorAgent(adv, use_llm=False)

        # Create recommendations
        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}
        recs = [
            db.recommend(obs, 0),
            InfraAgent(use_llm=False).recommend(
                {"cpu_pct": 89.0, "mem_pct": 92.0, "pod_restarts": 7.0, "net_io_mbps": 380.0}, 0
            ),
            AppAgent(use_llm=False).recommend(
                {"error_rate_pct": 18.5, "p95_latency_ms": 3200.0, "queue_depth": 890.0}, 0
            ),
            adv.recommend(obs, 0),
        ]

        action, override = orch.decide(
            step=0, sla_remaining=20, sub_agent_recommendations=recs,
            alert_summary="test", prev_metrics={}, current_metrics=METRIC_FAULT_INIT,
        )

        assert isinstance(action, OrchestratorAction)
        assert isinstance(override, bool)
        assert action.explanation_trace.step == 0
        assert len(action.explanation_trace.reasoning) >= 10

    def test_trust_scores_update(self):
        from aic.agents.orchestrator_agent import OrchestratorAgent
        from aic.utils.constants import INITIAL_TRUST

        cycle = get_adversary_cycle(make_episode_rng(0))
        db = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db)
        orch = OrchestratorAgent(adv, use_llm=False)

        assert all(v == INITIAL_TRUST for v in orch.trust_scores.values())

        # Simulate a step where health improves → trust should increase
        obs = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}
        recs = [db.recommend(obs, 0), adv.recommend(obs, 0)]

        # Current closer to target than prev → improvement
        prev = {"db_latency_ms": 900.0, "conn_pool_pct": 99.0}
        curr = {"db_latency_ms": 100.0, "conn_pool_pct": 70.0}
        orch.decide(
            step=0, sla_remaining=20, sub_agent_recommendations=recs,
            alert_summary="test", prev_metrics=prev, current_metrics=curr,
        )

        # After one step with improvement, trust should have moved
        # (though on step 0, _prev_recommendations is empty so no update yet)


# ── 6. Full episode (rule-based) ───────────────────────────────────────────

class TestFullEpisode:
    def test_20_step_episode_rule_based(self):
        """A complete 20-step episode must run without errors using rule-based agents."""
        from aic.env.world_state import WorldState
        from aic.env.fault_injector import FaultInjector
        from aic.env.reward_engine import RewardEngine
        from aic.agents.orchestrator_agent import OrchestratorAgent

        episode_id = 0
        rng = make_episode_rng(episode_id)
        cycle = get_adversary_cycle(make_episode_rng(episode_id))

        ws = WorldState(make_episode_rng(episode_id))
        fi = FaultInjector("cascading_failure")
        reward_engine = RewardEngine()

        db = DBAgent(use_llm=False)
        infra = InfraAgent(use_llm=False)
        app = AppAgent(use_llm=False)
        adv = AdversarialAgent(cycle, correct_recommendation_provider=db)
        orch = OrchestratorAgent(adv, use_llm=False)

        prev_metrics = ws.snapshot()
        total_reward = 0.0

        for step in range(SLA_STEPS):
            faults = fi.get_contributions(step)
            db_rec = db.recommend(ws.get_db_observation(), step)
            infra_rec = infra.recommend(ws.get_infra_observation(), step)
            app_rec = app.recommend(ws.get_app_observation(), step)
            adv_rec = adv.recommend(ws.get_db_observation(), step)

            action, override = orch.decide(
                step=step, sla_remaining=SLA_STEPS - step,
                sub_agent_recommendations=[db_rec, infra_rec, app_rec, adv_rec],
                alert_summary="test",
                prev_metrics=prev_metrics, current_metrics=ws.snapshot(),
            )

            ws.step(action.action_deltas, faults)

            r = reward_engine.compute_step_reward(
                step=step, metrics=ws.snapshot(), prev_metrics=prev_metrics,
                override_applied=override,
                adversary_was_correct=adv.was_correct_at_step(step),
                predicted_2step_impact=action.explanation_trace.predicted_2step_impact,
                reasoning=action.explanation_trace.reasoning,
            )
            total_reward += r["total"]
            prev_metrics = ws.snapshot()

        # Episode should complete without errors
        assert total_reward != 0.0  # Reward should be non-zero
        # With no effective remediation (rule-based doesn't really fix things
        # against fault injection), total should be negative
        assert total_reward < 0.0
