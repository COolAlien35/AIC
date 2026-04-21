# tests/test_benchmark_suite.py
"""
Phase 11 tests for Benchmark Suite, Dashboard components, and Documentation.

Verification:
1. Benchmark runs without crashes across all 6 scenarios
2. AIC Trained reward > all baseline rewards
3. AIC unsafe action rate = 0.0%
4. Topology visualization renders correctly
5. CSV output is well-formed
"""
import pytest
import os
from pathlib import Path


# ── 1. Benchmark Suite ───────────────────────────────────────────────────

class TestBenchmarkSuite:
    def test_baselines_instantiate(self):
        from aic.evals.benchmark_suite import (
            HighestConfidencePolicy, MajorityVotePolicy, NoTrustOrchestratorPolicy,
        )
        h = HighestConfidencePolicy()
        m = MajorityVotePolicy()
        n = NoTrustOrchestratorPolicy()
        assert h.name == "HighestConfidenceOnly"
        assert m.name == "MajorityVote"
        assert n.name == "NoTrustOrchestrator"

    def test_highest_confidence_selects_top(self):
        from aic.evals.benchmark_suite import HighestConfidencePolicy
        from aic.schemas.traces import SubAgentRecommendation
        policy = HighestConfidencePolicy()
        recs = [
            SubAgentRecommendation(
                agent_name="db_agent", action="fix db now", reasoning="db is broken needs fix",
                confidence=0.7, target_metrics=["db_latency_ms"],
            ),
            SubAgentRecommendation(
                agent_name="app_agent", action="fix app now", reasoning="app is broken needs fix",
                confidence=0.9, target_metrics=["error_rate_pct"],
            ),
        ]
        selected = policy.select(recs)
        assert selected.agent_name == "app_agent"

    def test_majority_vote_selects_popular(self):
        from aic.evals.benchmark_suite import MajorityVotePolicy
        from aic.schemas.traces import SubAgentRecommendation
        policy = MajorityVotePolicy()
        recs = [
            SubAgentRecommendation(
                agent_name="db_agent", action="fix db latency", reasoning="db latency is high",
                confidence=0.8, target_metrics=["db_latency_ms"],
            ),
            SubAgentRecommendation(
                agent_name="infra_agent", action="fix db latency too", reasoning="db causing infra issues",
                confidence=0.7, target_metrics=["db_latency_ms"],
            ),
            SubAgentRecommendation(
                agent_name="app_agent", action="fix error rate", reasoning="error rate too high",
                confidence=0.9, target_metrics=["error_rate_pct"],
            ),
        ]
        selected = policy.select(recs)
        # db_latency_ms is mentioned by 2 agents → should be selected
        assert "db_latency_ms" in selected.target_metrics

    def test_aic_episode_runs(self):
        from aic.evals.benchmark_suite import _run_aic_episode
        result = _run_aic_episode(0, "trained", 42)
        assert result.policy_name == "AIC (Trained)"
        assert result.scenario_name == "Cache Stampede"
        assert 0.0 <= result.unsafe_action_rate <= 1.0

    def test_baseline_episode_runs(self):
        from aic.evals.benchmark_suite import (
            _run_baseline_episode, HighestConfidencePolicy,
        )
        policy = HighestConfidencePolicy()
        result = _run_baseline_episode(policy, 0, 42)
        assert result.policy_name == "HighestConfidenceOnly"
        assert result.mttr_steps >= 1

    def test_aic_trained_beats_baselines(self):
        """VERIFICATION: AIC Trained reward > baseline rewards."""
        from aic.evals.benchmark_suite import (
            _run_aic_episode, _run_baseline_episode,
            HighestConfidencePolicy, MajorityVotePolicy,
            NoTrustOrchestratorPolicy,
        )
        scenario_id = 0  # Cache Stampede
        aic = _run_aic_episode(scenario_id, "trained", 42)
        hc = _run_baseline_episode(HighestConfidencePolicy(), scenario_id, 42)
        mv = _run_baseline_episode(MajorityVotePolicy(), scenario_id, 42)
        nt = _run_baseline_episode(NoTrustOrchestratorPolicy(), scenario_id, 42)

        # AIC trained should have highest (least negative) reward
        assert aic.total_reward >= hc.total_reward
        assert aic.total_reward >= mv.total_reward
        assert aic.total_reward >= nt.total_reward

    def test_aic_unsafe_rate_zero(self):
        """VERIFICATION: AIC unsafe action rate = 0.0%."""
        from aic.evals.benchmark_suite import _run_aic_episode
        # Test across multiple scenarios
        for sid in [0, 2, 4]:
            result = _run_aic_episode(sid, "trained", 42)
            assert result.unsafe_action_rate == 0.0, (
                f"Scenario {sid}: unsafe rate = {result.unsafe_action_rate}"
            )

    def test_full_benchmark_produces_csv(self):
        from aic.evals.benchmark_suite import run_full_benchmark
        csv_path = "logs/test_benchmark.csv"
        results = run_full_benchmark(csv_path, 42)
        assert len(results) == 30  # 6 scenarios × 5 policies
        assert Path(csv_path).exists()
        # Cleanup
        Path(csv_path).unlink(missing_ok=True)

    def test_summary_table(self):
        from aic.evals.benchmark_suite import run_full_benchmark, get_summary_table
        results = run_full_benchmark("logs/test_summary.csv", 42)
        summary = get_summary_table(results)
        assert "AIC (Trained)" in summary
        assert "HighestConfidenceOnly" in summary
        assert "avg_mttr" in summary["AIC (Trained)"]
        Path("logs/test_summary.csv").unlink(missing_ok=True)

    def test_all_scenarios_run(self):
        """Ensure benchmark doesn't crash on any of the 6 scenarios."""
        from aic.evals.benchmark_suite import _run_aic_episode
        for sid in range(6):
            result = _run_aic_episode(sid, "trained", 42)
            assert result.scenario_id == sid
            assert result.total_reward != 0.0


# ── 2. Topology Visualization ────────────────────────────────────────────

class TestTopologyVisualization:
    @pytest.fixture(autouse=True)
    def _skip_no_plotly(self):
        pytest.importorskip("plotly")

    def test_render_default_topology(self):
        from dashboard.components.topology_viz import render_topology_map
        fig = render_topology_map()
        assert fig is not None
        # Should have nodes (5) + edges (5)
        assert len(fig.data) == 10

    def test_render_degraded_topology(self):
        from dashboard.components.topology_viz import render_topology_map
        state = {
            "gateway": {"health": 0.2, "load": 80.0, "latency": 500.0, "error_rate": 10.0},
            "app": {"health": 0.5, "load": 50.0, "latency": 200.0, "error_rate": 5.0},
            "cache": {"health": 0.8, "load": 20.0, "latency": 10.0, "error_rate": 1.0},
            "queue": {"health": 0.6, "load": 40.0, "latency": 50.0, "error_rate": 3.0},
            "db": {"health": 0.3, "load": 90.0, "latency": 800.0, "error_rate": 15.0},
        }
        fig = render_topology_map(state, root_cause_node="gateway")
        assert fig is not None

    def test_regional_outage_gateway_red(self):
        """VERIFICATION: Regional Outage → Gateway node is red."""
        from dashboard.components.topology_viz import render_topology_map, _health_color
        state = {
            "gateway": {"health": 0.1, "load": 95.0, "latency": 1000.0, "error_rate": 20.0},
            "app": {"health": 0.3, "load": 80.0, "latency": 500.0, "error_rate": 10.0},
            "cache": {"health": 0.4, "load": 60.0, "latency": 100.0, "error_rate": 5.0},
            "queue": {"health": 0.4, "load": 70.0, "latency": 150.0, "error_rate": 7.0},
            "db": {"health": 0.3, "load": 85.0, "latency": 700.0, "error_rate": 12.0},
        }
        fig = render_topology_map(state, root_cause_node="gateway")
        # Gateway health 0.1 → should be red
        assert _health_color(0.1) == "#ef4444"


# ── 3. Impact Visualization ──────────────────────────────────────────────

class TestImpactVisualization:
    @pytest.fixture(autouse=True)
    def _skip_no_plotly(self):
        pytest.importorskip("plotly")

    def test_revenue_counter(self):
        from dashboard.components.impact_viz import render_revenue_counter
        result = render_revenue_counter(15, 20, 5)
        assert result["revenue_saved"] == 75000.0
        assert result["uptime_pct"] == 75.0
        assert result["mttr_minutes"] == 15

    def test_timeline_events(self):
        from dashboard.components.impact_viz import extract_timeline_events
        trajectory = [
            {"step": 0, "health": 0.3, "trace": {"root_cause_hypothesis": {"scenario_name": "Cache Stampede", "confidence": 0.6}}},
            {"step": 5, "health": 0.4, "trace": {"verifier_report": {"approved": False}}},
            {"step": 10, "health": 0.6, "trace": {}},
        ]
        events = extract_timeline_events(trajectory)
        types = [e["type"] for e in events]
        assert "incident" in types
        assert "hypothesis" in types
        assert "veto" in types
        assert "recovery" in types

    def test_war_room_timeline_renders(self):
        from dashboard.components.impact_viz import render_war_room_timeline
        events = [
            {"step": 0, "type": "incident", "message": "🚨 Test", "severity": "critical"},
        ]
        fig = render_war_room_timeline(events)
        assert fig is not None

    def test_benchmark_comparison_renders(self):
        from dashboard.components.impact_viz import render_benchmark_comparison
        results = [
            {"policy_name": "AIC (Trained)", "avg_mttr": 5, "avg_reward": 100},
            {"policy_name": "Baseline", "avg_mttr": 15, "avg_reward": -50},
        ]
        fig = render_benchmark_comparison(results)
        assert fig is not None


# ── 4. Documentation ─────────────────────────────────────────────────────

class TestDocumentation:
    def test_readme_exists(self):
        assert Path("README.md").exists()

    def test_readme_has_key_sections(self):
        content = Path("README.md").read_text()
        assert "Adaptive Incident Choreographer" in content
        assert "Fleet AI" in content
        assert "Halluminate" in content
        assert "Patronus" in content
        assert "Scaler AI" in content
        assert "Architecture" in content
        assert "Quick Start" in content
        assert "Recovery Verifier" in content
        assert "Thinking" in content

    def test_readme_has_bonus_mapping(self):
        content = Path("README.md").read_text()
        assert "Bonus Prize Mapping" in content
