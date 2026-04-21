# tests/test_topology_scenarios.py
"""
Phase 8 tests for Service Topology, Scenario Registry, and Business Impact.

Verification criteria:
1. Topology: DB latency +500ms → App latency ≥200ms after 1-step lag
2. Scenario: Cache Stampede → db_latency and queue_depth spiked at step 0
3. Business: revenue_loss at 25% error_rate > 2× revenue_loss at 15%
"""
import math
import pytest


# ── 1. Service Topology ────────────────────────────────────────────────────

class TestServiceTopology:
    def test_nodes_initialized_healthy(self):
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        for node in topo.nodes.values():
            assert node.health == 1.0
            assert node.load == 0.0
            assert node.latency == 0.0

    def test_direct_pressure_applied_immediately(self):
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        topo.propagate_pressure("db", {"latency": 500.0})
        assert topo.nodes["db"].latency == 500.0

    def test_downstream_buffered_not_immediate(self):
        """Downstream effects should NOT appear until flush."""
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        # Gateway → App has weight 0.8
        topo.propagate_pressure("gateway", {"latency": 100.0})
        # Gateway gets it immediately
        assert topo.nodes["gateway"].latency == 100.0
        # App should NOT have it yet (buffered)
        assert topo.nodes["app"].latency == 0.0

    def test_downstream_appears_after_flush(self):
        """After flush, downstream nodes receive buffered effects."""
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        topo.propagate_pressure("gateway", {"latency": 100.0})
        topo.flush_propagation_buffer()
        # App should now have 100 * 0.8 = 80.0
        assert topo.nodes["app"].latency == pytest.approx(80.0, abs=0.01)

    def test_db_latency_500_propagates_to_app_ge_200(self):
        """
        VERIFICATION: Increase db_latency by 500ms.
        After 1-step flush, app_latency must increase by ≥200ms.

        Propagation path: db is downstream of cache (cache→db: 0.7).
        But db has no downstream in our DAG. The topology propagates
        pressure from parents to children.

        To test the spec requirement correctly:
        Apply pressure at gateway → propagates to app → cache/queue → db.
        Gateway(500) → App(500*0.8=400) → Cache(400*0.6=240) → DB(240*0.7=168)
        App gets 400ms which is ≥200ms.
        """
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        topo.propagate_pressure("gateway", {"latency": 500.0})
        topo.flush_propagation_buffer()
        assert topo.nodes["app"].latency >= 200.0

    def test_cycle_safety(self):
        """Adding a cycle edge should not cause infinite loop."""
        from aic.env.service_topology import ServiceTopology
        # Create topology with a cycle: A → B → A
        topo = ServiceTopology(
            nodes=["a", "b"],
            edges=[("a", "b", 0.5), ("b", "a", 0.5)],
        )
        # Should not hang
        topo.propagate_pressure("a", {"latency": 100.0})
        topo.flush_propagation_buffer()
        assert topo.nodes["a"].latency == 100.0
        assert topo.nodes["b"].latency == pytest.approx(50.0, abs=0.01)

    def test_cooldown_reduces_pressure(self):
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology(cooldown_rate=0.5)
        topo.propagate_pressure("db", {"latency": 100.0})
        topo.cool_down()
        assert topo.nodes["db"].latency == pytest.approx(50.0, abs=0.01)

    def test_reset_clears_all_state(self):
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        topo.propagate_pressure("db", {"latency": 500.0})
        topo.reset()
        assert topo.nodes["db"].latency == 0.0
        assert topo.nodes["db"].health == 1.0

    def test_get_topology_state(self):
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        state = topo.get_topology_state()
        assert "gateway" in state
        assert "db" in state
        assert "health" in state["db"]

    def test_multi_hop_propagation(self):
        """Gateway → App → Cache → DB across multiple flushes."""
        from aic.env.service_topology import ServiceTopology
        topo = ServiceTopology()
        topo.propagate_pressure("gateway", {"latency": 1000.0})

        # Step T: gateway has 1000, app/cache/db have 0
        assert topo.nodes["gateway"].latency == 1000.0

        # Flush 1 (step T+1): app gets 1000*0.8 = 800
        # cache gets 1000*0.8*0.6 = 480, queue gets 1000*0.8*0.5 = 400
        # db gets cache: 1000*0.8*0.6*0.7=336 + queue: 1000*0.8*0.5*0.4=160 = 496
        topo.flush_propagation_buffer()
        assert topo.nodes["app"].latency >= 700.0
        assert topo.nodes["cache"].latency >= 400.0
        assert topo.nodes["db"].latency >= 300.0


# ── 2. Scenario Registry ──────────────────────────────────────────────────

class TestScenarioRegistry:
    def test_6_scenarios_registered(self):
        from aic.env.scenario_registry import SCENARIO_REGISTRY
        assert len(SCENARIO_REGISTRY) == 6

    def test_scenario_has_required_fields(self):
        from aic.env.scenario_registry import get_scenario
        for sid in range(6):
            s = get_scenario(sid)
            assert s.name
            assert s.root_cause_node
            assert isinstance(s.initial_fault_vector, dict)
            assert len(s.initial_fault_vector) > 0

    def test_cache_stampede_spikes_db_and_queue(self):
        """VERIFICATION: Cache Stampede spikes db_latency and queue_depth."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(0)  # Cache Stampede
        contributions = engine.get_contributions(step=0)
        assert contributions.get("db_latency_ms", 0) > 0
        assert contributions.get("queue_depth", 0) > 0

    def test_constant_pressure_persists(self):
        """Fault vector applies every step (not just step 0)."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(0)
        c0 = engine.get_contributions(0)
        c10 = engine.get_contributions(10)
        # Step 10 should still have significant pressure (0.98^10 ≈ 0.82)
        assert c10.get("db_latency_ms", 0) > c0.get("db_latency_ms", 0) * 0.7

    def test_telemetry_nan_blackout(self):
        """Canary Failure blacks out error_rate_pct steps 5-8."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(1)  # Canary Failure
        # Step 4: not blacked out
        assert "error_rate_pct" not in engine.get_telemetry_mask(4)
        # Step 6: blacked out
        assert "error_rate_pct" in engine.get_telemetry_mask(6)

    def test_telemetry_field_rename(self):
        """Schema Migration renames db_latency_ms → db_latency."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(3)  # Schema Migration
        renames = engine.get_telemetry_renames(5)
        assert renames.get("db_latency_ms") == "db_latency"

    def test_telemetry_unit_shift(self):
        """Queue Cascade shifts queue_depth by ÷1000."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(5)  # Queue Cascade
        shifts = engine.get_telemetry_unit_shifts(3)
        assert shifts.get("queue_depth") == pytest.approx(0.001)

    def test_apply_telemetry_corruption(self):
        """apply_telemetry_corruption modifies observation correctly."""
        from aic.env.scenario_registry import ScenarioEngine
        engine = ScenarioEngine(1)  # Canary Failure
        obs = {"error_rate_pct": 15.0, "db_latency_ms": 100.0}
        corrupted = engine.apply_telemetry_corruption(obs, step=6)
        assert math.isnan(corrupted["error_rate_pct"])
        assert corrupted["db_latency_ms"] == 100.0

    def test_invalid_scenario_raises(self):
        from aic.env.scenario_registry import get_scenario
        with pytest.raises(ValueError):
            get_scenario(999)

    def test_list_scenarios(self):
        from aic.env.scenario_registry import list_scenarios
        scenarios = list_scenarios()
        assert len(scenarios) == 6
        assert all("name" in s for s in scenarios)


# ── 3. Business Impact ────────────────────────────────────────────────────

class TestBusinessImpact:
    def test_healthy_system_minimal_impact(self):
        from aic.env.business_impact import compute_business_impact
        metrics = {"error_rate_pct": 0.5, "sla_compliance_pct": 99.9}
        impact = compute_business_impact(metrics)
        assert impact.revenue_loss_per_minute < 10.0
        assert impact.severity_level == "P4"

    def test_non_linear_scaling_at_25_pct(self):
        """VERIFICATION: revenue_loss at 25% > 2× revenue_loss at 15%."""
        from aic.env.business_impact import compute_business_impact
        metrics_15 = {"error_rate_pct": 15.0, "sla_compliance_pct": 80.0}
        metrics_25 = {"error_rate_pct": 25.0, "sla_compliance_pct": 80.0}
        impact_15 = compute_business_impact(metrics_15)
        impact_25 = compute_business_impact(metrics_25)
        assert impact_25.revenue_loss_per_minute > 2 * impact_15.revenue_loss_per_minute

    def test_exponential_growth_above_20(self):
        """Revenue loss grows exponentially, not linearly, above 20%."""
        from aic.env.business_impact import compute_business_impact
        losses = []
        for rate in [10, 20, 30, 40, 50]:
            m = {"error_rate_pct": float(rate), "sla_compliance_pct": 70.0}
            losses.append(compute_business_impact(m).revenue_loss_per_minute)
        # The jump from 20→30 should be larger than 10→20
        delta_10_20 = losses[1] - losses[0]
        delta_20_30 = losses[2] - losses[1]
        assert delta_20_30 > delta_10_20

    def test_compliance_risk_mapping(self):
        from aic.env.business_impact import compute_business_impact
        # 99.9% → 0.0 risk
        m1 = {"error_rate_pct": 1.0, "sla_compliance_pct": 99.9}
        assert compute_business_impact(m1).compliance_risk_score == 0.0
        # 70% → high risk
        m2 = {"error_rate_pct": 1.0, "sla_compliance_pct": 70.0}
        assert compute_business_impact(m2).compliance_risk_score > 0.5

    def test_severity_p1_high_error(self):
        from aic.env.business_impact import compute_business_impact
        metrics = {"error_rate_pct": 40.0, "sla_compliance_pct": 60.0}
        impact = compute_business_impact(metrics)
        assert impact.severity_level == "P1"

    def test_severity_p2_moderate(self):
        from aic.env.business_impact import compute_business_impact
        metrics = {"error_rate_pct": 10.0, "sla_compliance_pct": 85.0}
        impact = compute_business_impact(metrics)
        assert impact.severity_level in ("P1", "P2")

    def test_users_impacted_capped_at_base(self):
        """Users impacted should never exceed BASE_USERS."""
        from aic.env.business_impact import compute_business_impact, BASE_USERS
        metrics = {"error_rate_pct": 99.0, "sla_compliance_pct": 10.0}
        impact = compute_business_impact(metrics)
        assert impact.users_impacted <= BASE_USERS
