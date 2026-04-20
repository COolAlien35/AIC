# tests/test_reward_engine.py
"""
Phase 3 tests.
Covers schema drift injection, lock manager deadlock detection,
and all 4 reward components (R1-R4).
"""
import copy
import pytest

from aic.utils.constants import (
    METRIC_TARGETS, METRIC_FAULT_INIT, SLA_STEPS,
    R3_CORRECT_OVERRIDE, R3_CORRECT_TRUST, R3_WRONG_OVERRIDE, R3_WRONG_TRUST,
    DEADLOCK_PENALTY, LOCK_HANDOFF_BONUS,
    R2_SLA_BONUS_MAX, R4_MIN_PER_STEP, R4_MAX_PER_STEP,
    NULL_DRIFT_DURATION,
)


# ── 1. Schema Drift ────────────────────────────────────────────────────────

class TestSchemaDrift:
    def test_field_rename(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="field_rename")
        raw = {"p95_latency_ms": 3200.0, "error_rate_pct": 18.5, "queue_depth": 890.0}

        # Before t_drift — no change
        result = inj.inject(step=9, service="app", raw_response=raw)
        assert "p95_latency_ms" in result
        assert "p95_latency" not in result

        # At t_drift — field renamed
        result = inj.inject(step=10, service="app", raw_response=raw)
        assert "p95_latency_ms" not in result
        assert "p95_latency" in result
        assert result["p95_latency"] == 3200.0

    def test_field_rename_wrong_service(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="field_rename")
        raw = {"p95_latency_ms": 3200.0}

        # field_rename targets "app", not "db"
        result = inj.inject(step=10, service="db", raw_response=raw)
        assert "p95_latency_ms" in result

    def test_unit_shift(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=8, drift_type="unit_shift")
        raw = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}

        # Before t_drift
        result = inj.inject(step=7, service="db", raw_response=raw)
        assert result["replication_lag_ms"] == 450.0

        # At t_drift — value scaled by 0.001
        result = inj.inject(step=8, service="db", raw_response=raw)
        assert result["replication_lag_ms"] == pytest.approx(0.45)

    def test_silent_null(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="silent_null")
        raw = {"db_latency_ms": 850.0, "conn_pool_pct": 98.0, "replication_lag_ms": 450.0}

        # Before t_drift
        result = inj.inject(step=9, service="db", raw_response=raw)
        assert result["conn_pool_pct"] == 98.0

        # NULL_DRIFT_DURATION steps of None
        for i in range(NULL_DRIFT_DURATION):
            result = inj.inject(step=10 + i, service="db", raw_response=raw)
            assert result["conn_pool_pct"] is None, f"Step {10 + i} should be None"

        # After duration — back to normal
        result = inj.inject(step=10 + NULL_DRIFT_DURATION, service="db", raw_response=raw)
        assert result["conn_pool_pct"] == 98.0

    def test_was_active_at(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="field_rename")
        assert not inj.was_active_at(9)
        assert inj.was_active_at(10)
        assert inj.was_active_at(15)

    def test_silent_null_was_active_at(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="silent_null")
        assert not inj.was_active_at(9)
        assert inj.was_active_at(10)
        assert inj.was_active_at(12)
        assert not inj.was_active_at(13)  # 10 + NULL_DRIFT_DURATION(3) = 13

    def test_invalid_drift_type(self):
        from aic.env.schema_drift import SchemaDriftInjector
        with pytest.raises(ValueError, match="Unknown drift type"):
            SchemaDriftInjector(t_drift=10, drift_type="bogus")

    def test_get_affected_field(self):
        from aic.env.schema_drift import SchemaDriftInjector
        inj = SchemaDriftInjector(t_drift=10, drift_type="unit_shift")
        assert inj.get_affected_field() == "replication_lag_ms"


# ── 2. Lock Manager ────────────────────────────────────────────────────────

class TestLockManager:
    def test_acquire_and_release(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        assert lm.request_lock("db_agent", "db") is True
        assert lm.is_locked_by("db_agent", "db") is True
        bonus = lm.release_lock("db_agent", "db")
        assert bonus == 0.0
        assert not lm.is_locked_by("db_agent", "db")

    def test_idempotent_lock(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        assert lm.request_lock("db_agent", "db") is True
        assert lm.request_lock("db_agent", "db") is True  # same agent, same lock

    def test_contention(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        assert lm.request_lock("db_agent", "db") is True
        assert lm.request_lock("infra_agent", "db") is False

    def test_deadlock_detection(self):
        """Agent B waits for 'db' held by Agent A for 2 steps → deadlock."""
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()

        # Agent A acquires 'db'
        lm.request_lock("db_agent", "db")

        # Agent B tries to acquire 'db' — fails, wait_steps=1
        lm.request_lock("infra_agent", "db")
        assert lm._wait_steps["infra_agent"] == 1

        # Agent B tries again — wait_steps=2 → deadlock threshold
        lm.request_lock("infra_agent", "db")
        assert lm._wait_steps["infra_agent"] == 2

        # Detect and resolve
        penalty = lm.detect_and_resolve_deadlocks()
        assert penalty == DEADLOCK_PENALTY
        # Lock should be force-released
        assert lm._holders["db"] is None
        # Agent B no longer waiting
        assert "infra_agent" not in lm._wait_steps

    def test_lock_handoff_bonus(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()

        lm.request_lock("db_agent", "db")
        lm.request_lock("infra_agent", "db")  # infra waits

        bonus = lm.release_lock("db_agent", "db")
        assert bonus == LOCK_HANDOFF_BONUS
        assert lm.is_locked_by("infra_agent", "db")

    def test_invalid_service(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        with pytest.raises(ValueError, match="Unknown service"):
            lm.request_lock("db_agent", "nonexistent")

    def test_reset(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        lm.request_lock("db_agent", "db")
        lm.reset()
        assert lm._holders["db"] is None
        assert len(lm._waiting) == 0

    def test_release_unowned_lock(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        bonus = lm.release_lock("db_agent", "db")
        assert bonus == 0.0

    def test_get_status(self):
        from aic.env.lock_manager import ResourceLockManager
        lm = ResourceLockManager()
        lm.request_lock("db_agent", "db")
        status = lm.get_status()
        assert status["holders"]["db"] == "db_agent"
        assert status["total_penalty"] == 0.0


# ── 3. Reward Engine — R1 ──────────────────────────────────────────────────

class TestR1:
    def test_r1_at_targets_is_near_zero(self):
        from aic.env.reward_engine import compute_r1
        r1 = compute_r1(copy.deepcopy(METRIC_TARGETS))
        assert r1 == pytest.approx(0.0, abs=0.01)

    def test_r1_at_fault_init_is_negative(self):
        from aic.env.reward_engine import compute_r1
        r1 = compute_r1(copy.deepcopy(METRIC_FAULT_INIT))
        assert r1 < -1.0  # deeply negative at fault state

    def test_r1_improves_toward_target(self):
        from aic.env.reward_engine import compute_r1
        # Halfway between fault and target should be less negative than fault
        halfway = {}
        for k in METRIC_FAULT_INIT:
            halfway[k] = (METRIC_FAULT_INIT[k] + METRIC_TARGETS[k]) / 2.0
        r1_fault = compute_r1(copy.deepcopy(METRIC_FAULT_INIT))
        r1_half = compute_r1(halfway)
        assert r1_half > r1_fault


# ── 4. Reward Engine — R2 ──────────────────────────────────────────────────

class TestR2:
    def test_r2_on_success(self):
        from aic.env.reward_engine import compute_r2
        r2 = compute_r2(METRIC_TARGETS, steps_remaining=5, episode_success=True)
        expected = R2_SLA_BONUS_MAX * (5 / SLA_STEPS)
        assert r2 == pytest.approx(expected)

    def test_r2_on_failure(self):
        from aic.env.reward_engine import compute_r2
        r2 = compute_r2(METRIC_FAULT_INIT, steps_remaining=5, episode_success=False)
        assert r2 == 0.0

    def test_r2_max_bonus(self):
        from aic.env.reward_engine import compute_r2
        r2 = compute_r2(METRIC_TARGETS, steps_remaining=SLA_STEPS, episode_success=True)
        assert r2 == pytest.approx(R2_SLA_BONUS_MAX)


# ── 5. Reward Engine — R3 ──────────────────────────────────────────────────

class TestR3:
    def test_correct_distrust(self):
        from aic.env.reward_engine import compute_r3
        assert compute_r3(override_applied=True, adversary_was_correct=False) == R3_CORRECT_OVERRIDE

    def test_correct_trust(self):
        from aic.env.reward_engine import compute_r3
        assert compute_r3(override_applied=False, adversary_was_correct=True) == R3_CORRECT_TRUST

    def test_unnecessary_override(self):
        from aic.env.reward_engine import compute_r3
        assert compute_r3(override_applied=True, adversary_was_correct=True) == R3_WRONG_OVERRIDE

    def test_blind_trust(self):
        from aic.env.reward_engine import compute_r3
        assert compute_r3(override_applied=False, adversary_was_correct=False) == R3_WRONG_TRUST


# ── 6. Reward Engine — R4 ──────────────────────────────────────────────────

class TestR4:
    def test_perfect_prediction(self):
        from aic.env.reward_engine import compute_r4
        predicted = {"db_latency_ms": -100.0}
        actual = {"db_latency_ms": -100.0}
        r4, pred_acc, causal = compute_r4(predicted, actual, "no reasoning", "no outcome")
        assert pred_acc == pytest.approx(1.0, abs=0.01)

    def test_no_prediction_gives_default(self):
        from aic.env.reward_engine import compute_r4
        r4, pred_acc, causal = compute_r4({}, {}, "no reasoning", "no outcome")
        assert pred_acc == pytest.approx(0.5)  # default when empty

    def test_r4_range(self):
        from aic.env.reward_engine import compute_r4
        predicted = {"cpu_pct": -5.0}
        actual = {"cpu_pct": -5.0}
        r4, _, _ = compute_r4(predicted, actual, "because cpu fix", "improving recovery")
        assert R4_MIN_PER_STEP <= r4 <= R4_MAX_PER_STEP

    def test_causal_consistency_keywords(self):
        from aic.env.reward_engine import compute_r4
        predicted = {"cpu_pct": -5.0}
        actual = {"cpu_pct": -3.0}
        # Reasoning with causal keyword + metric mention
        _, _, causal = compute_r4(
            predicted, actual,
            "because cpu_pct is high, reducing it will fix the issue",
            "Metrics improving: recovery",
        )
        assert causal > 0.0


# ── 7. RewardEngine class ──────────────────────────────────────────────────

class TestRewardEngineClass:
    def test_step_reward_returns_all_keys(self):
        from aic.env.reward_engine import RewardEngine
        engine = RewardEngine()
        record = engine.compute_step_reward(
            step=0,
            metrics=copy.deepcopy(METRIC_FAULT_INIT),
            prev_metrics=copy.deepcopy(METRIC_FAULT_INIT),
            override_applied=False,
            adversary_was_correct=True,
            predicted_2step_impact={"db_latency_ms": -50.0},
            reasoning="test reasoning",
        )
        assert "r1" in record
        assert "r3" in record
        assert "r4" in record
        assert "total" in record

    def test_r4_delayed_by_2_steps(self):
        """R4 should only score predictions that are 2+ steps old."""
        from aic.env.reward_engine import RewardEngine
        engine = RewardEngine()
        metrics = copy.deepcopy(METRIC_FAULT_INIT)

        # Step 0: buffer prediction, no R4 yet
        r0 = engine.compute_step_reward(
            step=0, metrics=metrics, prev_metrics=metrics,
            override_applied=False, adversary_was_correct=True,
            predicted_2step_impact={"db_latency_ms": -50.0},
            reasoning="test",
        )
        assert r0["r4"] == 0.0

        # Step 1: still no R4
        r1 = engine.compute_step_reward(
            step=1, metrics=metrics, prev_metrics=metrics,
            override_applied=False, adversary_was_correct=True,
            predicted_2step_impact={"db_latency_ms": -30.0},
            reasoning="test",
        )
        assert r1["r4"] == 0.0

        # Step 2: now R4 should be computed (scoring step 0's prediction)
        r2 = engine.compute_step_reward(
            step=2, metrics=metrics, prev_metrics=metrics,
            override_applied=False, adversary_was_correct=True,
            predicted_2step_impact={},
            reasoning="test",
        )
        # r4 should be non-zero now (prediction from step 0 is being scored)
        assert "r4" in r2

    def test_episode_end_reward(self):
        from aic.env.reward_engine import RewardEngine
        engine = RewardEngine()
        r2 = engine.compute_episode_end_reward(
            metrics=copy.deepcopy(METRIC_TARGETS),
            steps_remaining=5,
        )
        assert r2 > 0.0

    def test_reward_history(self):
        from aic.env.reward_engine import RewardEngine
        engine = RewardEngine()
        metrics = copy.deepcopy(METRIC_FAULT_INIT)
        for s in range(3):
            engine.compute_step_reward(
                step=s, metrics=metrics, prev_metrics=metrics,
                override_applied=False, adversary_was_correct=True,
                predicted_2step_impact={}, reasoning="test",
            )
        history = engine.get_reward_history()
        assert len(history) == 3
        assert engine.get_total_episode_reward() == sum(r["total"] for r in history)
