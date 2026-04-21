# tests/test_scaffold.py
"""
Phase 1 scaffold tests.
Verifies: version string, constants import, seeding reproducibility,
adversary cycle balance, and episode logging round-trip.
"""
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── 1. Version ──────────────────────────────────────────────────────────────

def test_version():
    import aic
    assert aic.__version__ == "0.1.0"


# ── 2. Constants ────────────────────────────────────────────────────────────

def test_sla_steps():
    from aic.utils.constants import SLA_STEPS
    assert SLA_STEPS == 20


def test_metric_targets_length():
    from aic.utils.constants import METRIC_TARGETS
    assert len(METRIC_TARGETS) == 12


def test_metric_fault_init_length():
    from aic.utils.constants import METRIC_FAULT_INIT
    assert len(METRIC_FAULT_INIT) == 12


def test_metric_targets_keys_match_fault_init():
    from aic.utils.constants import METRIC_TARGETS, METRIC_FAULT_INIT
    assert set(METRIC_TARGETS.keys()) == set(METRIC_FAULT_INIT.keys())


def test_services():
    from aic.utils.constants import SERVICES
    assert SERVICES == ["db", "infra", "app"]


def test_all_agents():
    from aic.utils.constants import ALL_AGENTS
    assert len(ALL_AGENTS) == 6
    assert "adversarial_agent" in ALL_AGENTS


def test_weights_sum_to_one():
    from aic.utils.constants import WEIGHT_DB, WEIGHT_INFRA, WEIGHT_APP
    assert abs((WEIGHT_DB + WEIGHT_INFRA + WEIGHT_APP) - 1.0) < 1e-9


# ── 3. Seeding ──────────────────────────────────────────────────────────────

def test_make_episode_rng_reproducible():
    from aic.utils.seeding import make_episode_rng
    rng1 = make_episode_rng(0)
    rng2 = make_episode_rng(0)
    assert rng1.integers(0, 100) == rng2.integers(0, 100)


def test_make_episode_rng_different_episodes():
    from aic.utils.seeding import make_episode_rng
    rng1 = make_episode_rng(0)
    rng2 = make_episode_rng(1)
    # Different episodes should (almost certainly) produce different sequences
    vals1 = [rng1.integers(0, 10000) for _ in range(10)]
    vals2 = [rng2.integers(0, 10000) for _ in range(10)]
    assert vals1 != vals2


def test_adversary_cycle_balance():
    from aic.utils.seeding import make_episode_rng, get_adversary_cycle
    rng = make_episode_rng(0)
    cycle = get_adversary_cycle(rng)
    assert len(cycle) == 20
    assert sum(cycle) == 10  # exactly 50% True


def test_adversary_cycle_reproducible():
    from aic.utils.seeding import make_episode_rng, get_adversary_cycle
    cycle1 = get_adversary_cycle(make_episode_rng(7))
    cycle2 = get_adversary_cycle(make_episode_rng(7))
    assert cycle1 == cycle2


def test_adversary_cycle_all_bools():
    from aic.utils.seeding import make_episode_rng, get_adversary_cycle
    cycle = get_adversary_cycle(make_episode_rng(42))
    assert all(isinstance(v, (bool, np.bool_)) for v in cycle)


def test_get_t_drift_in_range():
    from aic.utils.seeding import make_episode_rng, get_t_drift
    from aic.utils.constants import T_DRIFT_MIN, T_DRIFT_MAX
    for ep in range(50):
        rng = make_episode_rng(ep)
        t = get_t_drift(rng, T_DRIFT_MIN, T_DRIFT_MAX)
        assert T_DRIFT_MIN <= t <= T_DRIFT_MAX


# ── 4. Logging ──────────────────────────────────────────────────────────────

def test_episode_logger_roundtrip():
    from aic.utils.logging_utils import EpisodeLogger, StepRecord, load_episode

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(log_dir=tmpdir, episode_id=0)

        record = StepRecord(
            episode_id=0,
            step=0,
            timestamp=time.time(),
            world_state={"db_latency_ms": 850.0, "cpu_pct": 89.0},
            agent_recommendations={"db_agent": "scale_pool"},
            orchestrator_action="increase_pool_size",
            reward_components={"r1": -5.0, "r2": 0.0},
            reward_total=-5.0,
            trust_scores={"db_agent": 0.5, "infra_agent": 0.5},
            schema_drift_active=False,
            schema_drift_type=None,
            deadlock_detected=False,
        )
        logger.log_step(record)

        # Finalize and check summary
        summary = logger.finalize(total_reward=-5.0, success=False)
        assert summary["episode_id"] == 0
        assert summary["total_steps"] == 1
        assert summary["total_reward"] == -5.0

        # Load back and verify
        loaded = load_episode(tmpdir, 0)
        assert len(loaded) == 1
        assert loaded[0]["step"] == 0
        assert loaded[0]["world_state"]["db_latency_ms"] == 850.0


def test_episode_logger_multiple_steps():
    from aic.utils.logging_utils import EpisodeLogger, StepRecord, load_episode

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(log_dir=tmpdir, episode_id=1)

        for step in range(5):
            record = StepRecord(
                episode_id=1,
                step=step,
                timestamp=time.time(),
                world_state={"db_latency_ms": 850.0 - step * 50},
                agent_recommendations={},
                orchestrator_action="noop",
                reward_components={"r1": float(-step)},
                reward_total=float(-step),
                trust_scores={},
                schema_drift_active=False,
                schema_drift_type=None,
                deadlock_detected=False,
            )
            logger.log_step(record)

        loaded = load_episode(tmpdir, 1)
        assert len(loaded) == 5
        assert loaded[4]["step"] == 4


def test_load_episode_file_not_found():
    from aic.utils.logging_utils import load_episode
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_episode(tmpdir, 9999)
