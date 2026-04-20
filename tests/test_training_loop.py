# tests/test_training_loop.py
"""
Phase 5 tests.
Covers training config, episode rollouts, trajectory caching,
reward curve CSV, and trained vs untrained comparison.
"""
import pickle
import tempfile
from pathlib import Path

import pytest

from aic.training.config import TrainingConfig
from aic.training.train import run_episode, train
from aic.utils.constants import ALL_AGENTS, INITIAL_TRUST, SLA_STEPS
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent


# ── 1. Config ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.model_name == "Qwen/Qwen2-0.5B-Instruct"
        assert config.lora_r == 8
        assert config.lora_alpha == 32
        assert config.num_episodes == 100
        assert config.checkpoint_interval == 25
        assert config.use_llm_agents is False

    def test_override(self):
        config = TrainingConfig(num_episodes=5, learning_rate=5e-5)
        assert config.num_episodes == 5
        assert config.learning_rate == 5e-5


# ── 2. Single episode ─────────────────────────────────────────────────────

class TestSingleEpisode:
    def test_run_episode_returns_expected_keys(self):
        config = TrainingConfig(num_episodes=1, use_llm_agents=False)
        db = DBAgent(use_llm=False)
        infra = InfraAgent(use_llm=False)
        app = AppAgent(use_llm=False)
        cycle = get_adversary_cycle(make_episode_rng(0, config.base_seed))
        adv = AdversarialAgent(cycle, db)
        orch = OrchestratorAgent(adv, use_llm=False)

        result = run_episode(0, config, orch, db, infra, app)

        assert "episode_id" in result
        assert "total_reward" in result
        assert "trajectory" in result
        assert "trust_evolution" in result
        assert "reward_history" in result
        assert "final_health" in result
        assert len(result["trajectory"]) == SLA_STEPS
        assert len(result["trust_evolution"]) == SLA_STEPS

    def test_trajectory_step_has_expected_fields(self):
        config = TrainingConfig(num_episodes=1, use_llm_agents=False)
        db = DBAgent(use_llm=False)
        infra = InfraAgent(use_llm=False)
        app = AppAgent(use_llm=False)
        cycle = get_adversary_cycle(make_episode_rng(0, config.base_seed))
        adv = AdversarialAgent(cycle, db)
        orch = OrchestratorAgent(adv, use_llm=False)

        result = run_episode(0, config, orch, db, infra, app)
        step0 = result["trajectory"][0]

        assert "step" in step0
        assert "metrics" in step0
        assert "health" in step0
        assert "action" in step0
        assert "trust_scores" in step0
        assert "reward" in step0
        assert "trace" in step0
        assert "drift_active" in step0

    def test_different_episodes_different_rewards(self):
        config = TrainingConfig(use_llm_agents=False)
        db = DBAgent(use_llm=False)
        infra = InfraAgent(use_llm=False)
        app = AppAgent(use_llm=False)

        results = []
        for ep in range(3):
            cycle = get_adversary_cycle(make_episode_rng(ep, config.base_seed))
            adv = AdversarialAgent(cycle, db)
            orch = OrchestratorAgent(adv, use_llm=False)
            results.append(run_episode(ep, config, orch, db, infra, app))

        rewards = [r["total_reward"] for r in results]
        # Not all identical (different drift types, adversary cycles)
        assert len(set(f"{r:.2f}" for r in rewards)) > 1


# ── 3. Training loop ──────────────────────────────────────────────────────

class TestTrainLoop:
    def test_train_5_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_episodes=5,
                checkpoint_interval=5,
                output_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
                trajectories_dir=f"{tmpdir}/assets",
            )
            results = train(config)

            assert len(results) == 5

            # Check trajectory cache exists
            traj_path = Path(tmpdir) / "assets" / "trained_trajectories.pkl"
            assert traj_path.exists()

            with open(traj_path, "rb") as f:
                cached = pickle.load(f)
            assert 0 in cached  # episode 0 always cached
            assert 4 in cached  # last episode always cached

    def test_reward_curve_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_episodes=3,
                checkpoint_interval=3,
                output_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
                trajectories_dir=f"{tmpdir}/assets",
            )
            train(config)

            csv_path = Path(tmpdir) / "logs" / "reward_curve.csv"
            assert csv_path.exists()

            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) == 3
            assert "episode" in df.columns
            assert "total_reward" in df.columns
            assert "avg_r1" in df.columns
            assert "avg_r3" in df.columns
            assert "avg_r4" in df.columns
            assert "sum_r1" in df.columns

    def test_checkpoint_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_episodes=5,
                checkpoint_interval=5,
                output_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
                trajectories_dir=f"{tmpdir}/assets",
            )
            train(config)

            cp_path = Path(tmpdir) / "checkpoints" / "checkpoint_ep004.json"
            assert cp_path.exists()

            import json
            with open(cp_path) as f:
                cp = json.load(f)
            assert cp["episode_id"] == 4
            assert len(cp["episode_results_so_far"]) == 5


# ── 4. Trained vs untrained comparison ─────────────────────────────────────

class TestTrainedVsUntrained:
    def test_adversary_cycle_per_episode(self):
        """Different episodes must have different adversary cycles."""
        c0 = get_adversary_cycle(make_episode_rng(0))
        c1 = get_adversary_cycle(make_episode_rng(1))
        assert c0 != c1
        assert sum(c0) == 10
        assert sum(c1) == 10

    def test_frozen_trust_stays_at_initial(self):
        """Untrained orchestrator should keep trust at 0.5."""
        from scripts.benchmark_untrained import FrozenTrustOrchestrator

        cycle = get_adversary_cycle(make_episode_rng(0))
        db = DBAgent(use_llm=False)
        adv = AdversarialAgent(cycle, db)
        orch = FrozenTrustOrchestrator(adv, use_llm=False)

        # Simulate a trust update call — should be no-op
        orch._update_trust_scores(0, {"db_latency_ms": 900}, {"db_latency_ms": 50})
        for agent in ALL_AGENTS:
            assert orch.trust_scores[agent] == INITIAL_TRUST
