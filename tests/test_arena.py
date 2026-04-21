# tests/test_arena.py
"""Tests for the Benchmark Arena and Leaderboard."""
import json
import pytest
from pathlib import Path
from aic.evals.arena import (
    run_arena, compute_composite_score,
    RandomRecoveryPolicy, OraclePolicy,
    ALL_BASELINES, POLICY_ORDER,
)
from aic.evals.leaderboard import (
    load_leaderboard, get_aic_vs_best_baseline, format_leaderboard_table,
    LeaderboardEntry,
)
from aic.evals.benchmark_suite import BenchmarkResult


class TestCompositeScore:

    def test_empty_returns_zero(self):
        assert compute_composite_score([]) == 0.0

    def test_perfect_results_score_near_one(self):
        perfect = BenchmarkResult(
            policy_name="test", scenario_id=0, scenario_name="test",
            mttr_steps=1, final_health=1.0, sla_met=True,
            adversary_suppression_rate=1.0, unsafe_action_rate=0.0,
            revenue_saved_usd=100000, total_reward=100.0,
        )
        score = compute_composite_score([perfect])
        assert score > 0.8

    def test_terrible_results_score_near_zero(self):
        bad = BenchmarkResult(
            policy_name="test", scenario_id=0, scenario_name="test",
            mttr_steps=20, final_health=0.0, sla_met=False,
            adversary_suppression_rate=0.0, unsafe_action_rate=1.0,
            revenue_saved_usd=0, total_reward=-100.0,
        )
        score = compute_composite_score([bad])
        assert score < 0.3


class TestPolicies:

    def test_random_policy_selects_from_candidates(self):
        from aic.schemas.traces import SubAgentRecommendation
        recs = [
            SubAgentRecommendation(
                agent_name=f"agent_{i}", action=f"action_{i}",
                reasoning="test reasoning text", confidence=0.5,
                target_metrics=["cpu_pct"],
            )
            for i in range(3)
        ]
        policy = RandomRecoveryPolicy(seed=42)
        selected = policy.select(recs)
        assert selected in recs

    def test_oracle_policy_picks_lowest_impact(self):
        from aic.schemas.traces import SubAgentRecommendation
        recs = [
            SubAgentRecommendation(
                agent_name="good", action="good action",
                reasoning="good reasoning text", confidence=0.5,
                target_metrics=["cpu_pct"],
                expected_impact={"cpu_pct": -50.0, "mem_pct": -30.0},
            ),
            SubAgentRecommendation(
                agent_name="bad", action="bad action",
                reasoning="bad reasoning text here", confidence=0.8,
                target_metrics=["cpu_pct"],
                expected_impact={"cpu_pct": -5.0},
            ),
        ]
        policy = OraclePolicy()
        selected = policy.select(recs)
        assert selected.agent_name == "good"  # bigger total impact


class TestArenaRun:

    def test_arena_produces_valid_output(self, tmp_path):
        out = str(tmp_path / "arena.json")
        result = run_arena(output_path=out, verbose=False)
        assert "leaderboard" in result
        assert "results" in result
        assert "scenario_wins" in result
        assert len(result["leaderboard"]) == len(POLICY_ORDER)
        # Check file was written
        assert Path(out).exists()
        with open(out) as f:
            data = json.load(f)
        assert len(data["leaderboard"]) > 0


class TestLeaderboard:

    def test_load_missing_file_returns_empty(self):
        entries = load_leaderboard("/nonexistent/path.json")
        assert entries == []

    def test_load_from_arena_output(self, tmp_path):
        out = str(tmp_path / "arena.json")
        run_arena(output_path=out, verbose=False)
        entries = load_leaderboard(out)
        assert len(entries) == len(POLICY_ORDER)
        assert all(isinstance(e, LeaderboardEntry) for e in entries)

    def test_aic_entries_flagged(self, tmp_path):
        out = str(tmp_path / "arena.json")
        run_arena(output_path=out, verbose=False)
        entries = load_leaderboard(out)
        aic_entries = [e for e in entries if e.is_aic]
        assert len(aic_entries) == 2  # Trained + Untrained

    def test_format_table_produces_output(self, tmp_path):
        out = str(tmp_path / "arena.json")
        run_arena(output_path=out, verbose=False)
        entries = load_leaderboard(out)
        table = format_leaderboard_table(entries)
        assert len(table) > 100
        assert "AIC" in table

    def test_aic_vs_baseline_deltas(self, tmp_path):
        out = str(tmp_path / "arena.json")
        run_arena(output_path=out, verbose=False)
        entries = load_leaderboard(out)
        adv = get_aic_vs_best_baseline(entries)
        assert "aic_policy" in adv
        assert "composite_advantage" in adv
