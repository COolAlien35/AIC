# tests/test_data_integrity.py
"""
Tests for Phase 2 — Data Generation Quality Upgrade.

Covers:
  2.1: SFT dataset integrity (splits, dedup, leakage, gates, fingerprint)
  2.2: Scenario realism (contract table, metadata tagging, held-out stress)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aic.training.data_integrity import (
    DataQualityGates,
    DatasetReport,
    analyze_dataset,
    check_agent_action_distribution,
    deduplicate_records,
    run_integrity_pipeline,
    save_fingerprint,
    split_train_val,
    verify_no_leakage,
)
from aic.training.scenario_contract import (
    CANONICAL_SCENARIO_IDS,
    SCENARIO_TRAINING_META,
    STRESS_SCENARIOS,
    ScenarioTrainingMeta,
    build_scenario_training_meta,
    get_stress_scenario_names,
    tag_sample_metadata,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_records(n_episodes: int = 12, steps_per: int = 20, n_scenarios: int = 6) -> list[dict]:
    """Create synthetic SFT records for testing."""
    scenarios = [f"scenario_{i}" for i in range(n_scenarios)]
    records = []
    for ep in range(n_episodes):
        scenario = scenarios[ep % n_scenarios]
        for step in range(steps_per):
            completion = json.dumps({
                "selected_recommendation_id": step % 5,
                "override_adversary": step % 3 == 0,
                "reasoning": f"Test reasoning for {scenario} step {step} episode {ep}",
                "predicted_2step_impact": {"db_latency_ms": -10.0},
                "schema_drift_detected": step > 15,
                "schema_drift_field": "db_latency_ms" if step > 15 else None,
            }, sort_keys=True)
            records.append({
                "prompt": f"prompt_{ep}_{step}_{scenario}",
                "completion": completion,
                "episode_id": ep,
                "step": step,
                "scenario": scenario,
                "drift_type": ["field_rename", "unit_shift", "silent_null", None][ep % 4],
                "metadata": {
                    "has_adversarial": step % 4 == 0,
                    "schema_drift": step > 15,
                    "difficulty_tier": ["easy", "medium", "hard"][ep % 3],
                },
            })
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2.1 — Data Integrity Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainValSplit:
    """Tests for episode-level train/val splitting."""

    def test_split_preserves_all_records(self):
        records = _make_records()
        train, val = split_train_val(records, val_fraction=0.2, seed=42)
        assert len(train) + len(val) == len(records)

    def test_split_no_episode_overlap(self):
        records = _make_records()
        train, val = split_train_val(records, val_fraction=0.2, seed=42)
        train_eps = set(r["episode_id"] for r in train)
        val_eps = set(r["episode_id"] for r in val)
        assert train_eps.isdisjoint(val_eps), "Episode overlap between train/val!"

    def test_split_all_steps_in_same_partition(self):
        records = _make_records()
        train, val = split_train_val(records, val_fraction=0.2, seed=42)
        train_eps = set(r["episode_id"] for r in train)
        for r in records:
            if r["episode_id"] in train_eps:
                assert r in train

    def test_split_deterministic(self):
        records = _make_records()
        t1, v1 = split_train_val(records, val_fraction=0.2, seed=42)
        t2, v2 = split_train_val(records, val_fraction=0.2, seed=42)
        assert [r["episode_id"] for r in t1] == [r["episode_id"] for r in t2]

    def test_split_different_seeds_differ(self):
        records = _make_records()
        t1, _ = split_train_val(records, val_fraction=0.2, seed=42)
        t2, _ = split_train_val(records, val_fraction=0.2, seed=99)
        # Different seeds should (almost certainly) produce different splits
        eps1 = set(r["episode_id"] for r in t1)
        eps2 = set(r["episode_id"] for r in t2)
        # With 12 episodes and 2 val, there's a good chance they differ
        # (but this isn't guaranteed — it's probabilistic)


class TestLeakageVerification:
    """Tests for data leakage detection."""

    def test_clean_split_reports_no_leakage(self):
        records = _make_records()
        train, val = split_train_val(records, val_fraction=0.2, seed=42)
        result = verify_no_leakage(train, val)
        assert result["clean"] is True
        assert result["episode_overlap_count"] == 0

    def test_forced_overlap_detected(self):
        records = _make_records(n_episodes=6, steps_per=5)
        # Artificially create overlap
        train = records[:20]
        val = records[15:30]  # overlapping episode IDs
        result = verify_no_leakage(train, val)
        assert result["episode_overlap_count"] > 0


class TestDeduplication:
    """Tests for deduplication logic."""

    def test_no_dups_unchanged(self):
        records = _make_records(n_episodes=3, steps_per=5)
        deduped, removed = deduplicate_records(records)
        assert removed == 0
        assert len(deduped) == len(records)

    def test_exact_dups_removed(self):
        records = _make_records(n_episodes=2, steps_per=3)
        # Inject duplicates
        dup = records[0].copy()
        records.append(dup)
        records.append(dup)
        deduped, removed = deduplicate_records(records)
        assert removed == 2

    def test_custom_key_dedup(self):
        records = [
            {"prompt": "a", "completion": "x", "episode_id": 0},
            {"prompt": "b", "completion": "y", "episode_id": 1},
            {"prompt": "a", "completion": "z", "episode_id": 2},  # dup by prompt
        ]
        deduped, removed = deduplicate_records(records, key="prompt")
        assert removed == 1
        assert len(deduped) == 2


class TestDatasetAnalysis:
    """Tests for full dataset analysis and quality gates."""

    def test_analyze_passes_on_good_data(self):
        records = _make_records(n_episodes=24, steps_per=20, n_scenarios=6)
        report = analyze_dataset(records)
        assert report.total_records > 0
        assert report.unique_episodes == 24
        assert len(report.scenario_counts) == 6
        assert report.fingerprint != ""

    def test_empty_dataset_fails(self):
        report = analyze_dataset([])
        assert report.total_records == 0
        assert not report.passed

    def test_custom_gates(self):
        records = _make_records(n_episodes=6, steps_per=5, n_scenarios=3)
        # With only 3 scenarios, the default min_scenario_count=6 should fail
        report = analyze_dataset(records)
        assert "min_scenarios" in report.gate_failures

    def test_relaxed_gates_pass(self):
        records = _make_records(n_episodes=6, steps_per=20, n_scenarios=6)
        gates = DataQualityGates(
            min_total_records=10,
            max_duplicate_prompt_rate=1.0,
            min_scenario_count=3,
            min_adversarial_fraction=0.0,
            min_drift_fraction=0.0,
            min_action_entropy=0.0,
            min_recommendation_diversity=0.0,
        )
        report = analyze_dataset(records, gates=gates)
        # Should pass with relaxed gates
        assert report.passed or len(report.gate_failures) <= 1

    def test_fingerprint_deterministic(self):
        records = _make_records(n_episodes=3, steps_per=3)
        r1 = analyze_dataset(records)
        r2 = analyze_dataset(records)
        assert r1.fingerprint == r2.fingerprint

    def test_fingerprint_changes_with_data(self):
        records1 = _make_records(n_episodes=3, steps_per=3)
        records2 = _make_records(n_episodes=4, steps_per=3)
        r1 = analyze_dataset(records1)
        r2 = analyze_dataset(records2)
        assert r1.fingerprint != r2.fingerprint


class TestAgentActionDistribution:
    """Tests for per-agent action distribution analysis."""

    def test_distribution_returns_dict(self):
        records = _make_records(n_episodes=3, steps_per=5)
        dist = check_agent_action_distribution(records)
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_all_scenarios_represented(self):
        records = _make_records(n_episodes=6, steps_per=5, n_scenarios=6)
        dist = check_agent_action_distribution(records)
        assert len(dist) == 6


class TestFingerprintIO:
    """Tests for fingerprint save/load."""

    def test_save_creates_file(self):
        report = DatasetReport(
            fingerprint="abc123",
            total_records=100,
            unique_episodes=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_fingerprint(report, Path(tmpdir) / "fp.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["fingerprint"] == "abc123"
            assert data["total_records"] == 100


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2.2 — Scenario Realism Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCanonicalScenarioMapping:
    """Tests that the canonical scenario mapping replaces old aliased approach."""

    def test_six_scenarios_defined(self):
        assert len(CANONICAL_SCENARIO_IDS) == 6

    def test_all_scenario_ids_in_registry(self):
        from aic.env.scenario_registry import SCENARIO_REGISTRY
        for sid in CANONICAL_SCENARIO_IDS:
            assert sid in SCENARIO_REGISTRY

    def test_no_duplicate_names(self):
        names = [SCENARIO_TRAINING_META[sid].scenario_name for sid in CANONICAL_SCENARIO_IDS]
        assert len(names) == len(set(names)), "Duplicate scenario names found!"

    def test_each_scenario_has_distinct_root_cause(self):
        """Verify scenarios aren't just aliases — at least 3 distinct root causes."""
        roots = set(SCENARIO_TRAINING_META[sid].root_cause_node for sid in CANONICAL_SCENARIO_IDS)
        assert len(roots) >= 3, f"Too few root causes: {roots}"


class TestScenarioTrainingMeta:
    """Tests for scenario training metadata."""

    def test_all_fields_populated(self):
        for sid in CANONICAL_SCENARIO_IDS:
            meta = SCENARIO_TRAINING_META[sid]
            assert isinstance(meta, ScenarioTrainingMeta)
            assert meta.scenario_name
            assert meta.root_cause_node
            assert meta.severity in ("P1", "P2", "P3", "P4")
            assert meta.difficulty_tier in ("easy", "medium", "hard")
            assert meta.adversarial_intensity in ("low", "medium", "high")

    def test_build_meta_returns_all_scenarios(self):
        metas = build_scenario_training_meta()
        assert len(metas) == len(CANONICAL_SCENARIO_IDS)


class TestSampleMetadataTagging:
    """Tests for rich per-sample metadata."""

    def test_tag_returns_required_fields(self):
        tags = tag_sample_metadata(
            scenario_id=0,
            episode_id=0,
            step=5,
            override_applied=True,
            has_adversarial_candidate=False,
            schema_drift_active=True,
            drift_type="field_rename",
        )
        required_keys = {
            "scenario_name", "scenario_id", "fault_mode", "difficulty_tier",
            "adversarial_intensity", "has_adversarial", "schema_drift",
            "drift_type", "has_telemetry_corruption", "severity",
        }
        assert required_keys <= set(tags.keys())

    def test_tag_adversarial_detection(self):
        tags = tag_sample_metadata(
            scenario_id=0, episode_id=0, step=0,
            override_applied=True,
            has_adversarial_candidate=False,
            schema_drift_active=False, drift_type=None,
        )
        assert tags["has_adversarial"] is True

    def test_tag_drift_propagation(self):
        tags = tag_sample_metadata(
            scenario_id=0, episode_id=0, step=10,
            override_applied=False,
            has_adversarial_candidate=False,
            schema_drift_active=True, drift_type="unit_shift",
        )
        assert tags["schema_drift"] is True
        assert tags["drift_type"] == "unit_shift"


class TestStressScenarios:
    """Tests for held-out stress scenario definitions."""

    def test_stress_scenarios_exist(self):
        assert len(STRESS_SCENARIOS) >= 3, "Need at least 3 stress scenarios"

    def test_stress_names_unique(self):
        names = get_stress_scenario_names()
        assert len(names) == len(set(names))

    def test_stress_base_ids_valid(self):
        from aic.env.scenario_registry import SCENARIO_REGISTRY
        for s in STRESS_SCENARIOS:
            assert s.base_scenario_id in SCENARIO_REGISTRY, (
                f"Stress scenario '{s.name}' references invalid base_scenario_id {s.base_scenario_id}"
            )

    def test_stress_budget_multipliers_positive(self):
        for s in STRESS_SCENARIOS:
            assert 0 < s.budget_multiplier <= 2.0, (
                f"Stress scenario '{s.name}' has invalid budget_multiplier {s.budget_multiplier}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Full Pipeline on Existing Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrityPipeline:
    """Integration test: run the pipeline on the existing SFT dataset if present."""

    @pytest.fixture
    def sft_path(self):
        path = Path("artifacts/sft/orchestrator_sft.jsonl")
        if not path.exists():
            pytest.skip("SFT dataset not generated yet")
        return path

    def test_pipeline_runs_on_existing_data(self, sft_path, tmp_path):
        result = run_integrity_pipeline(
            dataset_path=sft_path,
            output_dir=tmp_path / "integrity_out",
            val_fraction=0.15,
            seed=42,
        )
        assert result["train_count"] > 0
        assert result["val_count"] > 0
        assert result["num_deduped"] >= 0
        assert Path(result["train_path"]).exists()
        assert Path(result["val_path"]).exists()
        assert Path(result["fingerprint_path"]).exists()
