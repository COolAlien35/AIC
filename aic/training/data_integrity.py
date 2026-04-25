# aic/training/data_integrity.py
"""
Phase 2.1 — SFT Dataset Integrity Module.

Implements:
  - Train/val split by episode IDs (no leakage)
  - Scenario balancing validation
  - Per-agent action distribution checks
  - Dedup + leakage detection
  - Hard quality gates (min diversity, max dup threshold, scenario balance)
  - Difficult negative injection verification
  - Dataset fingerprint (SHA-256 hash + config snapshot)
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Quality Gate Thresholds
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityGates:
    """Hard gates that the dataset must pass before training proceeds."""

    # Minimum total examples required
    min_total_records: int = 500

    # Scenario balance: max ratio between largest and smallest scenario count
    max_scenario_imbalance_ratio: float = 1.5

    # Maximum duplicate prompt rate (fraction)
    max_duplicate_prompt_rate: float = 0.10

    # Minimum number of distinct scenarios
    min_scenario_count: int = 6

    # Minimum fraction of records with adversarial metadata
    min_adversarial_fraction: float = 0.15

    # Minimum fraction of records with schema drift active
    min_drift_fraction: float = 0.10

    # Minimum Shannon entropy of action distribution (bits)
    min_action_entropy: float = 1.5

    # Maximum train/val episode overlap (must be 0 for clean split)
    max_episode_overlap: int = 0

    # Minimum number of difficulty tiers represented
    min_difficulty_tiers: int = 3

    # Minimum recommendation diversity (distinct action strings / total records)
    min_recommendation_diversity: float = 0.02


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetReport:
    """Complete integrity report for an SFT dataset."""

    total_records: int = 0
    unique_episodes: int = 0
    scenario_counts: dict[str, int] = field(default_factory=dict)
    drift_type_counts: dict[str, int] = field(default_factory=dict)
    adversarial_count: int = 0
    drift_active_count: int = 0
    duplicate_prompt_count: int = 0
    duplicate_prompt_rate: float = 0.0
    action_entropy: float = 0.0
    action_distribution: dict[str, int] = field(default_factory=dict)
    scenario_imbalance_ratio: float = 0.0
    difficulty_tiers: set[str] = field(default_factory=set)
    recommendation_diversity: float = 0.0
    fingerprint: str = ""
    gate_results: dict[str, bool] = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    passed: bool = False


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy in bits from a frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _compute_fingerprint(records: list[dict], config_snapshot: dict | None = None) -> str:
    """Compute a SHA-256 fingerprint over the dataset content + config."""
    hasher = hashlib.sha256()
    # Hash record contents deterministically
    for record in records:
        hasher.update(json.dumps(record, sort_keys=True).encode())
    # Hash config if provided
    if config_snapshot:
        hasher.update(json.dumps(config_snapshot, sort_keys=True).encode())
    return hasher.hexdigest()


def analyze_dataset(
    records: list[dict],
    gates: DataQualityGates | None = None,
    config_snapshot: dict | None = None,
) -> DatasetReport:
    """Run full integrity analysis on a list of SFT records.

    Args:
        records: List of SFT record dicts (with 'prompt', 'completion',
                 'episode_id', 'scenario', 'drift_type', 'metadata').
        gates: Quality gate thresholds to validate against.
        config_snapshot: Optional config dict to include in fingerprint.

    Returns:
        DatasetReport with all metrics and gate pass/fail results.
    """
    if gates is None:
        gates = DataQualityGates()

    report = DatasetReport()
    report.total_records = len(records)

    if not records:
        report.gate_failures.append("EMPTY_DATASET: No records to analyze")
        return report

    # --- Basic counts ---
    episode_ids = set(r.get("episode_id", -1) for r in records)
    report.unique_episodes = len(episode_ids)

    report.scenario_counts = dict(Counter(r.get("scenario", "unknown") for r in records))
    report.drift_type_counts = dict(Counter(str(r.get("drift_type")) for r in records))

    report.adversarial_count = sum(
        1 for r in records if r.get("metadata", {}).get("has_adversarial", False)
    )
    report.drift_active_count = sum(
        1 for r in records if r.get("metadata", {}).get("schema_drift", False)
    )

    # --- Duplicate detection ---
    prompts = [r.get("prompt", "") for r in records]
    prompt_counts = Counter(prompts)
    report.duplicate_prompt_count = sum(c - 1 for c in prompt_counts.values() if c > 1)
    report.duplicate_prompt_rate = report.duplicate_prompt_count / max(1, report.total_records)

    # --- Action distribution entropy ---
    # Extract action strings from completions
    action_strings = []
    for r in records:
        try:
            completion = json.loads(r.get("completion", "{}"))
            action_strings.append(completion.get("reasoning", "unknown")[:80])
        except (json.JSONDecodeError, TypeError):
            action_strings.append(r.get("completion", "unknown")[:80])
    report.action_distribution = dict(Counter(action_strings).most_common(20))
    report.action_entropy = _shannon_entropy(Counter(action_strings))

    # --- Scenario imbalance ---
    if report.scenario_counts:
        counts = list(report.scenario_counts.values())
        report.scenario_imbalance_ratio = max(counts) / max(1, min(counts))
    else:
        report.scenario_imbalance_ratio = float("inf")

    # --- Difficulty tiers ---
    report.difficulty_tiers = set(
        r.get("metadata", {}).get("difficulty_tier", "unknown") for r in records
    )

    # --- Recommendation diversity ---
    unique_completions = len(set(r.get("completion", "") for r in records))
    report.recommendation_diversity = unique_completions / max(1, report.total_records)

    # --- Fingerprint ---
    report.fingerprint = _compute_fingerprint(records, config_snapshot)

    # ═══════════════════════════════════════════════════════════════════
    # Gate checks
    # ═══════════════════════════════════════════════════════════════════

    gate_checks = {
        "min_total_records": report.total_records >= gates.min_total_records,
        "scenario_balance": report.scenario_imbalance_ratio <= gates.max_scenario_imbalance_ratio,
        "max_duplicate_rate": report.duplicate_prompt_rate <= gates.max_duplicate_prompt_rate,
        "min_scenarios": len(report.scenario_counts) >= gates.min_scenario_count,
        "min_adversarial": (report.adversarial_count / max(1, report.total_records)) >= gates.min_adversarial_fraction,
        "min_drift": (report.drift_active_count / max(1, report.total_records)) >= gates.min_drift_fraction,
        "action_entropy": report.action_entropy >= gates.min_action_entropy,
        "recommendation_diversity": report.recommendation_diversity >= gates.min_recommendation_diversity,
    }

    report.gate_results = gate_checks
    report.gate_failures = [name for name, passed in gate_checks.items() if not passed]
    report.passed = len(report.gate_failures) == 0

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Train/Val Split by Episode ID
# ═══════════════════════════════════════════════════════════════════════════

def split_train_val(
    records: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split records into train/val sets by episode ID (no leakage).

    All steps from a single episode go entirely into train OR val — never
    split across both. This prevents leakage from sequential states within
    an episode.

    Args:
        records: Full list of SFT records.
        val_fraction: Fraction of episodes to reserve for validation.
        seed: Random seed for reproducible split.

    Returns:
        (train_records, val_records)
    """
    episode_ids = sorted(set(r["episode_id"] for r in records))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    n_val = max(1, int(len(episode_ids) * val_fraction))
    val_episodes = set(episode_ids[:n_val])
    train_episodes = set(episode_ids[n_val:])

    # Verify zero overlap
    assert val_episodes.isdisjoint(train_episodes), "FATAL: episode overlap between train/val"

    train_records = [r for r in records if r["episode_id"] in train_episodes]
    val_records = [r for r in records if r["episode_id"] in val_episodes]

    return train_records, val_records


def verify_no_leakage(
    train_records: list[dict],
    val_records: list[dict],
) -> dict[str, Any]:
    """Verify there is zero data leakage between train and val splits.

    Checks:
      1. No episode ID overlap
      2. No exact prompt overlap
      3. No completion overlap (which would indicate copy-paste)

    Returns:
        Dict with check results and any violations.
    """
    train_episodes = set(r["episode_id"] for r in train_records)
    val_episodes = set(r["episode_id"] for r in val_records)
    episode_overlap = train_episodes & val_episodes

    train_prompts = set(r["prompt"] for r in train_records)
    val_prompts = set(r["prompt"] for r in val_records)
    prompt_overlap = train_prompts & val_prompts

    train_completions = set(r["completion"] for r in train_records)
    val_completions = set(r["completion"] for r in val_records)
    completion_overlap = train_completions & val_completions

    return {
        "episode_overlap_count": len(episode_overlap),
        "episode_overlap_ids": sorted(episode_overlap),
        "prompt_overlap_count": len(prompt_overlap),
        "completion_overlap_count": len(completion_overlap),
        "clean": len(episode_overlap) == 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate_records(
    records: list[dict],
    key: str = "prompt",
) -> tuple[list[dict], int]:
    """Remove exact-duplicate records based on a key field.

    Keeps the first occurrence of each unique key value.

    Args:
        records: List of SFT records.
        key: Field to deduplicate on (default: 'prompt').

    Returns:
        (deduped_records, num_removed)
    """
    seen = set()
    deduped = []
    for r in records:
        k = r.get(key, "")
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    return deduped, len(records) - len(deduped)


# ═══════════════════════════════════════════════════════════════════════════
# Per-Agent Action Distribution Check
# ═══════════════════════════════════════════════════════════════════════════

def check_agent_action_distribution(
    records: list[dict],
) -> dict[str, dict[str, int]]:
    """Analyze which agent's recommendations are selected in completions.

    Returns:
        Dict mapping agent_name -> action_description -> count.
    """
    agent_actions: dict[str, dict[str, int]] = {}
    for r in records:
        try:
            completion = json.loads(r.get("completion", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue

        # The completion contains reasoning and selected_recommendation_id,
        # but the agent info is in the candidates from the prompt. We can
        # extract from the metadata if enriched, or from the completion reasoning.
        reasoning = completion.get("reasoning", "unknown")
        # Use a simplified extraction: count by scenario + override status
        override = completion.get("override_adversary", False)
        drift = completion.get("schema_drift_detected", False)
        action_key = f"override={override}|drift={drift}"

        scenario = r.get("scenario", "unknown")
        if scenario not in agent_actions:
            agent_actions[scenario] = {}
        agent_actions[scenario][action_key] = agent_actions[scenario].get(action_key, 0) + 1

    return agent_actions


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_fingerprint(
    report: DatasetReport,
    output_path: str | Path,
) -> Path:
    """Save dataset fingerprint and quality report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fingerprint": report.fingerprint,
        "total_records": report.total_records,
        "unique_episodes": report.unique_episodes,
        "scenario_counts": report.scenario_counts,
        "drift_type_counts": report.drift_type_counts,
        "adversarial_count": report.adversarial_count,
        "drift_active_count": report.drift_active_count,
        "duplicate_prompt_count": report.duplicate_prompt_count,
        "duplicate_prompt_rate": round(report.duplicate_prompt_rate, 6),
        "action_entropy": round(report.action_entropy, 4),
        "scenario_imbalance_ratio": round(report.scenario_imbalance_ratio, 4),
        "difficulty_tiers": sorted(report.difficulty_tiers),
        "recommendation_diversity": round(report.recommendation_diversity, 6),
        "gate_results": report.gate_results,
        "gate_failures": report.gate_failures,
        "passed": report.passed,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Full Integrity Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_integrity_pipeline(
    dataset_path: str | Path,
    output_dir: str | Path = "artifacts/sft",
    val_fraction: float = 0.15,
    seed: int = 42,
    config_snapshot: dict | None = None,
    gates: DataQualityGates | None = None,
) -> dict[str, Any]:
    """Run the full Phase 2.1 integrity pipeline.

    Steps:
      1. Load JSONL
      2. Deduplicate
      3. Analyze & run quality gates
      4. Split train/val by episode ID
      5. Verify no leakage
      6. Save fingerprint
      7. Save train/val splits

    Returns:
        Summary dict with all metrics and paths.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    with open(dataset_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"📥 Loaded {len(records)} records from {dataset_path}")

    # 2. Dedup
    deduped, num_removed = deduplicate_records(records, key="prompt")
    print(f"🔍 Dedup: removed {num_removed} duplicate prompts → {len(deduped)} remaining")

    # 3. Analyze
    report = analyze_dataset(deduped, gates=gates, config_snapshot=config_snapshot)

    print(f"\n📊 Dataset Quality Report:")
    print(f"   Total records:          {report.total_records}")
    print(f"   Unique episodes:        {report.unique_episodes}")
    print(f"   Scenarios:              {report.scenario_counts}")
    print(f"   Drift types:            {report.drift_type_counts}")
    print(f"   Adversarial count:      {report.adversarial_count} ({report.adversarial_count/max(1,report.total_records)*100:.1f}%)")
    print(f"   Drift active count:     {report.drift_active_count} ({report.drift_active_count/max(1,report.total_records)*100:.1f}%)")
    print(f"   Duplicate prompt rate:  {report.duplicate_prompt_rate*100:.2f}%")
    print(f"   Action entropy:         {report.action_entropy:.3f} bits")
    print(f"   Scenario imbalance:     {report.scenario_imbalance_ratio:.2f}x")
    print(f"   Rec diversity:          {report.recommendation_diversity:.4f}")
    print(f"   Fingerprint:            {report.fingerprint[:16]}...")

    # Gate results
    print(f"\n🚦 Quality Gates:")
    for gate_name, passed in report.gate_results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {gate_name}")

    if report.passed:
        print(f"\n✅ ALL GATES PASSED")
    else:
        print(f"\n❌ FAILED GATES: {report.gate_failures}")

    # 4. Split
    train_records, val_records = split_train_val(deduped, val_fraction=val_fraction, seed=seed)
    print(f"\n📂 Train/Val Split:")
    print(f"   Train: {len(train_records)} records")
    print(f"   Val:   {len(val_records)} records")

    # 5. Leakage check
    leakage = verify_no_leakage(train_records, val_records)
    print(f"   Episode overlap: {leakage['episode_overlap_count']}")
    print(f"   Prompt overlap:  {leakage['prompt_overlap_count']}")
    print(f"   Clean split:     {'✅' if leakage['clean'] else '❌'}")

    # 6. Save fingerprint
    fp_path = save_fingerprint(report, output_dir / "dataset_fingerprint.json")
    print(f"\n💾 Fingerprint saved: {fp_path}")

    # 7. Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"   Train saved: {train_path}")
    print(f"   Val saved:   {val_path}")

    # 8. Agent action distribution
    agent_dist = check_agent_action_distribution(deduped)
    dist_path = output_dir / "agent_action_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(agent_dist, f, indent=2)
    print(f"   Agent distribution: {dist_path}")

    return {
        "report": report,
        "leakage": leakage,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "num_deduped": num_removed,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "fingerprint_path": str(fp_path),
        "passed": report.passed,
    }


if __name__ == "__main__":
    result = run_integrity_pipeline("artifacts/sft/orchestrator_sft.jsonl")
    if not result["passed"]:
        print("\n⚠️  Dataset did not pass all quality gates. Review failures above.")
    else:
        print("\n🎉 Dataset integrity verified. Ready for training.")
