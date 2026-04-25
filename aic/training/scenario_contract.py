# aic/training/scenario_contract.py
"""
Phase 2.2 — Scenario Realism Contract.

Replaces the ambiguous SCENARIO_TO_FAULT_MODE mapping with the canonical
6-scenario registry. Each scenario has distinct dynamics, verified fault
vectors, and rich metadata tags.

Also provides:
  - Scenario contract table for documentation
  - Held-out stress scenario definitions
  - Per-sample metadata tagging (scenario_name, fault_mode, difficulty_tier,
    adversarial_intensity)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic.env.scenario_registry import SCENARIO_REGISTRY, ScenarioDefinition


# ═══════════════════════════════════════════════════════════════════════════
# Canonical Scenario Mapping (replaces SCENARIO_TO_FAULT_MODE)
# ═══════════════════════════════════════════════════════════════════════════

# Each entry maps a scenario_id to its canonical name and training metadata.
# Unlike the old SCENARIO_TO_FAULT_MODE list, every scenario here has DISTINCT
# fault dynamics defined in the scenario registry — no aliased remaps.

@dataclass
class ScenarioTrainingMeta:
    """Training metadata for a single scenario."""
    scenario_id: int
    scenario_name: str
    root_cause_node: str
    fault_injector_mode: str        # Valid FaultInjector mode for this scenario
    severity: str
    difficulty_tier: str            # "easy", "medium", "hard"
    adversarial_intensity: str      # "low", "medium", "high"
    has_telemetry_corruption: bool
    corruption_types: list[str]
    description: str


# Maps each scenario's root_cause_node to the most appropriate FaultInjector
# fault_mode. The FaultInjector only accepts 4 modes; this bridges the
# scenario registry's topology-based naming to the injector's 4-mode system.
ROOT_NODE_TO_FAULT_MODE: dict[str, str] = {
    "cache": "db_connection_saturation",   # Cache stampede → DB pressure
    "app": "cascading_failure",            # App faults cascade across services
    "gateway": "cascading_failure",        # Regional outage = cascading
    "db": "db_connection_saturation",      # DB migration → DB saturation
    "queue": "network_storm",              # Queue cascade → network/queue pressure
}


def _classify_difficulty(scenario: ScenarioDefinition) -> str:
    """Classify scenario difficulty based on fault pressure magnitude."""
    total_pressure = sum(abs(v) for v in scenario.initial_fault_vector.values())
    drift_pressure = sum(abs(v) for v in scenario.per_step_drift.values())
    combined = total_pressure + drift_pressure * 10  # weight progressive drift

    if combined < 200:
        return "easy"
    elif combined < 400:
        return "medium"
    else:
        return "hard"


def _classify_adversarial_intensity(scenario: ScenarioDefinition) -> str:
    """Classify adversarial intensity based on corruption rules and severity."""
    n_corruption = len(scenario.telemetry_corruption_rules)
    if scenario.severity == "P1" and n_corruption >= 1:
        return "high"
    elif scenario.severity == "P1" or n_corruption >= 1:
        return "medium"
    else:
        return "low"


def build_scenario_training_meta() -> dict[int, ScenarioTrainingMeta]:
    """Build the canonical training metadata for all 6 scenarios."""
    metas = {}
    for sid, scenario in SCENARIO_REGISTRY.items():
        corruption_types = [
            r.corruption_type for r in scenario.telemetry_corruption_rules
        ]
        fault_mode = ROOT_NODE_TO_FAULT_MODE.get(
            scenario.root_cause_node, "cascading_failure"
        )
        metas[sid] = ScenarioTrainingMeta(
            scenario_id=sid,
            scenario_name=scenario.name,
            root_cause_node=scenario.root_cause_node,
            fault_injector_mode=fault_mode,
            severity=scenario.severity,
            difficulty_tier=_classify_difficulty(scenario),
            adversarial_intensity=_classify_adversarial_intensity(scenario),
            has_telemetry_corruption=len(corruption_types) > 0,
            corruption_types=corruption_types,
            description=scenario.description,
        )
    return metas


SCENARIO_TRAINING_META = build_scenario_training_meta()

# Canonical ordered list for generation iteration (replaces SCENARIO_TO_FAULT_MODE)
CANONICAL_SCENARIO_IDS = sorted(SCENARIO_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Scenario Contract Table
# ═══════════════════════════════════════════════════════════════════════════

def print_scenario_contract_table() -> None:
    """Print a formatted scenario contract table for documentation."""
    print("=" * 100)
    print(f"{'ID':>3} | {'Name':<28} | {'Root':>8} | {'Sev':>3} | {'Diff':>6} | {'Adv':>6} | {'Corruption':<20}")
    print("-" * 100)
    for sid in CANONICAL_SCENARIO_IDS:
        meta = SCENARIO_TRAINING_META[sid]
        corr = ", ".join(meta.corruption_types) if meta.corruption_types else "none"
        print(
            f"{meta.scenario_id:>3} | {meta.scenario_name:<28} | {meta.root_cause_node:>8} | "
            f"{meta.severity:>3} | {meta.difficulty_tier:>6} | {meta.adversarial_intensity:>6} | {corr:<20}"
        )
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════
# Sample Metadata Tagging
# ═══════════════════════════════════════════════════════════════════════════

def tag_sample_metadata(
    scenario_id: int,
    episode_id: int,
    step: int,
    override_applied: bool,
    has_adversarial_candidate: bool,
    schema_drift_active: bool,
    drift_type: str | None,
) -> dict[str, Any]:
    """Build rich metadata tags for a single SFT sample.

    Tags include:
      - scenario_name
      - fault_mode (root_cause_node)
      - difficulty_tier
      - adversarial_intensity
      - has_adversarial
      - schema_drift
      - drift_type

    These are required by Phase 2.2 for downstream analysis.
    """
    meta = SCENARIO_TRAINING_META[scenario_id]
    return {
        "scenario_name": meta.scenario_name,
        "scenario_id": scenario_id,
        "fault_mode": meta.root_cause_node,
        "difficulty_tier": meta.difficulty_tier,
        "adversarial_intensity": meta.adversarial_intensity,
        "has_adversarial": bool(override_applied or has_adversarial_candidate),
        "schema_drift": schema_drift_active,
        "drift_type": drift_type,
        "has_telemetry_corruption": meta.has_telemetry_corruption,
        "severity": meta.severity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Held-Out Stress Scenarios
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StressScenarioConfig:
    """Configuration for a held-out stress scenario."""
    name: str
    description: str
    base_scenario_id: int          # Which registry scenario to derive from
    drift_type_override: str       # Force a specific drift type
    adversarial_accuracy: float    # Override adversary correct probability
    budget_multiplier: float       # Multiply episode budget
    extra_fault_pressure: dict[str, float]  # Additional per-step pressure


# Held-out stress scenarios for robustness evaluation.
# These are deliberately harder than training scenarios to test generalization.
STRESS_SCENARIOS: list[StressScenarioConfig] = [
    StressScenarioConfig(
        name="double_fault_stampede",
        description="Cache stampede + simultaneous field rename — tests multi-corruption resilience",
        base_scenario_id=0,  # Cache Stampede
        drift_type_override="field_rename",
        adversarial_accuracy=0.7,    # Adversary is MORE accurate (harder to override)
        budget_multiplier=0.5,       # Half budget
        extra_fault_pressure={"db_latency_ms": +40.0, "error_rate_pct": +1.0},
    ),
    StressScenarioConfig(
        name="blind_regional_outage",
        description="Regional outage with ALL telemetry blacked out — maximum partial observability",
        base_scenario_id=2,  # Regional Outage
        drift_type_override="silent_null",
        adversarial_accuracy=0.3,    # Adversary is unreliable
        budget_multiplier=0.7,
        extra_fault_pressure={"p95_latency_ms": +100.0, "queue_depth": +50.0},
    ),
    StressScenarioConfig(
        name="adversary_dominant_migration",
        description="Schema migration where adversary has highest confidence — trust calibration stress",
        base_scenario_id=3,  # Schema Migration Disaster
        drift_type_override="unit_shift",
        adversarial_accuracy=0.8,    # Adversary is very accurate (should be trusted)
        budget_multiplier=1.0,
        extra_fault_pressure={"replication_lag_ms": +20.0},
    ),
    StressScenarioConfig(
        name="budget_starved_compromise",
        description="Credential compromise with almost no budget — forced triage under scarcity",
        base_scenario_id=4,  # Credential Compromise
        drift_type_override="field_rename",
        adversarial_accuracy=0.5,
        budget_multiplier=0.3,       # Extremely limited budget
        extra_fault_pressure={"error_rate_pct": +2.0, "throughput_rps": -30.0},
    ),
]


def get_stress_scenario_names() -> list[str]:
    """Return names of all held-out stress scenarios."""
    return [s.name for s in STRESS_SCENARIOS]


if __name__ == "__main__":
    print("\n🏗️  Scenario Contract Table\n")
    print_scenario_contract_table()

    print(f"\n🔬 Held-Out Stress Scenarios: {len(STRESS_SCENARIOS)}")
    for s in STRESS_SCENARIOS:
        print(f"   - {s.name}: {s.description}")
