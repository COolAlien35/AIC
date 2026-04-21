# aic/env/scenario_registry.py
"""
Scenario Registry — 6 "Brutal Scenarios" that replace the generic FaultInjector.

Each scenario defines:
    - root_cause_node: which topology node is the origin
    - initial_fault_vector: CONSTANT per-step pressure (requires orchestrator action to negate)
    - per_step_drift: additional progressive degradation
    - telemetry_corruption_rules: field renames, NaN blackouts, or unit shifts
    - severity: expected incident severity

ScenarioEngine has the same interface as FaultInjector for drop-in replacement.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TelemetryCorruptionRule:
    """A single telemetry corruption rule applied by a scenario."""
    corruption_type: str     # "nan_blackout", "field_rename", "unit_shift"
    target_field: str        # metric field to corrupt
    renamed_field: Optional[str] = None    # for field_rename
    scale_factor: Optional[float] = None   # for unit_shift
    start_step: int = 0      # step at which corruption begins
    end_step: int = 20       # step at which corruption ends (exclusive)


@dataclass
class ScenarioDefinition:
    """Full definition of a brutal scenario."""
    scenario_id: int
    name: str
    description: str
    root_cause_node: str      # topology node where the fault originates
    severity: str             # expected severity: "P1", "P2", "P3", "P4"
    initial_fault_vector: dict[str, float]  # constant per-step pressure on metrics
    per_step_drift: dict[str, float]        # additional progressive degradation
    telemetry_corruption_rules: list[TelemetryCorruptionRule] = field(
        default_factory=list
    )


# ═══════════════════════════════════════════════════════════════════════════
# The 6 Brutal Scenarios
# ═══════════════════════════════════════════════════════════════════════════

SCENARIO_REGISTRY: dict[int, ScenarioDefinition] = {
    0: ScenarioDefinition(
        scenario_id=0,
        name="Cache Stampede",
        description=(
            "Redis cluster eviction storm. All cache keys expire simultaneously, "
            "causing a thundering herd of requests directly to the DB. "
            "Connection pool saturates, latency spikes, queue backs up."
        ),
        root_cause_node="cache",
        severity="P1",
        initial_fault_vector={
            "db_latency_ms": +80.0,
            "conn_pool_pct": +1.5,
            "queue_depth": +60.0,
            "error_rate_pct": +0.8,
            "p95_latency_ms": +50.0,
        },
        per_step_drift={
            "db_latency_ms": +5.0,
            "queue_depth": +10.0,
        },
        telemetry_corruption_rules=[],
    ),
    1: ScenarioDefinition(
        scenario_id=1,
        name="Canary Failure",
        description=(
            "A canary deployment with a critical bug receives 10% of traffic. "
            "Error rate spikes for a subset of users. Observability is partially "
            "blinded — error_rate reporting goes dark for 4 steps."
        ),
        root_cause_node="app",
        severity="P2",
        initial_fault_vector={
            "error_rate_pct": +2.0,
            "p95_latency_ms": +100.0,
            "throughput_rps": -30.0,
        },
        per_step_drift={
            "error_rate_pct": +0.5,
            "p95_latency_ms": +15.0,
        },
        telemetry_corruption_rules=[
            TelemetryCorruptionRule(
                corruption_type="nan_blackout",
                target_field="error_rate_pct",
                start_step=5,
                end_step=9,
            ),
        ],
    ),
    2: ScenarioDefinition(
        scenario_id=2,
        name="Regional Outage",
        description=(
            "An entire availability zone goes offline. All services degrade "
            "at 60% cascading intensity. Network I/O monitoring is blacked out."
        ),
        root_cause_node="gateway",
        severity="P1",
        initial_fault_vector={
            "mem_pct": +1.2,
            "cpu_pct": +0.9,
            "pod_restarts": +0.18,
            "p95_latency_ms": +30.0,
            "conn_pool_pct": +0.9,
            "db_latency_ms": +48.0,
            "replication_lag_ms": +12.0,
            "net_io_mbps": +18.0,
            "error_rate_pct": +0.6,
            "queue_depth": +36.0,
        },
        per_step_drift={
            "error_rate_pct": +0.3,
            "p95_latency_ms": +20.0,
        },
        telemetry_corruption_rules=[
            TelemetryCorruptionRule(
                corruption_type="nan_blackout",
                target_field="net_io_mbps",
                start_step=0,
                end_step=20,
            ),
        ],
    ),
    3: ScenarioDefinition(
        scenario_id=3,
        name="Schema Migration Disaster",
        description=(
            "A botched database migration locks critical tables and introduces "
            "schema incompatibility. DB latency explodes, replication breaks. "
            "The db_latency_ms field is silently renamed in telemetry."
        ),
        root_cause_node="db",
        severity="P1",
        initial_fault_vector={
            "db_latency_ms": +120.0,
            "replication_lag_ms": +30.0,
            "conn_pool_pct": +2.0,
            "error_rate_pct": +1.0,
        },
        per_step_drift={
            "db_latency_ms": +15.0,
            "replication_lag_ms": +5.0,
        },
        telemetry_corruption_rules=[
            TelemetryCorruptionRule(
                corruption_type="field_rename",
                target_field="db_latency_ms",
                renamed_field="db_latency",
                start_step=3,
                end_step=20,
            ),
        ],
    ),
    4: ScenarioDefinition(
        scenario_id=4,
        name="Credential Compromise",
        description=(
            "Leaked API credentials trigger a brute-force attack. Error rate "
            "spikes from auth failures, throughput drops as the WAF throttles "
            "traffic. SLA compliance monitoring is compromised."
        ),
        root_cause_node="app",
        severity="P1",
        initial_fault_vector={
            "error_rate_pct": +3.0,
            "throughput_rps": -50.0,
            "p95_latency_ms": +80.0,
            "cpu_pct": +1.0,
        },
        per_step_drift={
            "error_rate_pct": +0.8,
            "throughput_rps": -10.0,
        },
        telemetry_corruption_rules=[
            TelemetryCorruptionRule(
                corruption_type="nan_blackout",
                target_field="sla_compliance_pct",
                start_step=2,
                end_step=20,
            ),
        ],
    ),
    5: ScenarioDefinition(
        scenario_id=5,
        name="Queue Cascade",
        description=(
            "Message queue consumer group rebalance storm. Queue depth explodes, "
            "backpressure causes p95 latency spikes and error rate increases. "
            "Queue depth telemetry has a unit shift (reports in thousands)."
        ),
        root_cause_node="queue",
        severity="P2",
        initial_fault_vector={
            "queue_depth": +80.0,
            "p95_latency_ms": +60.0,
            "error_rate_pct": +1.5,
            "cpu_pct": +0.5,
        },
        per_step_drift={
            "queue_depth": +15.0,
            "p95_latency_ms": +10.0,
        },
        telemetry_corruption_rules=[
            TelemetryCorruptionRule(
                corruption_type="unit_shift",
                target_field="queue_depth",
                scale_factor=0.001,
                start_step=0,
                end_step=20,
            ),
        ],
    ),
}


def get_scenario(scenario_id: int) -> ScenarioDefinition:
    """Look up a scenario by ID."""
    if scenario_id not in SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown scenario_id {scenario_id}. "
            f"Valid IDs: {sorted(SCENARIO_REGISTRY.keys())}"
        )
    return copy.deepcopy(SCENARIO_REGISTRY[scenario_id])


def list_scenarios() -> list[dict[str, Any]]:
    """Return a summary list of all registered scenarios."""
    return [
        {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "root_cause_node": s.root_cause_node,
            "severity": s.severity,
        }
        for s in SCENARIO_REGISTRY.values()
    ]


class ScenarioEngine:
    """
    Drop-in replacement for FaultInjector that loads from the scenario registry.

    Same interface: get_contributions(step) → dict[str, float]
    Additional methods for telemetry corruption.
    """

    # Mild decay factor — scenarios require orchestrator intervention
    DECAY_BASE: float = 0.98

    def __init__(self, scenario_id: int):
        self.scenario = get_scenario(scenario_id)
        self.scenario_id = scenario_id

    def get_contributions(self, step: int) -> dict[str, float]:
        """
        Return per-metric fault contributions for the given step.

        Combines constant pressure (initial_fault_vector with mild decay)
        and progressive drift (per_step_drift).
        """
        decay = self.DECAY_BASE ** step
        contributions: dict[str, float] = {}

        # Constant pressure from initial fault vector (with mild decay)
        for metric, rate in self.scenario.initial_fault_vector.items():
            contributions[metric] = rate * decay

        # Additional progressive drift
        for metric, drift in self.scenario.per_step_drift.items():
            contributions[metric] = contributions.get(metric, 0.0) + drift

        return contributions

    def get_telemetry_mask(self, step: int) -> set[str]:
        """
        Return set of field names that should be masked (set to NaN)
        at the given step.
        """
        masked: set[str] = set()
        for rule in self.scenario.telemetry_corruption_rules:
            if rule.corruption_type == "nan_blackout":
                if rule.start_step <= step < rule.end_step:
                    masked.add(rule.target_field)
        return masked

    def get_telemetry_renames(self, step: int) -> dict[str, str]:
        """
        Return dict of old_field_name → new_field_name for active renames
        at the given step.
        """
        renames: dict[str, str] = {}
        for rule in self.scenario.telemetry_corruption_rules:
            if rule.corruption_type == "field_rename" and rule.renamed_field:
                if rule.start_step <= step < rule.end_step:
                    renames[rule.target_field] = rule.renamed_field
        return renames

    def get_telemetry_unit_shifts(self, step: int) -> dict[str, float]:
        """
        Return dict of field_name → scale_factor for active unit shifts
        at the given step.
        """
        shifts: dict[str, float] = {}
        for rule in self.scenario.telemetry_corruption_rules:
            if rule.corruption_type == "unit_shift" and rule.scale_factor is not None:
                if rule.start_step <= step < rule.end_step:
                    shifts[rule.target_field] = rule.scale_factor
        return shifts

    def apply_telemetry_corruption(
        self, observation: dict[str, Any], step: int
    ) -> dict[str, Any]:
        """
        Apply all telemetry corruption rules to an observation dict.

        Handles NaN blackouts, field renames, and unit shifts.
        Returns a modified copy.
        """
        result = observation.copy()

        # Apply NaN blackouts
        for field_name in self.get_telemetry_mask(step):
            if field_name in result:
                result[field_name] = float("nan")

        # Apply field renames
        for old_name, new_name in self.get_telemetry_renames(step).items():
            if old_name in result:
                result[new_name] = result.pop(old_name)

        # Apply unit shifts
        for field_name, factor in self.get_telemetry_unit_shifts(step).items():
            if field_name in result and result[field_name] is not None:
                try:
                    result[field_name] = result[field_name] * factor
                except (TypeError, ValueError):
                    pass

        return result

    def get_scenario_name(self) -> str:
        """Return the name of the active scenario."""
        return self.scenario.name

    def get_root_cause_node(self) -> str:
        """Return the root cause topology node."""
        return self.scenario.root_cause_node
