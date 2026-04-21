# aic/agents/incident_commander_agent.py
"""
Incident Commander Agent — sets strategic priority mode for the orchestrator.

The Commander sits above the orchestrator: it decides *what matters most*
given the current situation, then passes a weighting vector that the
orchestrator uses to bias candidate selection.

Priority Modes:
  fastest_recovery    — minimize MTTR at all costs
  safest_recovery     — minimize unsafe actions; prefer low blast radius
  protect_data        — prioritise DB/replication metrics
  minimize_user_impact — prioritise error_rate and SLA compliance
  contain_compromise  — security-first; quarantine over availability

Mode transitions happen automatically based on:
  - Root cause hypothesis (security → contain_compromise)
  - SLA time pressure (late steps → fastest_recovery)
  - Business impact severity (P1 → minimize_user_impact or fastest_recovery)
  - Metric profile (high DB coupling → protect_data)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aic.utils.constants import METRIC_TARGETS, SLA_STEPS


# ─────────────────────────────────────────────────────────────────────────
# Priority mode definitions
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PriorityMode:
    name: str
    display_name: str
    description: str
    metric_weights: dict[str, float]   # higher = more important to fix this metric
    prefer_low_blast_radius: bool
    urgency: str                        # "low" | "medium" | "high" | "critical"
    emoji: str


PRIORITY_MODES: dict[str, PriorityMode] = {
    "fastest_recovery": PriorityMode(
        name="fastest_recovery",
        display_name="⚡ Fastest Recovery",
        description="Minimize MTTR — accept higher risk if needed.",
        metric_weights={
            "p95_latency_ms": 2.5,
            "error_rate_pct": 2.0,
            "throughput_rps": 2.0,
            "sla_compliance_pct": 1.5,
        },
        prefer_low_blast_radius=False,
        urgency="critical",
        emoji="⚡",
    ),
    "safest_recovery": PriorityMode(
        name="safest_recovery",
        display_name="🛡️ Safest Recovery",
        description="Zero unsafe actions. Prefer low-blast-radius interventions.",
        metric_weights={
            "error_rate_pct": 1.5,
            "p95_latency_ms": 1.0,
            "pod_restarts": 2.0,
        },
        prefer_low_blast_radius=True,
        urgency="medium",
        emoji="🛡️",
    ),
    "protect_data": PriorityMode(
        name="protect_data",
        display_name="💾 Protect Data Integrity",
        description="DB and replication health above all else.",
        metric_weights={
            "db_latency_ms": 3.0,
            "replication_lag_ms": 3.0,
            "conn_pool_pct": 2.0,
        },
        prefer_low_blast_radius=True,
        urgency="high",
        emoji="💾",
    ),
    "minimize_user_impact": PriorityMode(
        name="minimize_user_impact",
        display_name="👥 Minimize User Impact",
        description="Protect SLA compliance and user-facing error rate.",
        metric_weights={
            "sla_compliance_pct": 3.0,
            "error_rate_pct": 2.5,
            "throughput_rps": 2.0,
            "p95_latency_ms": 1.5,
        },
        prefer_low_blast_radius=False,
        urgency="high",
        emoji="👥",
    ),
    "contain_compromise": PriorityMode(
        name="contain_compromise",
        display_name="🔒 Contain Compromise",
        description="Security-first. Quarantine over availability. Stop lateral movement.",
        metric_weights={
            "error_rate_pct": 2.0,
            "throughput_rps": 1.0,
            "p95_latency_ms": 1.0,
        },
        prefer_low_blast_radius=True,
        urgency="critical",
        emoji="🔒",
    ),
}


@dataclass
class CommanderDecision:
    """Output of the Incident Commander for one step."""
    mode: PriorityMode
    strategic_brief: str
    mode_reason: str
    candidate_weights: dict[str, float]   # agent_name → score adjustment
    prefer_low_blast_radius: bool
    step: int
    mode_history: list[str] = field(default_factory=list)


class IncidentCommanderAgent:
    """
    Strategic layer above the orchestrator.

    Assesses situation every step and outputs a PriorityMode + candidate
    weighting adjustments that bias orchestrator selection.
    """

    def __init__(self) -> None:
        self._mode_history: list[str] = []
        self._current_mode: Optional[PriorityMode] = None

    def reset(self) -> None:
        self._mode_history.clear()
        self._current_mode = None

    def assess_and_command(
        self,
        step: int,
        sla_remaining: int,
        current_metrics: dict[str, float],
        root_cause_hypothesis: Optional[dict] = None,
        business_severity: Optional[str] = None,
    ) -> CommanderDecision:
        """
        Determine the strategic priority mode for this step.

        Decision logic (in priority order):
          1. Security compromise detected → contain_compromise
          2. Late stage (≤ 5 steps left) → fastest_recovery
          3. Hypothesis = Schema Migration / DB → protect_data
          4. P1 business severity + user-facing → minimize_user_impact
          5. Default: balanced fastest_recovery
        """
        mode_name = "fastest_recovery"
        mode_reason = "Default: no special conditions detected."

        scenario_name = (root_cause_hypothesis or {}).get("scenario_name", "").lower()
        hyp_confidence = (root_cause_hypothesis or {}).get("confidence", 0.0)

        # ── Rule 1: Security compromise ──────────────────────────────────
        if (
            "credential" in scenario_name
            and hyp_confidence > 0.35
        ):
            mode_name = "contain_compromise"
            mode_reason = (
                f"Root cause analyst confident in 'Credential Compromise' "
                f"(confidence={hyp_confidence:.2f}). Switching to security-first mode."
            )

        # ── Rule 2: SLA time pressure ────────────────────────────────────
        elif sla_remaining <= 5:
            mode_name = "fastest_recovery"
            mode_reason = (
                f"Only {sla_remaining} steps remain. Maximum urgency — "
                f"speed over safety."
            )

        # ── Rule 3: DB / Schema heavy scenario ───────────────────────────
        elif (
            any(k in scenario_name for k in ("schema", "migration", "db", "cache"))
            and hyp_confidence > 0.3
            and current_metrics.get("replication_lag_ms", 0) > 200
        ):
            mode_name = "protect_data"
            mode_reason = (
                f"DB/schema root cause detected (confidence={hyp_confidence:.2f}) "
                f"with replication_lag={current_metrics.get('replication_lag_ms', 0):.0f}ms. "
                f"Data integrity priority."
            )

        # ── Rule 4: P1 business severity + high error rate ───────────────
        elif (
            business_severity in ("P1", "P2")
            and current_metrics.get("error_rate_pct", 0) > 10.0
        ):
            mode_name = "minimize_user_impact"
            mode_reason = (
                f"Business severity={business_severity} with "
                f"error_rate={current_metrics.get('error_rate_pct', 0):.1f}%. "
                f"Protecting user-facing SLA."
            )

        # ── Rule 5: Queue cascade pattern ────────────────────────────────
        elif (
            "queue" in scenario_name
            and current_metrics.get("queue_depth", 0) > 500
        ):
            mode_name = "fastest_recovery"
            mode_reason = (
                f"Queue cascade in progress (queue_depth="
                f"{current_metrics.get('queue_depth', 0):.0f}). Rapid draining priority."
            )

        # ── Default: check metric profile for protect_data signal ────────
        elif (
            current_metrics.get("db_latency_ms", 0) > 1000
            and current_metrics.get("replication_lag_ms", 0) > 300
        ):
            mode_name = "protect_data"
            mode_reason = (
                f"High DB latency ({current_metrics.get('db_latency_ms', 0):.0f}ms) "
                f"and replication lag ({current_metrics.get('replication_lag_ms', 0):.0f}ms). "
                f"Switching to data protection mode."
            )

        mode = PRIORITY_MODES[mode_name]
        self._current_mode = mode
        self._mode_history.append(mode_name)

        # Build candidate weights based on mode's metric_weights
        # Any agent whose target_metrics overlap with mode's priority gets a boost
        candidate_weights: dict[str, float] = {}

        # Brief
        strategic_brief = (
            f"{mode.emoji} Commander: {mode.display_name} | "
            f"Step {step}/{SLA_STEPS} | {mode.urgency.upper()} | "
            f"{mode_reason[:120]}"
        )

        decision = CommanderDecision(
            mode=mode,
            strategic_brief=strategic_brief,
            mode_reason=mode_reason,
            candidate_weights=candidate_weights,
            prefer_low_blast_radius=mode.prefer_low_blast_radius,
            step=step,
            mode_history=list(self._mode_history),
        )
        return decision

    def score_candidate(
        self,
        agent_name: str,
        target_metrics: list[str],
        decision: CommanderDecision,
    ) -> float:
        """
        Return a priority bonus score for a candidate recommendation
        based on the current commander decision.

        Higher = more aligned with current strategy.
        """
        bonus = 0.0
        for metric in target_metrics:
            weight = decision.mode.metric_weights.get(metric, 0.0)
            bonus += weight
        return bonus

    def get_mode_history_summary(self) -> dict[str, int]:
        """Return counts of each mode used during the episode."""
        counts: dict[str, int] = {}
        for m in self._mode_history:
            counts[m] = counts.get(m, 0) + 1
        return counts

    @property
    def current_mode(self) -> Optional[PriorityMode]:
        return self._current_mode
