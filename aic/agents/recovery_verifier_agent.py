# aic/agents/recovery_verifier_agent.py
"""
Recovery Verifier Agent — enterprise-grade safety gate.

This agent does NOT propose actions. It evaluates them.
It receives the Orchestrator's chosen recommendation and returns a
verification report with approved/denied status.

Veto conditions (approved=False):
1. risk_score > 0.8 (too risky regardless of rollback)
2. blast_radius == "high" AND rollback_plan is empty/missing
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_VERIFIER


@dataclass
class VerifierReport:
    """Output of the Recovery Verifier's evaluation."""
    approved: bool
    risk_score: float
    blast_radius: str
    verification_reasoning: str
    vetoed_action: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Safe minimal action used when all recommendations are vetoed
SAFE_MINIMAL_ACTION = SubAgentRecommendation(
    agent_name=AGENT_VERIFIER,
    action="Wait and Observe — all recommendations vetoed by safety verifier",
    reasoning="All proposed actions were deemed too risky or lacked adequate rollback plans. Defaulting to observation mode to prevent catastrophic intervention.",
    confidence=1.0,
    target_metrics=[],
    expected_impact={},
    bid=0.0,
    action_cost=0.1,
    risk_score=0.0,
    blast_radius="low",
    rollback_plan="No action taken — no rollback needed",
)


class RecoveryVerifierAgent:
    """
    Safety gate that evaluates recommendations before execution.

    Does not extend BaseSubAgent — it evaluates, not proposes.
    """

    # Risk threshold above which actions are always vetoed
    RISK_THRESHOLD: float = 0.8

    def __init__(self, risk_threshold: float = 0.8):
        self.RISK_THRESHOLD = risk_threshold
        self._veto_log: list[VerifierReport] = []

    def reset(self) -> None:
        """Clear veto log for a new episode."""
        self._veto_log.clear()

    def verify(
        self,
        recommendation: SubAgentRecommendation,
        current_metrics: Optional[dict[str, float]] = None,
    ) -> VerifierReport:
        """
        Evaluate a recommendation for safety.

        Args:
            recommendation: The action to evaluate.
            current_metrics: Optional current system metrics for context.

        Returns:
            VerifierReport with approved status and reasoning.
        """
        risk = recommendation.risk_score
        blast = recommendation.blast_radius
        rollback = recommendation.rollback_plan.strip() if recommendation.rollback_plan else ""

        reasons: list[str] = []

        # Check 1: Risk score too high
        if risk > self.RISK_THRESHOLD:
            reasons.append(
                f"Risk score {risk:.2f} exceeds threshold {self.RISK_THRESHOLD}. "
                f"Action '{recommendation.action[:80]}...' is too dangerous."
            )

        # Check 2: High blast radius without rollback
        if blast == "high" and not rollback:
            reasons.append(
                f"Blast radius is 'high' but no rollback plan provided. "
                f"Cannot approve irreversible high-impact action."
            )

        approved = len(reasons) == 0

        report = VerifierReport(
            approved=approved,
            risk_score=risk,
            blast_radius=blast,
            verification_reasoning=(
                "Action approved — within safety parameters."
                if approved
                else " | ".join(reasons)
            ),
            vetoed_action=None if approved else recommendation.action,
        )

        if not approved:
            self._veto_log.append(report)

        return report

    def get_veto_log(self) -> list[dict]:
        """Return all veto reports from this episode."""
        return [r.to_dict() for r in self._veto_log]

    def get_safe_minimal_action(self) -> SubAgentRecommendation:
        """Return the safe fallback action."""
        return SAFE_MINIMAL_ACTION
