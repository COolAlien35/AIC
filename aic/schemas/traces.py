# aic/schemas/traces.py
"""
Pydantic V2 models for explanation traces, agent recommendations,
and orchestrator actions.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExplanationTrace(BaseModel):
    """
    Structured explanation trace emitted by orchestrator every step.
    Stored in trace history, scored by reward engine.
    """
    step: int = Field(ge=0, le=20)
    followed_agent: Optional[str] = Field(
        default=None,
        description="Agent whose recommendation was ultimately followed for this step",
    )
    action_taken: str = Field(min_length=1, max_length=500)
    reasoning: str = Field(min_length=10, max_length=2000)
    sub_agent_trust_scores: dict[str, float] = Field(
        description="Trust score per agent, keys are agent names, values in [0, 1]"
    )
    override_applied: bool
    override_reason: Optional[str] = Field(default=None, max_length=500)
    predicted_2step_impact: dict[str, float] = Field(
        description=(
            "Predicted metric changes 2 steps ahead. "
            "Keys are metric names, values are expected deltas."
        )
    )
    schema_drift_detected: bool
    schema_drift_field: Optional[str] = Field(default=None)
    verifier_report: Optional[dict] = Field(
        default=None,
        description="Recovery Verifier report: approved, risk_score, verification_reasoning",
    )
    root_cause_hypothesis: Optional[dict] = Field(
        default=None,
        description="Root cause analyst hypothesis: scenario_name, confidence",
    )
    runbook_evidence: Optional[dict] = Field(
        default=None,
        description="Knowledge agent evidence: incident_id, remediation, confidence",
    )
    simulation_scores: Optional[dict] = Field(
        default=None,
        description="Counterfactual simulation results for the chosen action",
    )
    # ── New: Debate / Commander / Observability ──────────────────────────
    debate_transcript: Optional[list[dict]] = Field(
        default=None,
        description="Structured debate transcript for this step",
    )
    debate_changed_selection: Optional[bool] = Field(
        default=None,
        description="Whether the structured debate altered the pre-debate top selection",
    )
    commander_mode: Optional[str] = Field(
        default=None,
        description="Incident Commander strategic priority mode name",
    )
    commander_brief: Optional[str] = Field(
        default=None,
        description="One-line strategic brief from the Incident Commander",
    )
    observability_report: Optional[dict] = Field(
        default=None,
        description="ObservabilityAgent telemetry trustworthiness assessment",
    )
    business_impact_snapshot: Optional[dict] = Field(
        default=None,
        description="BusinessImpact snapshot at this step",
    )

    @field_validator("sub_agent_trust_scores")
    @classmethod
    def validate_trust_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for agent, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Trust score for {agent} must be in [0, 1], got {score}"
                )
        return v

    @field_validator("override_reason")
    @classmethod
    def override_reason_required_when_overriding(
        cls, v: Optional[str], info
    ) -> Optional[str]:
        if info.data.get("override_applied") and not v:
            raise ValueError(
                "override_reason must be provided when override_applied is True"
            )
        return v

    @field_validator("schema_drift_field")
    @classmethod
    def drift_field_required_when_detected(
        cls, v: Optional[str], info
    ) -> Optional[str]:
        if info.data.get("schema_drift_detected") and not v:
            raise ValueError(
                "schema_drift_field must be provided when schema_drift_detected is True"
            )
        return v


class SubAgentRecommendation(BaseModel):
    """A recommendation from one sub-agent to the orchestrator."""
    agent_name: str
    action: str = Field(min_length=1, max_length=300)
    reasoning: str = Field(min_length=5, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    target_metrics: list[str] = Field(
        description="Metric names this action targets"
    )
    expected_impact: dict[str, float] = Field(
        default_factory=dict,
        description="Predicted metric changes from this action",
    )
    bid: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Competitive bid strength for limited intervention budget (0..1).",
    )
    action_cost: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Intervention cost units for this action (higher = consumes more budget).",
    )
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Risk score of this action (0.0=safe, 1.0=dangerous)",
    )
    blast_radius: str = Field(
        default="low",
        description="Blast radius: 'low', 'medium', or 'high'",
    )
    rollback_plan: str = Field(
        default="",
        max_length=500,
        description="Description of how to rollback this action if it fails",
    )
    # ── Debate layer fields (populated by DebateCoordinator) ─────────────
    criticisms: list[str] = Field(
        default_factory=list,
        description="Criticisms of this recommendation from other agents",
    )
    supports: list[str] = Field(
        default_factory=list,
        description="Agents/reasons that support this recommendation",
    )
    rebuttals: list[str] = Field(
        default_factory=list,
        description="Rebuttals to criticisms of this recommendation",
    )
    debate_round: int = Field(
        default=0,
        description="Debate round number this recommendation participated in",
    )


class OrchestratorAction(BaseModel):
    """Parsed output of the orchestrator's decision for one step."""
    action_description: str = Field(min_length=1, max_length=500)
    target_service: str = Field(description="One of: db, infra, app")
    action_deltas: dict[str, float] = Field(
        description=(
            "Expected metric changes from this action. "
            "Used to update world state."
        )
    )
    trust_override: Optional[str] = Field(
        default=None,
        description="Name of agent being overridden, if any",
    )
    explanation_trace: ExplanationTrace
