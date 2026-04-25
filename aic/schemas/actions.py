"""Structured action schemas for orchestrator training and environment stepping."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CandidateRecommendation(BaseModel):
    """Serialized recommendation shown to the orchestrator policy."""

    recommendation_id: int = Field(ge=0)
    agent_name: str
    action: str = Field(min_length=1, max_length=300)
    reasoning: str = Field(min_length=5, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    target_metrics: list[str] = Field(default_factory=list)
    expected_impact: dict[str, float] = Field(default_factory=dict)
    bid: float = Field(default=0.0, ge=0.0, le=1.0)
    action_cost: float = Field(default=0.3, ge=0.0, le=2.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    blast_radius: str = Field(default="low")
    rollback_plan: str = Field(default="", max_length=500)


class OrchestratorDecision(BaseModel):
    """Structured action emitted by the trainable orchestrator policy."""

    selected_recommendation_id: int = Field(
        ge=0,
        description="ID of the candidate recommendation to execute.",
    )
    override_adversary: bool = Field(
        default=False,
        description="Whether the policy is intentionally overriding the adversarial agent.",
    )
    reasoning: str = Field(
        min_length=5,
        max_length=1200,
        description="Short causal justification for the selected action.",
    )
    predicted_2step_impact: dict[str, float] = Field(
        default_factory=dict,
        description="Predicted metric changes two steps ahead.",
    )
    schema_drift_detected: bool = Field(default=False)
    schema_drift_field: Optional[str] = Field(default=None)


class ParsedActionResult(BaseModel):
    """Environment-side parse result used for robust reward shaping."""

    decision: Optional[OrchestratorDecision] = None
    format_valid: bool = True
    selection_valid: bool = True
    parse_error: Optional[str] = None
    used_legacy_fallback: bool = False
