"""Helpers for driving the AIC environment from heuristic and learned policies."""
from __future__ import annotations

from aic.schemas.actions import OrchestratorDecision
from aic.schemas.traces import OrchestratorAction, SubAgentRecommendation


def materialize_recommendations(obs: dict) -> list[SubAgentRecommendation]:
    """Convert serialized recommendation dicts back into schema instances."""
    return [
        SubAgentRecommendation.model_validate(rec)
        for rec in obs.get("sub_agent_recommendations", [])
    ]


def select_candidate_id(
    obs: dict,
    action_description: str,
    followed_agent: str | None,
) -> int:
    """Resolve a heuristic action back to the environment candidate slate."""
    candidates = obs.get("candidate_recommendations", [])
    for candidate in candidates:
        if (
            candidate.get("agent_name") == followed_agent
            and candidate.get("action") == action_description
        ):
            return int(candidate["recommendation_id"])
    for candidate in candidates:
        if candidate.get("action") == action_description:
            return int(candidate["recommendation_id"])
    for candidate in candidates:
        if candidate.get("agent_name") == "recovery_verifier":
            return int(candidate["recommendation_id"])
    return 0


def make_structured_action(
    obs: dict,
    action: OrchestratorAction,
    followed_agent: str | None,
    override_applied: bool,
) -> dict:
    """Convert heuristic orchestrator output into environment-native structured JSON."""
    candidate_id = select_candidate_id(obs, action.action_description, followed_agent)
    decision = OrchestratorDecision(
        selected_recommendation_id=candidate_id,
        override_adversary=override_applied,
        reasoning=action.explanation_trace.reasoning,
        predicted_2step_impact=action.explanation_trace.predicted_2step_impact,
        schema_drift_detected=action.explanation_trace.schema_drift_detected,
        schema_drift_field=action.explanation_trace.schema_drift_field,
    )
    return decision.model_dump()
