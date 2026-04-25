"""Prompt formatting utilities for orchestrator SFT and RL training."""
from __future__ import annotations

import json
from typing import Any

from aic.schemas.actions import OrchestratorDecision
from aic.schemas.observations import OrchestratorObservation


ORCHESTRATOR_POLICY_SYSTEM_PROMPT = (
    "You are the Adaptive Incident Choreographer orchestrator. "
    "Select exactly one candidate recommendation ID, explain why, and predict "
    "the 2-step metric impact. Output strict JSON only."
)


def normalize_observation(observation: dict[str, Any] | OrchestratorObservation) -> OrchestratorObservation:
    """Normalize dict observations into the Pydantic orchestrator schema."""
    if isinstance(observation, OrchestratorObservation):
        return observation
    return OrchestratorObservation.model_validate(observation)


def build_orchestrator_prompt(observation: dict[str, Any] | OrchestratorObservation) -> str:
    """Render an orchestrator observation into a compact prompt string."""
    obs = normalize_observation(observation)

    lines = [
        "# Incident Context",
        f"Step: {obs.step}",
        f"SLA Remaining Steps: {obs.sla_remaining_steps}",
        f"Episode Budget Remaining: {obs.episode_budget_remaining:.2f}",
        f"Schema Drift Active: {obs.schema_drift_active}",
        f"Schema Drift Type: {obs.schema_drift_type}",
        "",
        "# Shared Noisy Signal",
        json.dumps(getattr(obs, 'shared_noisy_signal', {}), sort_keys=True),
        "",
        "# Alert Summary",
        obs.alert_summary_text,
        "",
        "# Current Metrics",
        json.dumps(obs.current_metrics, sort_keys=True),
        "",
        "# Trust Scores",
        json.dumps(obs.current_trust_scores, sort_keys=True),
        "",
        "# Observation Masks (private info asymmetry)",
        json.dumps(getattr(obs, 'observation_masks', {}), sort_keys=True),
        "",
        "# Candidate Recommendations",
    ]

    for candidate in obs.candidate_recommendations:
        lines.extend(
            [
                f"- id={candidate.recommendation_id} agent={candidate.agent_name} confidence={candidate.confidence:.2f} bid={getattr(candidate,'bid',0.0):.2f} cost={getattr(candidate,'action_cost',0.0):.2f}",
                f"  action={candidate.action}",
                f"  reasoning={candidate.reasoning}",
                f"  risk={candidate.risk_score:.2f} blast_radius={candidate.blast_radius}",
                f"  target_metrics={candidate.target_metrics}",
                f"  expected_impact={candidate.expected_impact}",
            ]
        )

    lines.extend(
        [
            "",
            "# Output JSON Schema",
            json.dumps(
                {
                    "selected_recommendation_id": 0,
                    "override_adversary": False,
                    "reasoning": "brief causal explanation",
                    "predicted_2step_impact": {"db_latency_ms": -100.0},
                    "schema_drift_detected": False,
                    "schema_drift_field": None,
                },
                indent=2,
            ),
        ]
    )
    return "\n".join(lines)


def build_chat_messages(observation: dict[str, Any] | OrchestratorObservation) -> list[dict[str, str]]:
    """Create chat messages for instruct-style models."""
    return [
        {"role": "system", "content": ORCHESTRATOR_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": build_orchestrator_prompt(observation)},
    ]


def serialize_decision(decision: OrchestratorDecision | dict[str, Any]) -> str:
    """Serialize an orchestrator decision to canonical JSON."""
    if isinstance(decision, OrchestratorDecision):
        payload = decision.model_dump()
    else:
        payload = OrchestratorDecision.model_validate(decision).model_dump()
    return json.dumps(payload, sort_keys=True)


def parse_decision(text: str) -> OrchestratorDecision:
    """Parse a model completion into the structured orchestrator decision schema."""
    return OrchestratorDecision.model_validate_json(text)
