"""Prompt formatting utilities for orchestrator SFT and RL training.

Two layers:
  1. Legacy ``build_orchestrator_prompt`` (kept for backward compatibility with
     the heuristic SFT data generator).
  2. Compact, chat-template based path (``build_chat_messages_compact``,
     ``render_chat_prompt``) used by SFT, GRPO, and benchmark inference so the
     prompt distribution is identical at all three stages.

Why the compact path exists
---------------------------
The legacy prompt routinely emits 2.5k-3.5k tokens (system + alert + sorted
metric/trust/mask dicts + 7 candidates with full reasoning + JSON schema). The
old training command capped ``max_prompt_length`` at 256 which decapitated the
prompt before any candidate ID was visible — the model literally could not see
what it was supposed to choose. The compact path targets <=700 tokens after
chat-template wrapping and fits comfortably within ``max_prompt_length=1024``.
"""
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


# ---------------------------------------------------------------------------
# Legacy verbose prompt — retained for SFT data generator parity.
# ---------------------------------------------------------------------------

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
    """Create chat messages for instruct-style models (legacy verbose path)."""
    return [
        {"role": "system", "content": ORCHESTRATOR_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": build_orchestrator_prompt(observation)},
    ]


# ---------------------------------------------------------------------------
# Compact path (used for training + GRPO + inference)
# ---------------------------------------------------------------------------

COMPACT_SYSTEM_PROMPT = (
    "You are the Adaptive Incident Choreographer. "
    "Choose ONE candidate id, give a short reason, predict the 2-step metric "
    "impact. Reply with strict JSON: "
    '{"selected_recommendation_id":int,"override_adversary":bool,'
    '"reasoning":str,"predicted_2step_impact":{metric:float},'
    '"schema_drift_detected":bool,"schema_drift_field":str|null}'
)


def _fmt_float(value: float) -> str:
    """Compact float repr (max 1 decimal) to keep token count low."""
    if value is None:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _compact_metrics(metrics: dict[str, float]) -> str:
    """Render the metrics dict as a single line `k=v` series."""
    if not metrics:
        return "(none)"
    parts = [f"{k}={_fmt_float(float(v))}" for k, v in metrics.items()]
    return " ".join(parts)


def _compact_trust(trust: dict[str, float]) -> str:
    if not trust:
        return "(none)"
    return " ".join(f"{k}={float(v):.2f}" for k, v in trust.items())


def build_compact_user_text(observation: dict[str, Any] | OrchestratorObservation) -> str:
    """Build a token-frugal user message describing the incident state.

    Removed vs the legacy prompt:
      - observation_masks dict (~120 tokens)
      - shared_noisy_signal dict (~25 tokens)
      - trace_history list
      - sub_agent_recommendations duplicate
      - per-candidate target_metrics / expected_impact / rollback_plan
      - JSON schema example block
    """
    obs = normalize_observation(observation)

    drift = "yes" if obs.schema_drift_active else "no"
    lines: list[str] = [
        f"step={obs.step} sla_left={obs.sla_remaining_steps} "
        f"budget={obs.episode_budget_remaining:.1f} drift={drift}",
        f"alerts: {obs.alert_summary_text.strip() or 'none'}",
        f"metrics: {_compact_metrics(obs.current_metrics)}",
        f"trust: {_compact_trust(obs.current_trust_scores)}",
        "candidates:",
    ]
    for c in obs.candidate_recommendations:
        action = (c.action or "").strip().replace("\n", " ")
        if len(action) > 90:
            action = action[:87] + "..."
        lines.append(
            f"  id={c.recommendation_id} agent={c.agent_name} "
            f"conf={c.confidence:.2f} risk={c.risk_score:.2f} | {action}"
        )
    return "\n".join(lines)


def build_chat_messages_compact(
    observation: dict[str, Any] | OrchestratorObservation,
) -> list[dict[str, str]]:
    """Compact chat messages used at SFT, GRPO, and inference time."""
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": build_compact_user_text(observation)},
    ]


def render_chat_prompt(
    tokenizer,
    observation: dict[str, Any] | OrchestratorObservation,
    add_generation_prompt: bool = True,
) -> str:
    """Apply the model's chat template to the compact observation messages.

    This is the single source of truth for prompt formatting at inference.
    SFT also uses these same messages to build training examples so the
    distribution is identical.
    """
    messages = build_chat_messages_compact(observation)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


# ---------------------------------------------------------------------------
# Decision (de)serialization helpers
# ---------------------------------------------------------------------------

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
