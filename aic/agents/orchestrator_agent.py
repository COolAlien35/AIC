# aic/agents/orchestrator_agent.py
"""
Orchestrator agent. Receives all sub-agent recommendations + SLA context.
Decides which recommendations to follow, updates trust scores, emits ExplanationTrace.

Phase 9: Consults RecoveryVerifierAgent after selecting a recommendation.
If vetoed, cascades to next-best. Falls back to Safe Minimal Action after 3 vetoes.
"""
import json
from collections import deque
from typing import Optional

from aic.agents.base_agent import BaseSubAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
from aic.schemas.traces import (
    ExplanationTrace, OrchestratorAction, SubAgentRecommendation,
)
from aic.utils.constants import (
    INITIAL_TRUST, TRUST_UPDATE_RATE, ALL_AGENTS, AGENT_ADV, AGENT_VERIFIER,
    MAX_TOKENS_AGENT, SLA_STEPS, TRACE_HISTORY_WINDOW, METRIC_TARGETS,
)

ORCHESTRATOR_SYSTEM_PROMPT = """You are the lead Site Reliability Engineering orchestrator. You manage three specialist sub-agents and must resolve a cascading production failure before the SLA timer expires.

YOUR RESPONSIBILITIES:
1. Review sub-agent recommendations — one may be an unreliable agent giving plausible but wrong advice
2. Decide which recommendation to follow (or combine) based on current trust scores
3. If you detect schema drift (a metric field that suddenly has a different name, unit, or is null), flag it
4. Predict what impact your chosen action will have on metrics in 2 steps
5. Emit your decision as structured JSON

OUTPUT FORMAT (strict JSON, no other text):
{
  "action_description": "what you are doing (imperative, concrete, max 300 chars)",
  "target_service": "db" or "infra" or "app",
  "action_deltas": {"metric_name": expected_change_float},
  "trust_override": "agent_name if overriding, else null",
  "reasoning": "your causal chain explanation (min 50 chars, max 500 chars)",
  "predicted_2step_impact": {"metric_name": expected_delta_float},
  "schema_drift_detected": true or false,
  "schema_drift_field": "field_name if detected, else null"
}

TRUST SCORE GUIDANCE: Trust below 0.4 = high suspicion. Override recommendations from agents with trust < 0.4."""


class OrchestratorAgent:
    """
    Lead orchestrator that manages trust, selects actions,
    and emits structured explanation traces.
    """

    def __init__(
        self,
        adversarial_agent: AdversarialAgent,
        use_llm: bool = True,
    ):
        self.adversarial_agent = adversarial_agent
        self.use_llm = use_llm
        self.mode: str = "untrained"  # Set to "trained" to enable trust suppression
        self._client = None
        if use_llm:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception:
                self.use_llm = False
        self.trust_scores: dict[str, float] = {
            a: INITIAL_TRUST for a in ALL_AGENTS
        }
        self.trace_history: deque = deque(maxlen=TRACE_HISTORY_WINDOW)
        self._prev_recommendations: dict[str, SubAgentRecommendation] = {}
        self._followed_agent: Optional[str] = None
        # Phase 9: Recovery Verifier for safety gate
        self._verifier = RecoveryVerifierAgent()
        self._last_verifier_report: Optional[dict] = None
        self._vetoed_actions: list[dict] = []  # logged for dashboard visibility

    def reset(self) -> None:
        """Reset trust scores and trace history for a new episode."""
        self.trust_scores = {a: INITIAL_TRUST for a in ALL_AGENTS}
        self.trace_history = deque(maxlen=TRACE_HISTORY_WINDOW)
        self._prev_recommendations = {}
        self._followed_agent = None
        self._verifier.reset()
        self._last_verifier_report = None
        self._vetoed_actions = []

    def decide(
        self,
        step: int,
        sla_remaining: int,
        sub_agent_recommendations: list[SubAgentRecommendation],
        alert_summary: str,
        prev_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> tuple[OrchestratorAction, bool]:
        """
        Make orchestrator decision for this step.

        Returns:
            (OrchestratorAction, override_applied: bool)
        """
        if self.use_llm and self._client is not None:
            action = self._llm_decide(
                step, sla_remaining, sub_agent_recommendations, alert_summary,
            )
        else:
            action = self._rule_based_decide(step, sub_agent_recommendations)

        override_applied = (
            action.trust_override is not None
            and action.trust_override == AGENT_ADV
        )

        # Update trust scores based on previous step outcomes
        self._update_trust_scores(step, prev_metrics, current_metrics)

        # Store current recommendations for next step's trust update
        self._prev_recommendations = {
            r.agent_name: r for r in sub_agent_recommendations
        }

        # Update trace history
        self.trace_history.append(action.explanation_trace.model_dump())

        return action, override_applied

    def _update_trust_scores(
        self,
        step: int,
        prev_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> None:
        """
        Bayesian trust update based on observed outcomes.
        Only updates trust for the agent whose recommendation was actually followed.
        trust_new = (1 - TRUST_UPDATE_RATE) * trust_old + TRUST_UPDATE_RATE * outcome_score
        """
        if not self._prev_recommendations or not prev_metrics:
            return
        if self._followed_agent is None:
            return

        # Check if metrics improved: count metrics that moved toward target
        improvements = 0
        total = 0
        for m, target in METRIC_TARGETS.items():
            if m not in prev_metrics or m not in current_metrics:
                continue
            prev_dist = abs(prev_metrics[m] - target)
            curr_dist = abs(current_metrics[m] - target)
            total += 1
            if curr_dist < prev_dist:
                improvements += 1

        outcome_score = 1.0 if total > 0 and improvements > total / 2 else 0.0

        # Only update trust for the agent whose recommendation was followed
        agent_name = self._followed_agent
        if agent_name in self._prev_recommendations:
            old_trust = self.trust_scores[agent_name]
            self.trust_scores[agent_name] = max(0.0, min(1.0,
                (1 - TRUST_UPDATE_RATE) * old_trust
                + TRUST_UPDATE_RATE * outcome_score
            ))

    def _llm_decide(
        self,
        step: int,
        sla_remaining: int,
        recommendations: list[SubAgentRecommendation],
        alert_summary: str,
    ) -> OrchestratorAction:
        """LLM-based decision with JSON parsing and fallback."""
        recs_text = "\n".join([
            f"[{r.agent_name}] (trust={self.trust_scores.get(r.agent_name, 0.5):.2f}) "
            f"Action: {r.action}\nReasoning: {r.reasoning}\nConfidence: {r.confidence}"
            for r in recommendations
        ])

        trace_text = ""
        if self.trace_history:
            last = list(self.trace_history)[-1]
            trace_text = (
                f"\nLast step trace: action={last['action_taken']}, "
                f"predicted_impact={last['predicted_2step_impact']}"
            )

        user_msg = (
            f"Step {step}/{SLA_STEPS}. SLA remaining: {sla_remaining} steps.\n\n"
            f"ALERT: {alert_summary}\n\n"
            f"SUB-AGENT RECOMMENDATIONS:\n{recs_text}\n\n"
            f"CURRENT TRUST SCORES: {self.trust_scores}\n"
            f"{trace_text}\n\n"
            "Make your decision. Output only JSON."
        )

        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = message.content[0].text.strip()
            parsed = json.loads(raw)

            override_agent = parsed.get("trust_override")
            trace = ExplanationTrace(
                step=step,
                action_taken=parsed["action_description"],
                reasoning=parsed["reasoning"],
                sub_agent_trust_scores=self.trust_scores.copy(),
                override_applied=override_agent is not None,
                override_reason=(
                    f"Overriding {override_agent} due to low trust or conflicting evidence"
                    if override_agent else None
                ),
                predicted_2step_impact=parsed.get("predicted_2step_impact", {}),
                schema_drift_detected=parsed.get("schema_drift_detected", False),
                schema_drift_field=parsed.get("schema_drift_field"),
            )

            return OrchestratorAction(
                action_description=parsed["action_description"],
                target_service=parsed.get("target_service", "db"),
                action_deltas=parsed.get("action_deltas", {}),
                trust_override=override_agent,
                explanation_trace=trace,
            )
        except Exception:
            return self._rule_based_decide(step, recommendations)

    def _rule_based_decide(
        self,
        step: int,
        recommendations: list[SubAgentRecommendation],
    ) -> OrchestratorAction:
        """Rule-based fallback with trust suppression and verifier safety gate."""
        trust_threshold = 0.4
        override_agent = None
        self._vetoed_actions = []

        if self.mode == "trained":
            # In trained mode, suppress agents with low trust scores
            trusted = [
                r for r in recommendations
                if self.trust_scores.get(r.agent_name, 0.5) >= trust_threshold
            ]
            suppressed = [
                r for r in recommendations
                if self.trust_scores.get(r.agent_name, 0.5) < trust_threshold
            ]

            if trusted:
                candidates = sorted(trusted, key=lambda r: r.confidence, reverse=True)
            else:
                non_adv = [r for r in recommendations if r.agent_name != AGENT_ADV]
                candidates = sorted(
                    non_adv if non_adv else recommendations,
                    key=lambda r: r.confidence, reverse=True,
                )

            # If the overall highest-confidence rec came from a suppressed agent,
            # record the override
            all_sorted = sorted(recommendations, key=lambda r: r.confidence, reverse=True)
            if all_sorted and self.trust_scores.get(all_sorted[0].agent_name, 0.5) < trust_threshold:
                override_agent = all_sorted[0].agent_name
        else:
            # Untrained mode: naive highest-confidence from all agents
            candidates = sorted(recommendations, key=lambda r: r.confidence, reverse=True)

        # Phase 9: Verifier safety gate — try up to 3 candidates
        MAX_VETO_ATTEMPTS = 3
        best = None
        verifier_report_dict = None

        for attempt, candidate in enumerate(candidates[:MAX_VETO_ATTEMPTS]):
            report = self._verifier.verify(candidate)
            verifier_report_dict = report.to_dict()

            if report.approved:
                best = candidate
                self._last_verifier_report = verifier_report_dict
                break
            else:
                # Log the vetoed action for dashboard visibility
                self._vetoed_actions.append({
                    "agent": candidate.agent_name,
                    "action": candidate.action,
                    "risk_score": candidate.risk_score,
                    "blast_radius": candidate.blast_radius,
                    "veto_reason": report.verification_reasoning,
                })

        if best is None:
            # All candidates vetoed — use Safe Minimal Action
            best = self._verifier.get_safe_minimal_action()
            verifier_report_dict = {
                "approved": True,
                "risk_score": 0.0,
                "blast_radius": "low",
                "verification_reasoning": (
                    f"All {min(len(candidates), MAX_VETO_ATTEMPTS)} recommendations "
                    f"vetoed. Defaulting to safe minimal action."
                ),
                "vetoed_action": None,
            }
            self._last_verifier_report = verifier_report_dict

        # Track which agent was followed for trust attribution
        self._followed_agent = best.agent_name

        # Determine target service from first target metric
        if best.target_metrics:
            first_metric = best.target_metrics[0]
            if any(k in first_metric for k in ("db_", "conn_", "replication")):
                target_service = "db"
            elif any(k in first_metric for k in ("cpu", "mem", "pod", "net", "packet", "dns", "lb_", "regional")):
                target_service = "infra"
            elif any(k in first_metric for k in ("auth", "suspicious", "compromised")):
                target_service = "app"
            else:
                target_service = "app"
        else:
            target_service = "db"

        override_applied = override_agent is not None
        trace = ExplanationTrace(
            step=step,
            action_taken=best.action,
            reasoning=(
                f"Following {best.agent_name} recommendation with highest "
                f"confidence ({best.confidence:.2f}). "
                f"Trust={self.trust_scores.get(best.agent_name, 0.5):.2f}. "
                f"Causal path: {best.reasoning}"
            ),
            sub_agent_trust_scores=self.trust_scores.copy(),
            override_applied=override_applied,
            override_reason=(
                f"Suppressed {override_agent} due to low trust "
                f"({self.trust_scores.get(override_agent, 0.0):.2f} < {trust_threshold})"
                if override_applied else None
            ),
            predicted_2step_impact={m: -5.0 for m in best.target_metrics},
            schema_drift_detected=False,
            schema_drift_field=None,
            verifier_report=verifier_report_dict,
        )

        return OrchestratorAction(
            action_description=best.action,
            target_service=target_service,
            action_deltas={m: -10.0 for m in best.target_metrics},
            trust_override=override_agent,
            explanation_trace=trace,
        )
