# aic/agents/security_agent.py
"""
Security specialist sub-agent.
Monitors: auth_failure_rate, suspicious_token_count, compromised_ip_count.
Uses claude-haiku-4-5-20251001 with rule-based fallback.

Security actions are inherently high-risk/high-blast-radius — they trade
availability for safety.
"""
import json

from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_SEC, MAX_TOKENS_AGENT

SECURITY_AGENT_SYSTEM_PROMPT = """You are a specialized Security SRE agent. You monitor three metrics:
- auth_failure_rate: Authentication failure rate per minute (target: 0.1)
- suspicious_token_count: Number of suspicious API tokens detected (target: 0)
- compromised_ip_count: Number of known-compromised IPs active (target: 0)

You must recommend ONE specific security remediation action.
Security actions are high-impact — always include a rollback plan.
Your output MUST be valid JSON with exactly these fields:
{
  "action": "specific action description (imperative, concrete, max 200 chars)",
  "reasoning": "causal explanation (max 300 chars)",
  "confidence": 0.0 to 1.0,
  "target_metrics": ["list", "of", "metric", "names"],
  "risk_score": 0.0 to 1.0,
  "blast_radius": "low" or "medium" or "high",
  "rollback_plan": "how to revert this action if it fails (max 200 chars)"
}

Output ONLY the JSON object. No preamble."""

SEC_CORRECT_ACTIONS = {
    "revoke_tokens": {
        "action": "Revoke all suspicious API tokens and force re-authentication for affected sessions",
        "reasoning": "Suspicious token issuance indicates credential compromise. Immediate revocation limits blast radius of potential data exfiltration.",
        "confidence": 0.90,
        "target_metrics": ["suspicious_token_count", "auth_failure_rate"],
        "expected_impact": {"suspicious_token_count": -10.0, "auth_failure_rate": -5.0},
        "risk_score": 0.5,
        "blast_radius": "medium",
        "rollback_plan": "Re-issue tokens for verified legitimate sessions and restore access",
    },
    "isolate_service": {
        "action": "Isolate compromised service from network and redirect traffic to quarantine endpoint",
        "reasoning": "Compromised IPs accessing the service indicate active breach. Isolation prevents lateral movement at the cost of temporary unavailability.",
        "confidence": 0.92,
        "target_metrics": ["compromised_ip_count", "auth_failure_rate"],
        "expected_impact": {"compromised_ip_count": -5.0, "auth_failure_rate": -10.0},
        "risk_score": 0.85,
        "blast_radius": "high",
        "rollback_plan": "Remove network isolation rules and restore service connectivity",
    },
    "degraded_safe_mode": {
        "action": "Switch to degraded-safe mode: disable non-essential API endpoints and enable enhanced logging",
        "reasoning": "Elevated auth failures suggest credential stuffing attack. Degraded mode reduces attack surface while maintaining core functionality.",
        "confidence": 0.78,
        "target_metrics": ["auth_failure_rate", "suspicious_token_count"],
        "expected_impact": {"auth_failure_rate": -3.0},
        "risk_score": 0.4,
        "blast_radius": "medium",
        "rollback_plan": "Re-enable all API endpoints and revert to standard logging level",
    },
}


class SecurityAgent(BaseSubAgent):
    """Security specialist agent with LLM + rule-based fallback."""

    @property
    def agent_name(self) -> str:
        return AGENT_SEC

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._client = None
        if use_llm:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception:
                self.use_llm = False

    def recommend(
        self, observation: dict, step: int, episode_context: str = "",
    ) -> SubAgentRecommendation:
        if not self.use_llm or self._client is None:
            return self._rule_based_recommend(observation)
        return self._llm_recommend(observation, step, episode_context)

    def _llm_recommend(
        self, observation: dict, step: int, episode_context: str,
    ) -> SubAgentRecommendation:
        obs_str = json.dumps(observation, indent=2)
        user_message = (
            f"Current Security metrics at step {step}:\n{obs_str}\n"
            f"{f'Context: {episode_context}' if episode_context else ''}\n"
            "Recommend one specific security remediation action. Output only JSON."
        )
        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=SECURITY_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = message.content[0].text.strip()
            parsed = json.loads(raw_text)
            return SubAgentRecommendation(
                agent_name=AGENT_SEC,
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed.get("confidence", 0.7)),
                target_metrics=parsed.get("target_metrics", ["auth_failure_rate"]),
                expected_impact=parsed.get("expected_impact", {}),
                risk_score=float(parsed.get("risk_score", 0.5)),
                blast_radius=parsed.get("blast_radius", "high"),
                rollback_plan=parsed.get("rollback_plan", ""),
            )
        except Exception:
            return self._rule_based_recommend(observation)

    def _rule_based_recommend(self, observation: dict) -> SubAgentRecommendation:
        """Rule-based fallback for testing and when LLM is unavailable."""
        auth_failures = observation.get("auth_failure_rate", 0) or 0
        suspicious_tokens = observation.get("suspicious_token_count", 0) or 0
        compromised_ips = observation.get("compromised_ip_count", 0) or 0

        # Decision tree — security actions escalate with severity
        if compromised_ips > 3:
            template = SEC_CORRECT_ACTIONS["isolate_service"]
        elif suspicious_tokens > 5 or (auth_failures > 10 and suspicious_tokens > 0):
            template = SEC_CORRECT_ACTIONS["revoke_tokens"]
        else:
            template = SEC_CORRECT_ACTIONS["degraded_safe_mode"]

        return SubAgentRecommendation(
            agent_name=AGENT_SEC,
            action=template["action"],
            reasoning=template["reasoning"],
            confidence=template["confidence"],
            target_metrics=template["target_metrics"],
            expected_impact=template["expected_impact"],
            risk_score=template["risk_score"],
            blast_radius=template["blast_radius"],
            rollback_plan=template["rollback_plan"],
        )
