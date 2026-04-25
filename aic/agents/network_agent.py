# aic/agents/network_agent.py
"""
Network specialist sub-agent.
Monitors: packet_loss_pct, dns_latency_ms, lb_5xx_count, regional_latency_ms.
Uses claude-haiku-4-5-20251001 with rule-based fallback.
"""
import json

from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_NET, MAX_TOKENS_AGENT

NETWORK_AGENT_SYSTEM_PROMPT = """You are a specialized Network SRE agent. You monitor four metrics:
- packet_loss_pct: Packet loss percentage (target: 0.0%)
- dns_latency_ms: DNS resolution latency in milliseconds (target: 5ms)
- lb_5xx_count: Load balancer 5xx error count per minute (target: 0)
- regional_latency_ms: Cross-region latency in milliseconds (target: 30ms)

You must recommend ONE specific remediation action.
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

NET_CORRECT_ACTIONS = {
    "drain_lb": {
        "action": "Drain failing load balancer and redistribute traffic to healthy instances",
        "reasoning": "High 5xx error count with packet loss indicates a failing LB node. Draining isolates the bad node while maintaining availability.",
        "confidence": 0.85,
        "target_metrics": ["lb_5xx_count", "packet_loss_pct"],
        "expected_impact": {"lb_5xx_count": -50.0, "packet_loss_pct": -2.0},
        "risk_score": 0.4,
        "blast_radius": "medium",
        "rollback_plan": "Re-enable drained LB node and restore original traffic weights",
    },
    "reroute_az": {
        "action": "Reroute traffic to healthy availability zone and deprioritize degraded region",
        "reasoning": "Elevated regional latency indicates AZ degradation. Rerouting preserves SLA by shifting load to healthy regions.",
        "confidence": 0.82,
        "target_metrics": ["regional_latency_ms", "packet_loss_pct"],
        "expected_impact": {"regional_latency_ms": -100.0, "packet_loss_pct": -1.5},
        "risk_score": 0.6,
        "blast_radius": "high",
        "rollback_plan": "Restore original routing weights and re-enable degraded AZ",
    },
    "dns_failover": {
        "action": "Trigger DNS failover to secondary resolver and flush DNS cache across all edge nodes",
        "reasoning": "DNS latency spike suggests primary resolver degradation. Failover to secondary restores name resolution speed.",
        "confidence": 0.88,
        "target_metrics": ["dns_latency_ms", "regional_latency_ms"],
        "expected_impact": {"dns_latency_ms": -80.0, "regional_latency_ms": -30.0},
        "risk_score": 0.3,
        "blast_radius": "medium",
        "rollback_plan": "Revert DNS to primary resolver and invalidate secondary cache",
    },
}


class NetworkAgent(BaseSubAgent):
    """Network specialist agent with LLM + rule-based fallback."""

    @property
    def agent_name(self) -> str:
        return AGENT_NET

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
            f"Current Network metrics at step {step}:\n{obs_str}\n"
            f"{f'Context: {episode_context}' if episode_context else ''}\n"
            "Recommend one specific remediation action. Output only JSON."
        )
        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=NETWORK_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = message.content[0].text.strip()
            parsed = json.loads(raw_text)
            return SubAgentRecommendation(
                agent_name=AGENT_NET,
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed.get("confidence", 0.7)),
                target_metrics=parsed.get("target_metrics", ["dns_latency_ms"]),
                expected_impact=parsed.get("expected_impact", {}),
                risk_score=float(parsed.get("risk_score", 0.3)),
                blast_radius=parsed.get("blast_radius", "medium"),
                rollback_plan=parsed.get("rollback_plan", ""),
            )
        except Exception:
            return self._rule_based_recommend(observation)

    def _rule_based_recommend(self, observation: dict) -> SubAgentRecommendation:
        """Rule-based fallback for testing and when LLM is unavailable."""
        dns_latency = observation.get("dns_latency_ms", 0) or 0
        lb_5xx = observation.get("lb_5xx_count", 0) or 0
        regional_lat = observation.get("regional_latency_ms", 0) or 0
        packet_loss = observation.get("packet_loss_pct", 0) or 0

        # Decision tree
        if dns_latency > 50:
            template = NET_CORRECT_ACTIONS["dns_failover"]
        elif lb_5xx > 20 or packet_loss > 5:
            template = NET_CORRECT_ACTIONS["drain_lb"]
        elif regional_lat > 100:
            template = NET_CORRECT_ACTIONS["reroute_az"]
        else:
            # Default to DNS failover for general network issues
            template = NET_CORRECT_ACTIONS["dns_failover"]

        return SubAgentRecommendation(
            agent_name=AGENT_NET,
            action=template["action"],
            reasoning=template["reasoning"],
            confidence=template["confidence"],
            target_metrics=template["target_metrics"],
            expected_impact=template["expected_impact"],
            bid=min(1.0, 0.2 + float(template["confidence"]) * 0.8),
            action_cost=0.75,
            risk_score=template["risk_score"],
            blast_radius=template["blast_radius"],
            rollback_plan=template["rollback_plan"],
        )
