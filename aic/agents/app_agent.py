# aic/agents/app_agent.py
"""
App specialist sub-agent.
Monitors: error_rate_pct, p95_latency_ms, queue_depth.
Uses claude-haiku-4-5-20251001 with rule-based fallback.
"""
import json

from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_APP, MAX_TOKENS_AGENT

APP_AGENT_SYSTEM_PROMPT = """You are a specialized Application SRE agent. You monitor three metrics:
- error_rate_pct: Application error rate percentage (target: 0.5%)
- p95_latency_ms: P95 response latency in milliseconds (target: 200ms)
- queue_depth: Request queue depth (target: 50)

You will receive current metric values and must recommend ONE specific remediation action.
Your output MUST be valid JSON with exactly these fields:
{
  "action": "specific action description (imperative, concrete, max 200 chars)",
  "reasoning": "causal explanation of why this action addresses the root cause (max 300 chars)",
  "confidence": 0.0 to 1.0,
  "target_metrics": ["list", "of", "metric", "names"]
}

Output ONLY the JSON object."""

APP_CORRECT_ACTIONS = {
    "high_error_rate": {
        "action": "Enable circuit breaker on failing endpoints with 50% threshold and 30s recovery window",
        "reasoning": "Error rate above threshold indicates cascading failures. Circuit breaker isolates failing paths and allows gradual recovery.",
        "confidence": 0.86,
        "target_metrics": ["error_rate_pct", "p95_latency_ms"],
    },
    "high_queue_depth": {
        "action": "Enable adaptive rate limiting at 80% of current throughput and activate request priority queue",
        "reasoning": "Queue depth buildup indicates incoming rate exceeding processing capacity. Rate limiting prevents queue overflow.",
        "confidence": 0.82,
        "target_metrics": ["queue_depth", "error_rate_pct"],
    },
    "high_latency": {
        "action": "Enable response caching for read-heavy endpoints with 60s TTL and activate connection keep-alive",
        "reasoning": "High P95 latency with moderate error rate suggests slow backends. Caching absorbs repeated reads and reduces upstream pressure.",
        "confidence": 0.78,
        "target_metrics": ["p95_latency_ms", "queue_depth"],
    },
}


class AppAgent(BaseSubAgent):
    """App specialist agent with LLM + rule-based fallback."""

    @property
    def agent_name(self) -> str:
        return AGENT_APP

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
            f"Current App metrics at step {step}:\n{obs_str}\n"
            f"{f'Context: {episode_context}' if episode_context else ''}\n"
            "Recommend one specific remediation action. Output only JSON."
        )
        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=APP_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = message.content[0].text.strip()
            parsed = json.loads(raw_text)
            return SubAgentRecommendation(
                agent_name=AGENT_APP,
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed.get("confidence", 0.7)),
                target_metrics=parsed.get("target_metrics", ["error_rate_pct"]),
            )
        except Exception:
            return self._rule_based_recommend(observation)

    def _rule_based_recommend(self, observation: dict) -> SubAgentRecommendation:
        """Rule-based fallback."""
        error_rate = observation.get("error_rate_pct", 0) or 0
        queue = observation.get("queue_depth", 0) or 0
        latency = observation.get("p95_latency_ms", 0) or 0

        if error_rate > 5.0:
            template = APP_CORRECT_ACTIONS["high_error_rate"]
        elif queue > 200:
            template = APP_CORRECT_ACTIONS["high_queue_depth"]
        else:
            template = APP_CORRECT_ACTIONS["high_latency"]

        return SubAgentRecommendation(
            agent_name=AGENT_APP,
            action=template["action"],
            reasoning=template["reasoning"],
            confidence=template["confidence"],
            target_metrics=template["target_metrics"],
            bid=min(1.0, max(0.0, float(template["confidence"]))),
            action_cost=0.4,
        )
