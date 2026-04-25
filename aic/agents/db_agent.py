# aic/agents/db_agent.py
"""
DB specialist sub-agent.
Monitors: db_latency_ms, conn_pool_pct, replication_lag_ms.
Uses claude-haiku-4-5-20251001 with rule-based fallback.
"""
import json

from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_DB, MAX_TOKENS_AGENT

DB_AGENT_SYSTEM_PROMPT = """You are a specialized Database SRE agent. You monitor three metrics:
- db_latency_ms: Database query latency in milliseconds (target: 50ms)
- conn_pool_pct: Connection pool utilization percentage (target: 60%)
- replication_lag_ms: Replication lag in milliseconds (target: 10ms)

You will receive current metric values and must recommend ONE specific remediation action.
Your output MUST be valid JSON with exactly these fields:
{
  "action": "specific action description (imperative, concrete, max 200 chars)",
  "reasoning": "causal explanation of why this action addresses the root cause (max 300 chars)",
  "confidence": 0.0 to 1.0,
  "target_metrics": ["list", "of", "metric", "names"]
}

Output ONLY the JSON object. No preamble, no explanation outside the JSON."""

DB_CORRECT_ACTIONS = {
    "high_latency_high_pool": {
        "action": "Drain connection pool to 40% capacity and enable connection queuing with 30s timeout",
        "reasoning": "Pool saturation at high utilization is causing query queuing. Controlled drain with queuing prevents connection storms.",
        "confidence": 0.88,
        "target_metrics": ["conn_pool_pct", "db_latency_ms"],
    },
    "high_latency_low_pool": {
        "action": "Enable query result caching and set slow query log threshold to 100ms for analysis",
        "reasoning": "High latency without pool pressure indicates inefficient queries. Cache hot paths while identifying slow queries.",
        "confidence": 0.72,
        "target_metrics": ["db_latency_ms"],
    },
    "replication_lag": {
        "action": "Pause non-critical batch jobs and throttle write throughput to 60% to allow replica catchup",
        "reasoning": "Replication lag indicates write volume exceeding replica apply speed. Throttling writes allows replica to catch up.",
        "confidence": 0.85,
        "target_metrics": ["replication_lag_ms", "db_latency_ms"],
    },
}


class DBAgent(BaseSubAgent):
    """DB specialist agent with LLM + rule-based fallback."""

    @property
    def agent_name(self) -> str:
        return AGENT_DB

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
            f"Current DB metrics at step {step}:\n{obs_str}\n"
            f"{f'Context: {episode_context}' if episode_context else ''}\n"
            "Recommend one specific remediation action. Output only JSON."
        )
        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=DB_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = message.content[0].text.strip()
            parsed = json.loads(raw_text)
            return SubAgentRecommendation(
                agent_name=AGENT_DB,
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed.get("confidence", 0.7)),
                target_metrics=parsed.get("target_metrics", ["db_latency_ms"]),
            )
        except Exception:
            return self._rule_based_recommend(observation)

    def _rule_based_recommend(self, observation: dict) -> SubAgentRecommendation:
        """Rule-based fallback — used in testing and when LLM unavailable."""
        conn_pool = observation.get("conn_pool_pct", 0)
        repl_lag = observation.get("replication_lag_ms", 0)

        if repl_lag is not None and repl_lag > 100:
            template = DB_CORRECT_ACTIONS["replication_lag"]
        elif conn_pool is not None and conn_pool > 80:
            template = DB_CORRECT_ACTIONS["high_latency_high_pool"]
        else:
            template = DB_CORRECT_ACTIONS["high_latency_low_pool"]

        return SubAgentRecommendation(
            agent_name=AGENT_DB,
            action=template["action"],
            reasoning=template["reasoning"],
            confidence=template["confidence"],
            target_metrics=template["target_metrics"],
            bid=min(1.0, max(0.0, float(template["confidence"]))),
            action_cost=0.55,
        )
