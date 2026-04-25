# aic/agents/infra_agent.py
"""
Infra specialist sub-agent.
Monitors: cpu_pct, mem_pct, pod_restarts, net_io_mbps.
Uses claude-haiku-4-5-20251001 with rule-based fallback.
"""
import json

from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_INFRA, MAX_TOKENS_AGENT

INFRA_AGENT_SYSTEM_PROMPT = """You are a specialized Infrastructure SRE agent. You monitor four metrics:
- cpu_pct: CPU utilization percentage (target: 45%)
- mem_pct: Memory utilization percentage (target: 60%)
- pod_restarts: Number of pod restarts (target: 0)
- net_io_mbps: Network I/O in MB/s (target: 100 MB/s)

You will receive current metric values and must recommend ONE specific remediation action.
Your output MUST be valid JSON with exactly these fields:
{
  "action": "specific action description (imperative, concrete, max 200 chars)",
  "reasoning": "causal explanation of why this action addresses the root cause (max 300 chars)",
  "confidence": 0.0 to 1.0,
  "target_metrics": ["list", "of", "metric", "names"]
}

Output ONLY the JSON object."""

INFRA_CORRECT_ACTIONS = {
    "high_memory": {
        "action": "Trigger garbage collection and set memory limit to 75% with OOM-kill threshold at 90%",
        "reasoning": "Memory at critical levels indicates leak or cache bloat. GC plus hard limits prevent OOM cascades.",
        "confidence": 0.84,
        "target_metrics": ["mem_pct", "pod_restarts"],
    },
    "high_pod_restarts": {
        "action": "Set pod restart backoff to 60s, enable liveness probe grace period of 120s, and capture core dumps",
        "reasoning": "Rapid restarts indicate crash-loop. Backoff prevents resource churn while dumps provide root cause data.",
        "confidence": 0.80,
        "target_metrics": ["pod_restarts", "cpu_pct"],
    },
    "high_cpu": {
        "action": "Enable CPU throttling at 80% and redistribute workload across available nodes with affinity rules",
        "reasoning": "CPU near saturation causes context-switch overhead. Throttling plus redistribution balances the cluster.",
        "confidence": 0.77,
        "target_metrics": ["cpu_pct", "mem_pct"],
    },
    "high_network": {
        "action": "Enable network QoS policies and rate-limit non-critical traffic to 50% of current bandwidth",
        "reasoning": "Network I/O spike suggests noisy-neighbor or DDoS. QoS policies protect critical service traffic.",
        "confidence": 0.75,
        "target_metrics": ["net_io_mbps", "error_rate_pct"],
    },
}


class InfraAgent(BaseSubAgent):
    """Infra specialist agent with LLM + rule-based fallback."""

    @property
    def agent_name(self) -> str:
        return AGENT_INFRA

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
            f"Current Infra metrics at step {step}:\n{obs_str}\n"
            f"{f'Context: {episode_context}' if episode_context else ''}\n"
            "Recommend one specific remediation action. Output only JSON."
        )
        try:
            message = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS_AGENT,
                system=INFRA_AGENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = message.content[0].text.strip()
            parsed = json.loads(raw_text)
            return SubAgentRecommendation(
                agent_name=AGENT_INFRA,
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                confidence=float(parsed.get("confidence", 0.7)),
                target_metrics=parsed.get("target_metrics", ["cpu_pct"]),
            )
        except Exception:
            return self._rule_based_recommend(observation)

    def _rule_based_recommend(self, observation: dict) -> SubAgentRecommendation:
        """Rule-based fallback."""
        mem = observation.get("mem_pct", 0) or 0
        pod = observation.get("pod_restarts", 0) or 0
        cpu = observation.get("cpu_pct", 0) or 0
        net = observation.get("net_io_mbps", 0) or 0

        if mem > 85:
            template = INFRA_CORRECT_ACTIONS["high_memory"]
        elif pod > 3:
            template = INFRA_CORRECT_ACTIONS["high_pod_restarts"]
        elif net > 200:
            template = INFRA_CORRECT_ACTIONS["high_network"]
        else:
            template = INFRA_CORRECT_ACTIONS["high_cpu"]

        return SubAgentRecommendation(
            agent_name=AGENT_INFRA,
            action=template["action"],
            reasoning=template["reasoning"],
            confidence=template["confidence"],
            target_metrics=template["target_metrics"],
            bid=min(1.0, max(0.0, float(template["confidence"]))),
            action_cost=0.45,
        )
