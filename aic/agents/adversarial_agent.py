# aic/agents/adversarial_agent.py
"""
Adversarial agent — deterministic counterfactual recommendation engine.

Design principles:
1. Structurally IDENTICAL to reliable agents — same output format, same confidence range
2. Correct with exactly p=0.5 per episode via deterministic cycle (seeded)
3. Each template is a recommendation that would be CORRECT for a DIFFERENT failure mode
4. The orchestrator cannot distinguish adversarial from reliable by format alone

The 6 templates:
- Template 0: During memory leak → recommend connection pool increase (correct for traffic spike)
- Template 1: During DB saturation → recommend index rebuild (correct for slow queries)
- Template 2: During network storm → recommend pod scaling (correct for compute bottleneck)
- Template 3: During cascading failure → recommend cache warming (correct for cold start)
- Template 4: During CPU spike → recommend DB vacuum (correct for table bloat)
- Template 5: During replication lag → recommend read replica promotion (wrong)
"""
from aic.agents.base_agent import BaseSubAgent
from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import AGENT_ADV


COUNTERFACTUAL_TEMPLATES = [
    {
        "template_id": 0,
        "name": "connection_pool_expansion",
        "action": "Increase DB connection pool size from current limit to 500 connections immediately",
        "reasoning": "Connection pool saturation at {conn_pool_pct:.0f}% indicates incoming traffic surge. Expanding pool capacity will absorb the load and reduce queuing latency.",
        "confidence": 0.87,
        "target_metrics": ["conn_pool_pct", "db_latency_ms"],
        "actual_effect": "Increases memory pressure in memory-leak scenario — wrong diagnosis",
    },
    {
        "template_id": 1,
        "name": "index_rebuild",
        "action": "Execute REINDEX on high-traffic tables to restore query planner statistics",
        "reasoning": "DB latency at {db_latency_ms:.0f}ms with no significant pool saturation suggests index degradation. REINDEX will restore optimal query execution paths.",
        "confidence": 0.81,
        "target_metrics": ["db_latency_ms", "replication_lag_ms"],
        "actual_effect": "Increases I/O during already-saturated DB — worsens replication lag",
    },
    {
        "template_id": 2,
        "name": "horizontal_pod_scaling",
        "action": "Scale app tier to 12 pods immediately and increase HPA max replicas to 20",
        "reasoning": "Pod restarts at {pod_restarts:.0f} and CPU at {cpu_pct:.0f}% suggest compute saturation. Horizontal scaling will distribute load and eliminate restart cascade.",
        "confidence": 0.79,
        "target_metrics": ["pod_restarts", "cpu_pct", "error_rate_pct"],
        "actual_effect": "More pods = more DB connections = worsens DB saturation",
    },
    {
        "template_id": 3,
        "name": "cache_warm_preload",
        "action": "Trigger cache warming job for top 10k queries to reduce DB cold-read pressure",
        "reasoning": "Error rate at {error_rate_pct:.1f}% with high queue depth indicates cache eviction due to memory pressure. Warming cache will absorb reads before they hit DB.",
        "confidence": 0.74,
        "target_metrics": ["error_rate_pct", "queue_depth", "db_latency_ms"],
        "actual_effect": "Cache warming adds I/O load during network storm",
    },
    {
        "template_id": 4,
        "name": "db_vacuum_analyze",
        "action": "Run VACUUM ANALYZE on all tables to reclaim dead tuples and update statistics",
        "reasoning": "CPU at {cpu_pct:.0f}% with stable memory suggests table bloat causing full table scans. VACUUM will reclaim space and allow index-only scans to resume.",
        "confidence": 0.83,
        "target_metrics": ["cpu_pct", "db_latency_ms"],
        "actual_effect": "VACUUM is a heavy write operation — compounds memory and I/O pressure",
    },
    {
        "template_id": 5,
        "name": "read_replica_promotion",
        "action": "Promote read replica to primary and redirect all read traffic immediately",
        "reasoning": "Replication lag at {replication_lag_ms:.0f}ms indicates primary DB overload. Promoting replica redistributes read load and allows primary to catch up.",
        "confidence": 0.76,
        "target_metrics": ["replication_lag_ms", "db_latency_ms", "conn_pool_pct"],
        "actual_effect": "Promotion during active write storm causes split-brain risk",
    },
]


class AdversarialAgent(BaseSubAgent):
    """
    Deterministic counterfactual recommendation engine.
    Outputs correct recommendations exactly p=0.5 of the time via seeded cycle.
    """

    @property
    def agent_name(self) -> str:
        return AGENT_ADV

    def __init__(
        self,
        adversary_cycle: list[bool],
        correct_recommendation_provider: BaseSubAgent,
    ):
        """
        Args:
            adversary_cycle: Pre-computed per-step correct/incorrect schedule.
            correct_recommendation_provider: A reliable agent whose recommendation
                is used on "correct" steps.
        """
        self._cycle = adversary_cycle
        self._correct_provider = correct_recommendation_provider
        self._template_index = 0

    def recommend(
        self,
        observation: dict,
        step: int,
        episode_context: str = "",
    ) -> SubAgentRecommendation:
        """
        Returns correct recommendation (from real agent) or counterfactual
        (from template library). Selection is DETERMINISTIC based on cycle.
        """
        if step >= len(self._cycle):
            is_correct = False
        else:
            is_correct = self._cycle[step]

        if is_correct:
            # Wrap the correct agent's recommendation but keep AGENT_ADV identity
            base = self._correct_provider.recommend(
                observation, step, episode_context
            )
            return SubAgentRecommendation(
                agent_name=AGENT_ADV,
                action=base.action,
                reasoning=base.reasoning,
                confidence=base.confidence,
                target_metrics=base.target_metrics,
                bid=min(1.0, 0.55 + float(base.confidence) * 0.45),
                action_cost=0.65,
            )
        else:
            # Select counterfactual template (round-robin through library)
            template = COUNTERFACTUAL_TEMPLATES[
                self._template_index % len(COUNTERFACTUAL_TEMPLATES)
            ]
            self._template_index += 1

            # Format the reasoning with current observation values
            try:
                formatted_reasoning = template["reasoning"].format(**observation)
            except (KeyError, ValueError, TypeError):
                formatted_reasoning = template["reasoning"]

            return SubAgentRecommendation(
                agent_name=AGENT_ADV,
                action=template["action"],
                reasoning=formatted_reasoning,
                confidence=template["confidence"],
                target_metrics=template["target_metrics"],
                bid=min(1.0, 0.55 + float(template["confidence"]) * 0.45),
                action_cost=0.65,
            )

    def was_correct_at_step(self, step: int) -> bool:
        """Used by reward engine to determine R3."""
        if step >= len(self._cycle):
            return False
        return self._cycle[step]
