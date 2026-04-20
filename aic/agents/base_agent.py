# aic/agents/base_agent.py
"""
Abstract base class for all sub-agents in the AIC system.
Each sub-agent receives a sliced observation (only its metrics)
and returns a structured recommendation.
"""
from abc import ABC, abstractmethod

from aic.schemas.traces import SubAgentRecommendation


class BaseSubAgent(ABC):
    """
    Abstract base class for all sub-agents.
    Each sub-agent receives a sliced observation (only its metrics)
    and returns a structured recommendation.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique identifier string for this agent."""
        ...

    @abstractmethod
    def recommend(
        self,
        observation: dict,
        step: int,
        episode_context: str = "",
    ) -> SubAgentRecommendation:
        """
        Given sliced observation, return a recommendation.

        Args:
            observation: Metric dict for this agent's observation space only.
                         May contain drifted/null values if schema drift active.
            step: Current episode step.
            episode_context: Optional additional context (fault description, etc.)

        Returns:
            SubAgentRecommendation with action, reasoning, confidence, target_metrics.
        """
        ...
