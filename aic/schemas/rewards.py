"""Typed reward models for OpenEnv evaluation and API responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AICReward(BaseModel):
    """Normalized OpenEnv reward wrapper.

    ``score`` is the submission-facing 0.0-1.0 score. ``raw_reward`` preserves
    the native environment reward for auditability and plots.
    """

    score: float = Field(ge=0.0, le=1.0)
    raw_reward: float
    components: dict[str, float] = Field(default_factory=dict)
    success: bool = False
    health: float = Field(ge=0.0, le=1.0)
    done: bool = False

