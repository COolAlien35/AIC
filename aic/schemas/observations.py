# aic/schemas/observations.py
"""
Pydantic V2 observation models for each agent type.
Each sub-agent sees only its slice of the world state.
"""
from pydantic import BaseModel, Field


class DBObservation(BaseModel):
    """Observation visible to the DB agent."""
    db_latency_ms: float
    conn_pool_pct: float
    replication_lag_ms: float
    # Schema drift fields — may differ from expected
    raw_data: dict = Field(
        default_factory=dict,
        description="Raw API response before validation",
    )
    drift_detected: bool = False


class InfraObservation(BaseModel):
    """Observation visible to the Infra agent."""
    cpu_pct: float
    mem_pct: float
    pod_restarts: float
    net_io_mbps: float
    raw_data: dict = Field(default_factory=dict)
    drift_detected: bool = False


class AppObservation(BaseModel):
    """Observation visible to the App agent."""
    error_rate_pct: float
    p95_latency_ms: float
    queue_depth: float
    raw_data: dict = Field(default_factory=dict)
    drift_detected: bool = False


class OrchestratorObservation(BaseModel):
    """What the orchestrator sees each step — full situational awareness."""
    alert_summary_text: str
    sla_remaining_steps: int
    sub_agent_recommendations: list[dict]  # Serialized SubAgentRecommendation list
    trace_history: list[dict]              # Last N ExplanationTrace dicts
    current_trust_scores: dict[str, float]
    step: int
