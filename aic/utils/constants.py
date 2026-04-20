# aic/utils/constants.py
"""
Centralized constants for the Adaptive Incident Choreographer.
All magic numbers live here — import from this module everywhere.
"""

# Episode configuration
SLA_STEPS: int = 20                    # Total steps per episode
T_DRIFT_MIN: int = 8                   # Earliest step for schema drift injection
T_DRIFT_MAX: int = 15                  # Latest step for schema drift injection
DEADLOCK_TIMEOUT_STEPS: int = 2        # Steps before deadlock is declared
NOISE_STD: float = 0.05                # Gaussian noise std on all metrics

# Causal coupling
ALPHA_DB_APP: float = 0.4              # DB→App latency coupling coefficient
DB_APP_LAG_STEPS: int = 2              # Lag before DB spike hits App metrics

# Reward weights
WEIGHT_DB: float = 0.35
WEIGHT_INFRA: float = 0.30
WEIGHT_APP: float = 0.35

# Reward component values
R2_SLA_BONUS_MAX: float = 50.0
SLA_HEALTH_THRESHOLD: float = 0.10    # Metrics within 10% of target for SLA bonus

R3_CORRECT_OVERRIDE: float = +15.0    # override applied, adversary was wrong
R3_CORRECT_TRUST: float = +5.0        # trusted, adversary was correct
R3_WRONG_OVERRIDE: float = -10.0      # override applied, adversary was correct
R3_WRONG_TRUST: float = -20.0         # trusted, adversary was wrong

R4_MAX_PER_STEP: float = 5.0
R4_MIN_PER_STEP: float = -5.0

DEADLOCK_PENALTY: float = -20.0
LOCK_HANDOFF_BONUS: float = +5.0

# Service names (keys used throughout)
SERVICE_DB: str = "db"
SERVICE_INFRA: str = "infra"
SERVICE_APP: str = "app"
SERVICES: list[str] = [SERVICE_DB, SERVICE_INFRA, SERVICE_APP]

# Agent names
AGENT_DB: str = "db_agent"
AGENT_INFRA: str = "infra_agent"
AGENT_APP: str = "app_agent"
AGENT_ADV: str = "adversarial_agent"
ALL_AGENTS: list[str] = [AGENT_DB, AGENT_INFRA, AGENT_APP, AGENT_ADV]

# Initial trust scores
INITIAL_TRUST: float = 0.5            # All agents start at 0.5
TRUST_UPDATE_RATE: float = 0.1        # Bayesian update step size

# Target (healthy) metric values
METRIC_TARGETS: dict[str, float] = {
    "db_latency_ms": 50.0,
    "conn_pool_pct": 60.0,
    "replication_lag_ms": 10.0,
    "cpu_pct": 45.0,
    "mem_pct": 60.0,
    "pod_restarts": 0.0,
    "net_io_mbps": 100.0,
    "error_rate_pct": 0.5,
    "p95_latency_ms": 200.0,
    "queue_depth": 50.0,
    "throughput_rps": 1000.0,
    "sla_compliance_pct": 99.9,
}

# Metric initial fault values (degraded starting state)
METRIC_FAULT_INIT: dict[str, float] = {
    "db_latency_ms": 850.0,
    "conn_pool_pct": 98.0,
    "replication_lag_ms": 450.0,
    "cpu_pct": 89.0,
    "mem_pct": 92.0,
    "pod_restarts": 7.0,
    "net_io_mbps": 380.0,
    "error_rate_pct": 18.5,
    "p95_latency_ms": 3200.0,
    "queue_depth": 890.0,
    "throughput_rps": 180.0,
    "sla_compliance_pct": 71.2,
}

# Observation space keys per agent
OBS_DB: list[str] = ["db_latency_ms", "conn_pool_pct", "replication_lag_ms"]
OBS_INFRA: list[str] = ["cpu_pct", "mem_pct", "pod_restarts", "net_io_mbps"]
OBS_APP: list[str] = ["error_rate_pct", "p95_latency_ms", "queue_depth"]

# Adversarial agent config
ADV_CORRECT_PROBABILITY: float = 0.5   # Target long-run accuracy
NUM_COUNTERFACTUAL_TEMPLATES: int = 6

# Schema drift types
DRIFT_FIELD_RENAME: str = "field_rename"
DRIFT_UNIT_SHIFT: str = "unit_shift"
DRIFT_SILENT_NULL: str = "silent_null"
DRIFT_TYPES: list[str] = [DRIFT_FIELD_RENAME, DRIFT_UNIT_SHIFT, DRIFT_SILENT_NULL]
NULL_DRIFT_DURATION: int = 3           # Steps that null drift persists

# Training
DEFAULT_EPISODES: int = 100
CHECKPOINT_INTERVAL: int = 25         # Save checkpoint every N episodes
MAX_TOKENS_AGENT: int = 512           # Max tokens per LLM agent response
TEMPERATURE_AGENT: float = 0.3        # Low temp for reliable outputs

# Dashboard
TRACE_HISTORY_WINDOW: int = 5         # Steps of trace history shown to orchestrator
DASHBOARD_REFRESH_SECONDS: int = 2
