"""Utility helpers for war-room style rollouts and dashboards."""
from __future__ import annotations

from typing import Optional

from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import METRIC_TARGETS


# Fallback effects for core world-state metrics when a recommendation does not
# provide explicit expected_impact values.
DEFAULT_ACTION_EFFECTS: dict[str, float] = {
    "db_latency_ms": -150.0,
    "conn_pool_pct": -12.0,
    "replication_lag_ms": -90.0,
    "cpu_pct": -8.0,
    "mem_pct": -8.0,
    "pod_restarts": -1.0,
    "net_io_mbps": -35.0,
    "error_rate_pct": -2.5,
    "p95_latency_ms": -180.0,
    "queue_depth": -75.0,
    "throughput_rps": +120.0,
    "sla_compliance_pct": +4.0,
}


# Synthetic specialist metrics are mapped back to the 12-metric world state so
# Network/Security actions can have concrete effect in rollouts.
SYNTHETIC_TO_WORLD_EFFECTS: dict[str, dict[str, float]] = {
    "packet_loss_pct": {
        "error_rate_pct": -1.2,
        "p95_latency_ms": -120.0,
    },
    "dns_latency_ms": {
        "p95_latency_ms": -160.0,
        "throughput_rps": +45.0,
    },
    "lb_5xx_count": {
        "error_rate_pct": -1.5,
        "p95_latency_ms": -140.0,
    },
    "regional_latency_ms": {
        "p95_latency_ms": -220.0,
        "net_io_mbps": -30.0,
        "throughput_rps": +60.0,
    },
    "auth_failure_rate": {
        "error_rate_pct": -2.0,
        "sla_compliance_pct": +2.5,
    },
    "suspicious_token_count": {
        "error_rate_pct": -1.0,
        "throughput_rps": +30.0,
    },
    "compromised_ip_count": {
        "error_rate_pct": -1.2,
        "sla_compliance_pct": +1.8,
    },
}


def build_network_observation(metrics: dict[str, float]) -> dict[str, float]:
    """Derive network specialist observation from the 12-metric world state."""
    net_io = metrics.get("net_io_mbps", 100.0)
    p95 = metrics.get("p95_latency_ms", 200.0)
    error_rate = metrics.get("error_rate_pct", 0.5)
    throughput = metrics.get("throughput_rps", 1000.0)

    return {
        "packet_loss_pct": round(max(0.0, (net_io - 100.0) / 30.0), 2),
        "dns_latency_ms": round(max(5.0, p95 / 18.0), 2),
        "lb_5xx_count": round(max(0.0, error_rate * 3.5), 2),
        "regional_latency_ms": round(max(30.0, (net_io / 3.0) + (1000.0 - throughput) / 12.0), 2),
    }


def build_security_observation(metrics: dict[str, float]) -> dict[str, float]:
    """Derive security specialist observation from the 12-metric world state."""
    error_rate = metrics.get("error_rate_pct", 0.5)
    throughput = metrics.get("throughput_rps", 1000.0)
    sla = metrics.get("sla_compliance_pct", 99.9)

    suspicious_tokens = max(0, int((1000.0 - throughput) / 120.0))
    compromised_ips = max(0, int((100.0 - sla) / 6.0))

    return {
        "auth_failure_rate": round(max(0.1, error_rate * 0.85), 2),
        "suspicious_token_count": suspicious_tokens,
        "compromised_ip_count": compromised_ips,
    }


def build_action_deltas(recommendation: SubAgentRecommendation) -> dict[str, float]:
    """
    Convert a recommendation into actionable world-state deltas.

    Priority order:
    1. explicit expected_impact on known world metrics,
    2. synthetic specialist metric mapping,
    3. default fallback effect for core target metrics.
    """
    action_deltas: dict[str, float] = {}

    for metric, delta in (recommendation.expected_impact or {}).items():
        if metric in METRIC_TARGETS:
            action_deltas[metric] = action_deltas.get(metric, 0.0) + float(delta)

    if action_deltas:
        return action_deltas

    for metric in recommendation.target_metrics:
        if metric in SYNTHETIC_TO_WORLD_EFFECTS:
            for mapped_metric, mapped_delta in SYNTHETIC_TO_WORLD_EFFECTS[metric].items():
                action_deltas[mapped_metric] = action_deltas.get(mapped_metric, 0.0) + mapped_delta
        elif metric in DEFAULT_ACTION_EFFECTS:
            action_deltas[metric] = action_deltas.get(metric, 0.0) + DEFAULT_ACTION_EFFECTS[metric]

    return action_deltas


def project_metrics_to_topology_state(
    metrics: dict[str, float],
    root_cause_node: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """Project flat world metrics into a 5-node topology state for visualization."""

    def _norm(metric_name: str, invert: bool = False) -> float:
        target = METRIC_TARGETS.get(metric_name, 1.0)
        value = metrics.get(metric_name, target)
        if target == 0.0:
            ratio = min(1.0, abs(value) / 10.0)
        else:
            ratio = min(1.0, abs(value - target) / max(abs(target), 1e-6))
        return 1.0 - ratio if invert else ratio

    def _state(load: float, latency: float, error_rate: float) -> dict[str, float]:
        pressure = min(1.0, (load / 100.0) * 0.35 + (latency / 5000.0) * 0.4 + (error_rate / 100.0) * 0.25)
        return {
            "health": round(max(0.0, 1.0 - pressure), 4),
            "load": round(load, 4),
            "latency": round(latency, 4),
            "error_rate": round(error_rate, 4),
        }

    gateway_load = max(0.0, metrics.get("net_io_mbps", 100.0) - 100.0)
    gateway_latency = max(0.0, metrics.get("p95_latency_ms", 200.0) * 0.4)
    gateway_error = max(0.0, metrics.get("error_rate_pct", 0.5) * 1.2)

    app_load = max(0.0, 1000.0 - metrics.get("throughput_rps", 1000.0)) / 10.0
    app_latency = max(0.0, metrics.get("p95_latency_ms", 200.0))
    app_error = max(0.0, metrics.get("error_rate_pct", 0.5) * 2.0)

    cache_load = max(0.0, metrics.get("queue_depth", 50.0) - 50.0) / 8.0
    cache_latency = max(0.0, metrics.get("p95_latency_ms", 200.0) * 0.3)
    cache_error = max(0.0, metrics.get("error_rate_pct", 0.5))

    queue_load = max(0.0, metrics.get("queue_depth", 50.0) - 50.0) / 5.0
    queue_latency = max(0.0, metrics.get("p95_latency_ms", 200.0) * 0.45)
    queue_error = max(0.0, metrics.get("error_rate_pct", 0.5) * 1.1)

    db_load = max(0.0, metrics.get("conn_pool_pct", 60.0) - 60.0)
    db_latency = max(0.0, metrics.get("db_latency_ms", 50.0))
    db_error = max(0.0, metrics.get("replication_lag_ms", 10.0) / 20.0)

    state = {
        "gateway": _state(gateway_load, gateway_latency, gateway_error),
        "app": _state(app_load, app_latency, app_error),
        "cache": _state(cache_load, cache_latency, cache_error),
        "queue": _state(queue_load, queue_latency, queue_error),
        "db": _state(db_load, db_latency, db_error),
    }

    if root_cause_node and root_cause_node in state:
        # Make the root cause visually stand out just a bit more.
        state[root_cause_node]["health"] = round(max(0.0, state[root_cause_node]["health"] - 0.08), 4)

    return state