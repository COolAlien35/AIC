# aic/env/service_topology.py
"""
Service Topology Graph — Directed Acyclic Graph (DAG) of service dependencies.

Topology:
    Gateway → App → Cache → DB
                  → Queue → DB

Pressure propagation uses a 1-step buffer: a delta applied to the DB at step T
only affects App at step T+1 (via flush_propagation_buffer).

Edge weights (coupling coefficients) control how much pressure flows downstream.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServiceNode:
    """A single service in the topology graph."""
    name: str
    health: float = 1.0       # 0.0 (dead) to 1.0 (healthy)
    load: float = 0.0         # current load pressure (arbitrary units)
    latency: float = 0.0      # accumulated latency contribution (ms)
    error_rate: float = 0.0   # accumulated error rate contribution (%)
    cooldown_rate: float = 0.95  # per-step decay toward baseline

    def apply_delta(self, delta: dict[str, float]) -> None:
        """Apply a pressure delta to this node."""
        self.load += delta.get("load", 0.0)
        self.latency += delta.get("latency", 0.0)
        self.error_rate += delta.get("error_rate", 0.0)
        # Recompute health based on accumulated pressure
        pressure = (
            min(self.load / 100.0, 1.0) * 0.3
            + min(self.latency / 5000.0, 1.0) * 0.4
            + min(self.error_rate / 50.0, 1.0) * 0.3
        )
        self.health = max(0.0, 1.0 - pressure)

    def cool_down(self) -> None:
        """Decay elevated metrics toward baseline (zero pressure)."""
        self.load *= self.cooldown_rate
        self.latency *= self.cooldown_rate
        self.error_rate *= self.cooldown_rate
        # Recompute health
        pressure = (
            min(self.load / 100.0, 1.0) * 0.3
            + min(self.latency / 5000.0, 1.0) * 0.4
            + min(self.error_rate / 50.0, 1.0) * 0.3
        )
        self.health = max(0.0, 1.0 - pressure)

    def snapshot(self) -> dict[str, float]:
        """Return current state as a dict."""
        return {
            "health": round(self.health, 4),
            "load": round(self.load, 4),
            "latency": round(self.latency, 4),
            "error_rate": round(self.error_rate, 4),
        }


# Default topology edges: (source, target, coupling_coefficient)
DEFAULT_EDGES: list[tuple[str, str, float]] = [
    ("gateway", "app", 0.8),
    ("app", "cache", 0.6),
    ("app", "queue", 0.5),
    ("cache", "db", 0.7),
    ("queue", "db", 0.4),
]

DEFAULT_NODES: list[str] = ["gateway", "app", "cache", "queue", "db"]


class ServiceTopology:
    """
    Directed graph of service dependencies with buffered pressure propagation.

    Pressure applied to a node at step T propagates to downstream dependents
    at step T+1 via the propagation buffer.
    """

    def __init__(
        self,
        nodes: Optional[list[str]] = None,
        edges: Optional[list[tuple[str, str, float]]] = None,
        cooldown_rate: float = 0.95,
    ):
        node_names = nodes or DEFAULT_NODES
        edge_list = edges or DEFAULT_EDGES

        # Build node registry
        self.nodes: dict[str, ServiceNode] = {
            name: ServiceNode(name=name, cooldown_rate=cooldown_rate)
            for name in node_names
        }

        # Adjacency list: source → [(target, weight)]
        self._downstream: dict[str, list[tuple[str, float]]] = {
            name: [] for name in node_names
        }
        # Reverse adjacency: target → [(source, weight)]
        self._upstream: dict[str, list[tuple[str, float]]] = {
            name: [] for name in node_names
        }

        for src, dst, weight in edge_list:
            if src in self.nodes and dst in self.nodes:
                self._downstream[src].append((dst, weight))
                self._upstream[dst].append((src, weight))

        # Propagation buffer: list of (node_name, delta_dict) to apply next step
        self._propagation_buffer: list[tuple[str, dict[str, float]]] = []

    def reset(self) -> None:
        """Reset all nodes to healthy baseline and clear buffers."""
        for node in self.nodes.values():
            node.health = 1.0
            node.load = 0.0
            node.latency = 0.0
            node.error_rate = 0.0
        self._propagation_buffer.clear()

    def propagate_pressure(
        self, node_name: str, delta: dict[str, float]
    ) -> None:
        """
        Apply pressure delta to the named node immediately.
        Downstream effects are BUFFERED for the next step.

        Args:
            node_name: Name of the node receiving direct pressure.
            delta: Dict with keys 'load', 'latency', 'error_rate'.
        """
        if node_name not in self.nodes:
            return

        # Apply delta to the target node immediately
        self.nodes[node_name].apply_delta(delta)

        # Buffer downstream propagation (BFS with visited set for cycle safety)
        visited: set[str] = {node_name}
        queue: deque[tuple[str, dict[str, float]]] = deque()

        # Enqueue downstream neighbors
        for downstream_name, weight in self._downstream.get(node_name, []):
            if downstream_name not in visited:
                scaled_delta = {k: v * weight for k, v in delta.items()}
                queue.append((downstream_name, scaled_delta))
                visited.add(downstream_name)

        # BFS to collect all downstream effects
        while queue:
            current_name, current_delta = queue.popleft()
            # Buffer this effect for next step
            self._propagation_buffer.append((current_name, current_delta))

            # Continue propagation further downstream
            for next_name, weight in self._downstream.get(current_name, []):
                if next_name not in visited:
                    next_delta = {
                        k: v * weight for k, v in current_delta.items()
                    }
                    queue.append((next_name, next_delta))
                    visited.add(next_name)

    def flush_propagation_buffer(self) -> None:
        """
        Apply all buffered downstream effects from the previous step.
        Call this once at the beginning of each new step.
        """
        for node_name, delta in self._propagation_buffer:
            if node_name in self.nodes:
                self.nodes[node_name].apply_delta(delta)
        self._propagation_buffer.clear()

    def cool_down(self) -> None:
        """Apply natural cooldown to all nodes."""
        for node in self.nodes.values():
            node.cool_down()

    def get_node(self, name: str) -> Optional[ServiceNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def get_topology_state(self) -> dict[str, dict[str, float]]:
        """Return a snapshot of all node states for logging."""
        return {name: node.snapshot() for name, node in self.nodes.items()}

    def get_flat_state(self) -> dict[str, float]:
        """Return a flat dict of all node metrics (prefixed by node name)."""
        flat: dict[str, float] = {}
        for name, node in self.nodes.items():
            snap = node.snapshot()
            for metric, value in snap.items():
                flat[f"{name}_{metric}"] = value
        return flat

    def get_downstream(self, node_name: str) -> list[tuple[str, float]]:
        """Return downstream neighbors and their coupling weights."""
        return self._downstream.get(node_name, [])

    def get_upstream(self, node_name: str) -> list[tuple[str, float]]:
        """Return upstream neighbors and their coupling weights."""
        return self._upstream.get(node_name, [])
