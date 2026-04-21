# dashboard/components/topology_viz.py
"""
Live Topology Map — visual DAG of service dependencies.

Renders the Gateway → App → Cache/Queue → DB topology with
nodes colored by health (green/yellow/red) and edges showing
coupling pressure.
"""
from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


# Node positions for the DAG layout
NODE_POSITIONS: dict[str, tuple[float, float]] = {
    "gateway": (0.5, 1.0),
    "app": (0.5, 0.75),
    "cache": (0.25, 0.5),
    "queue": (0.75, 0.5),
    "db": (0.5, 0.25),
}

# Edge connections: (source, target)
EDGES: list[tuple[str, str]] = [
    ("gateway", "app"),
    ("app", "cache"),
    ("app", "queue"),
    ("cache", "db"),
    ("queue", "db"),
]

# Display names
NODE_LABELS: dict[str, str] = {
    "gateway": "🌐 Gateway",
    "app": "📱 App",
    "cache": "⚡ Cache",
    "queue": "📨 Queue",
    "db": "🗄️ DB",
}


def _health_color(health: float) -> str:
    """Map health score to color."""
    if health > 0.7:
        return "#34d399"   # emerald
    elif health > 0.4:
        return "#fbbf24"   # gold
    else:
        return "#fb7185"   # rose


def _health_size(health: float) -> int:
    """Map health to node size (unhealthy = larger, attention-drawing)."""
    return max(35, int(60 - health * 30))


def render_topology_map(
    topology_state: Optional[dict[str, dict[str, float]]] = None,
    root_cause_node: Optional[str] = None,
    height: int = 400,
) -> go.Figure:
    """
    Render the service topology DAG with health-based coloring.

    Args:
        topology_state: Dict of {node_name: {health, load, latency, error_rate}}
        root_cause_node: If set, this node gets a pulsing border
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Default healthy state
    if topology_state is None:
        topology_state = {
            name: {"health": 1.0, "load": 0.0, "latency": 0.0, "error_rate": 0.0}
            for name in NODE_POSITIONS
        }

    # Draw edges
    for src, dst in EDGES:
        x0, y0 = NODE_POSITIONS[src]
        x1, y1 = NODE_POSITIONS[dst]
        src_health = topology_state.get(src, {}).get("health", 1.0)
        dst_health = topology_state.get(dst, {}).get("health", 1.0)
        avg_health = (src_health + dst_health) / 2

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(
                color=_health_color(avg_health),
                width=2 if avg_health > 0.7 else 3,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Draw nodes
    for name, (x, y) in NODE_POSITIONS.items():
        state = topology_state.get(name, {})
        health = state.get("health", 1.0)
        load = state.get("load", 0.0)
        latency = state.get("latency", 0.0)
        error_rate = state.get("error_rate", 0.0)
        color = _health_color(health)
        size = _health_size(health)

        is_root = (name == root_cause_node)

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=size,
                color=color,
                opacity=0.9,
                line=dict(
                    width=4 if is_root else 2,
                    color="#fbbf24" if is_root else "rgba(52,211,153,0.2)",
                ),
                symbol="circle",
            ),
            text=[NODE_LABELS.get(name, name)],
            textposition="bottom center",
            textfont=dict(
                size=12,
                color="white",
                family="Inter",
            ),
            hovertemplate=(
                f"<b>{name.title()}</b><br>"
                f"Health: {health:.2f}<br>"
                f"Load: {load:.1f}<br>"
                f"Latency: {latency:.1f}ms<br>"
                f"Error Rate: {error_rate:.1f}%<br>"
                f"<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0c0f0a",
        plot_bgcolor="#0c0f0a",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            visible=False, range=[-0.1, 1.1],
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False, range=[0.1, 1.15],
            fixedrange=True,
        ),
        font=dict(family="Inter", color="#9ca89a"),
        title=dict(
            text="🔗 Service Topology",
            font=dict(size=14),
            x=0.5,
        ),
    )

    return fig
