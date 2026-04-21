# dashboard/components/impact_viz.py
"""
Executive Impact Panel — "War Room" components for the AIC dashboard.

Components:
1. Revenue Counter: Real-time "Money Saved" display
2. War-Room Timeline: Vertical log of incident lifecycle events
3. Benchmark Comparison: Side-by-side AIC vs baseline results
"""
from __future__ import annotations

from typing import Optional
import plotly.graph_objects as go


# Revenue model
REVENUE_PER_STEP_USD = 5000.0
REVENUE_PER_MINUTE_USD = 8333.0  # ~$500k/hour


def render_revenue_counter(
    healthy_steps: int,
    total_steps: int,
    mttr_steps: int,
) -> dict:
    """
    Compute revenue impact metrics.

    Returns dict with: revenue_saved, revenue_at_risk, uptime_pct, mttr_minutes
    """
    revenue_saved = healthy_steps * REVENUE_PER_STEP_USD
    revenue_at_risk = (total_steps - healthy_steps) * REVENUE_PER_STEP_USD
    uptime_pct = (healthy_steps / total_steps * 100) if total_steps > 0 else 0.0
    # Each step ≈ 3 minutes in production
    mttr_minutes = mttr_steps * 3

    return {
        "revenue_saved": revenue_saved,
        "revenue_at_risk": revenue_at_risk,
        "uptime_pct": round(uptime_pct, 1),
        "mttr_minutes": mttr_minutes,
    }


def render_war_room_timeline(
    events: list[dict],
    height: int = 400,
) -> go.Figure:
    """
    Render a vertical timeline of incident lifecycle events.

    Events format: [{"step": int, "type": str, "message": str, "severity": str}]
    Types: "incident", "hypothesis", "veto", "recovery", "runbook"
    """
    fig = go.Figure()

    type_colors = {
        "incident": "#ef4444",
        "hypothesis": "#8b5cf6",
        "veto": "#f59e0b",
        "recovery": "#10b981",
        "runbook": "#3b82f6",
        "simulation": "#06b6d4",
    }
    type_symbols = {
        "incident": "x",
        "hypothesis": "diamond",
        "veto": "triangle-up",
        "recovery": "star",
        "runbook": "square",
        "simulation": "circle",
    }

    if not events:
        events = [{"step": 0, "type": "incident", "message": "Waiting for data...", "severity": "info"}]

    x_vals = [e["step"] for e in events]
    y_vals = list(range(len(events) - 1, -1, -1))  # reversed for newest at top
    texts = [e["message"] for e in events]
    colors = [type_colors.get(e["type"], "#94a3b8") for e in events]
    symbols = [type_symbols.get(e["type"], "circle") for e in events]

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(
            size=14,
            color=colors,
            symbol=symbols,
            line=dict(width=1, color="rgba(255,255,255,0.3)"),
        ),
        text=texts,
        textposition="middle right",
        textfont=dict(size=10, color="#e2e8f0", family="Inter"),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17",
        height=height,
        margin=dict(l=20, r=200, t=30, b=20),
        xaxis=dict(title="Step", gridcolor="#2a3042"),
        yaxis=dict(visible=False),
        font=dict(family="Inter", color="#94a3b8"),
        title=dict(
            text="⏱️ War Room Timeline",
            font=dict(size=14),
            x=0.5,
        ),
    )

    return fig


def render_benchmark_comparison(
    results: list[dict],
    height: int = 350,
) -> go.Figure:
    """
    Render benchmark comparison bar chart.

    results: list of {policy_name, avg_mttr, avg_reward, ...}
    """
    fig = go.Figure()

    policies = [r["policy_name"] for r in results]
    mttrs = [r.get("avg_mttr", 20) for r in results]
    rewards = [r.get("avg_reward", 0) for r in results]

    policy_colors = {
        "AIC (Trained)": "#10b981",
        "AIC (Untrained)": "#6366f1",
        "HighestConfidenceOnly": "#f59e0b",
        "MajorityVote": "#ef4444",
        "NoTrustOrchestrator": "#94a3b8",
    }
    colors = [policy_colors.get(p, "#6b7280") for p in policies]

    fig.add_trace(go.Bar(
        x=policies,
        y=mttrs,
        name="MTTR (steps)",
        marker_color=colors,
        text=[f"{m:.0f}" for m in mttrs],
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17",
        height=height,
        margin=dict(l=20, r=20, t=40, b=60),
        yaxis=dict(title="MTTR (steps)", gridcolor="#2a3042"),
        font=dict(family="Inter", color="#94a3b8"),
        title=dict(
            text="📊 Benchmark: Mean Time to Recovery",
            font=dict(size=14),
            x=0.5,
        ),
    )

    return fig


def extract_timeline_events(trajectory: list[dict]) -> list[dict]:
    """
    Extract war-room timeline events from a trajectory.

    Scans trajectory for: incident start, hypothesis changes,
    verifier vetoes, and recovery detection.
    """
    events: list[dict] = []

    events.append({
        "step": 0,
        "type": "incident",
        "message": "🚨 Incident triggered",
        "severity": "critical",
    })

    prev_hypothesis = None
    recovered = False

    for step_data in trajectory:
        step = step_data.get("step", 0)
        trace = step_data.get("trace", {})
        health = step_data.get("health", 0.0)

        # Hypothesis changes
        hyp = trace.get("root_cause_hypothesis")
        if hyp and isinstance(hyp, dict):
            hyp_name = hyp.get("scenario_name")
            if hyp_name and hyp_name != prev_hypothesis:
                events.append({
                    "step": step,
                    "type": "hypothesis",
                    "message": f"🔍 Hypothesis: {hyp_name} ({hyp.get('confidence', 0):.0%})",
                    "severity": "info",
                })
                prev_hypothesis = hyp_name

        # Runbook evidence
        runbook = trace.get("runbook_evidence")
        if runbook and isinstance(runbook, dict) and runbook.get("incident_id"):
            events.append({
                "step": step,
                "type": "runbook",
                "message": f"📖 Runbook: {runbook.get('source_file', 'N/A')}",
                "severity": "info",
            })

        # Verifier vetoes
        verifier = trace.get("verifier_report")
        if verifier and isinstance(verifier, dict) and not verifier.get("approved", True):
            events.append({
                "step": step,
                "type": "veto",
                "message": f"🛡️ Verifier blocked unsafe action",
                "severity": "warning",
            })

        # Recovery detection
        if health > 0.5 and not recovered:
            events.append({
                "step": step,
                "type": "recovery",
                "message": f"✅ Recovery detected (health={health:.2f})",
                "severity": "success",
            })
            recovered = True

    return events
