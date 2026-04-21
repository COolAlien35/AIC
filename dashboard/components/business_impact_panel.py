# dashboard/components/business_impact_panel.py
"""
Business Guardian Panel — revenue and SLO impact visualization.

Shows:
  - Live cumulative revenue loss counter (per step)
  - AIC vs baseline revenue loss comparison
  - Users impacted tracker
  - SLO compliance severity gauge
  - Enterprise customer impact
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

BG = "#0c0f0a"
GRID = "#1e2b1a"
TEMPLATE = "plotly_dark"

REVENUE_PER_STEP_USD = 5000.0  # matches benchmark_suite
BASE_USERS = 50_000
ENTERPRISE_CUSTOMERS = 12  # hypothetical


def render_business_impact_panel(
    trajectory: list[dict],
    step: int,
    scenario_name: str = "",
) -> None:
    """
    Render the Business Guardian panel for the current episode.

    Args:
        trajectory: Full episode trajectory list.
        step: Current step being viewed.
        scenario_name: Active scenario name.
    """
    # Compute cumulative revenue loss and users impacted per step
    rev_loss_curve: list[float] = []
    users_curve: list[int] = []
    compliance_curve: list[float] = []

    for step_data in trajectory:
        biz = step_data.get("trace", {}).get("business_impact_snapshot") or {}
        rev_loss_curve.append(biz.get("revenue_loss_per_minute", 0.0))
        users_curve.append(biz.get("users_impacted", 0))
        compliance_curve.append(biz.get("compliance_risk_score", 0.0))

    # Current step values
    current_biz = {}
    if step < len(trajectory):
        current_biz = trajectory[step].get("trace", {}).get("business_impact_snapshot") or {}

    rev_now = current_biz.get("revenue_loss_per_minute", 0.0)
    users_now = current_biz.get("users_impacted", 0)
    severity = current_biz.get("severity_level", "P4")
    compliance_risk = current_biz.get("compliance_risk_score", 0.0)

    # Revenue protected (healthy steps × rev_per_step)
    health_at_step = trajectory[step].get("health", 0.0) if step < len(trajectory) else 0.0
    revenue_protected = sum(
        REVENUE_PER_STEP_USD * t.get("health", 0.0)
        for t in trajectory[:step + 1]
    )

    # ── Headline metrics ──────────────────────────────────────────────────
    severity_colors = {"P1": "#fb7185", "P2": "#fbbf24", "P3": "#14b8a6", "P4": "#34d399"}
    sev_color = severity_colors.get(severity, "#6b7a68")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(251,191,36,0.06), rgba(251,113,133,0.04));
            border: 1px solid rgba(251,191,36,0.2);
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 14px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            animation: fadeInUp 0.4s ease-out both;
        ">
            <span style="color:{sev_color};font-weight:600;font-size:1rem;font-family:'Inter',sans-serif;">
                🚨 Incident Severity: {severity}
            </span>
            {'&nbsp;&nbsp;<span style="color:#fb7185;font-size:0.75rem;font-family:JetBrains Mono,monospace;">● ACTIVE COMPLIANCE RISK</span>' if compliance_risk > 0.5 else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "💸 Rev. Loss/min",
            f"${rev_now:,.0f}",
            delta=f"${rev_now - rev_loss_curve[max(0, step-1)]:+.0f}" if step > 0 else None,
            delta_color="inverse",
        )
    with c2:
        st.metric(
            "👥 Users Impacted",
            f"{users_now:,}",
            delta=f"{users_now - users_curve[max(0, step-1)]:+,}" if step > 0 else None,
            delta_color="inverse",
        )
    with c3:
        st.metric(
            "🛡️ Revenue Protected",
            f"${revenue_protected:,.0f}",
            delta="AIC recovery value",
            delta_color="normal",
        )
    with c4:
        ent_impacted = int(ENTERPRISE_CUSTOMERS * compliance_risk)
        st.metric(
            "🏢 Enterprise Impacted",
            f"{ent_impacted}/{ENTERPRISE_CUSTOMERS}",
            delta_color="off",
        )

    st.divider()

    # ── Revenue loss curve ─────────────────────────────────────────────────
    st.markdown("### 💸 Revenue Loss Rate Over Time")

    fig_rev = go.Figure()
    steps = list(range(len(rev_loss_curve)))

    fig_rev.add_trace(go.Scatter(
        x=steps, y=rev_loss_curve,
        name="Revenue Loss/min",
        fill="tozeroy",
        line=dict(color="#fb7185", width=2),
        fillcolor="rgba(251,113,133,0.08)",
        mode="lines",
    ))

    # Mark current step
    if step < len(rev_loss_curve):
        fig_rev.add_vline(
            x=step, line_dash="dash", line_color="rgba(255,255,255,0.3)",
            annotation_text=f"Step {step}", annotation_position="top",
        )

    fig_rev.update_layout(
        template=TEMPLATE,
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=200,
        margin=dict(l=20, r=20, t=20, b=30),
        yaxis=dict(title="$/min", gridcolor=GRID),
        xaxis=dict(title="Step", gridcolor=GRID),
        font=dict(family="Inter", color="#9ca89a"),
        showlegend=False,
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # ── Compliance risk gauge ──────────────────────────────────────────────
    st.markdown("### 🔒 SLO Compliance Risk")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=compliance_risk * 100,
        title={"text": "Compliance Risk Score", "font": {"color": "#9ca89a", "size": 14}},
        delta={"reference": 0, "suffix": "%"},
        number={"suffix": "%", "font": {"color": "#9ca89a"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca89a"},
            "bar": {"color": sev_color},
            "bgcolor": GRID,
            "bordercolor": "#1e2b1a",
            "steps": [
                {"range": [0, 20], "color": "rgba(52,211,153,0.15)"},
                {"range": [20, 50], "color": "rgba(251,191,36,0.12)"},
                {"range": [50, 80], "color": "rgba(251,113,133,0.1)"},
                {"range": [80, 100], "color": "rgba(251,113,133,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#fb7185", "width": 2},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ))

    fig_gauge.update_layout(
        paper_bgcolor=BG,
        height=220,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter", color="#9ca89a"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SLO impact text ────────────────────────────────────────────────────
    if compliance_risk > 0.7:
        st.error("⚠️ **Regulatory compliance at risk** — SLA breach exceeds enterprise tier thresholds.")
    elif compliance_risk > 0.4:
        st.warning("⚠️ SLO error budget partially consumed. Enterprise tiers at risk.")
    else:
        st.success("✅ SLO compliance within acceptable parameters.")
