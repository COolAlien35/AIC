# dashboard/components/leaderboard_panel.py
"""
Leaderboard Panel — Streamlit component for the Arena scoreboard.

Shows:
  - Gold/silver/bronze medals per rank
  - Color-coded rows (AIC = green, baselines = gray/red)
  - Composite score bar chart
  - Per-scenario radar chart
  - AIC advantage headline metrics
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from aic.evals.leaderboard import (
    load_leaderboard, get_aic_vs_best_baseline, get_scenario_results,
    LeaderboardEntry, MEDALS, POLICY_IS_AIC,
)

BG = "#0a0e17"
CARD = "#1a1f2e"
GRID = "#2a3042"
TEMPLATE = "plotly_dark"

_POLICY_COLORS = {
    "AIC (Trained)": "#10b981",
    "AIC (Untrained)": "#3b82f6",
    "Oracle (Upper Bound)": "#8b5cf6",
    "HighestConfidenceOnly": "#f59e0b",
    "MajorityVote": "#64748b",
    "NoTrustOrchestrator": "#ef4444",
    "RandomRecovery": "#6b7280",
}


def render_leaderboard_panel(arena_path: str = "logs/arena_results.json") -> None:
    """Render the full leaderboard panel."""
    entries = load_leaderboard(arena_path)

    if not entries:
        st.warning("⚠️ No arena results found. Run `python scripts/run_arena.py` first.")
        if st.button("🏃 Run Arena Now"):
            with st.spinner("Running arena benchmark (this takes ~60s)…"):
                try:
                    from aic.evals.arena import run_arena
                    run_arena(output_path=arena_path, verbose=False)
                    st.success("✅ Arena complete! Reload to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Arena run failed: {e}")
        return

    # ── Headline metrics ──────────────────────────────────────────────────
    adv = get_aic_vs_best_baseline(entries)
    if adv:
        st.markdown("### 🏆 AIC vs Best Baseline")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(
                "Composite Score", f"{adv['composite_advantage']:+.3f}",
                help="AIC composite score advantage over best baseline",
                delta_color="normal" if adv['composite_advantage'] > 0 else "inverse",
            )
        with c2:
            st.metric(
                "MTTR Improvement", f"{adv['mttr_improvement_steps']:.1f} steps",
                delta="faster", delta_color="normal" if adv['mttr_improvement_steps'] > 0 else "inverse",
            )
        with c3:
            st.metric(
                "SLA Success", f"+{adv['sla_improvement_pct']:.1f}%",
                delta_color="normal" if adv['sla_improvement_pct'] > 0 else "inverse",
            )
        with c4:
            st.metric(
                "Revenue Saved", f"+${adv['revenue_delta_usd']:,.0f}",
                delta_color="normal" if adv['revenue_delta_usd'] > 0 else "inverse",
            )
        with c5:
            st.metric(
                "Scenario Wins", f"{adv['scenario_wins_aic']}/6",
                delta=f"vs {adv['scenario_wins_best_baseline']}/6 baseline",
                delta_color="normal",
            )

    st.divider()

    # ── Leaderboard table ─────────────────────────────────────────────────
    st.markdown("### 📊 Full Leaderboard")

    for entry in entries:
        medal = MEDALS.get(entry.rank, "  ")
        color = _POLICY_COLORS.get(entry.policy, "#64748b")
        is_aic = entry.policy in POLICY_IS_AIC
        bg_opacity = "0.12" if is_aic else "0.04"

        score_bar_pct = int(entry.composite_score * 100)

        st.markdown(
            f"""
            <div style="
                background: rgba({('16,185,129' if is_aic else '100,116,139')},{bg_opacity});
                border-left: 4px solid {color};
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 8px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.3rem;">{medal}</span>
                        <strong style="color: {color}; font-size: 1rem; margin-left: 8px;">
                            #{entry.rank} {entry.policy}
                        </strong>
                        {'<span style="background:#10b981;color:#0a0e17;padding:2px 8px;border-radius:12px;font-size:0.7rem;margin-left:8px;font-weight:700;">AIC</span>' if is_aic else ''}
                    </div>
                    <div style="text-align: right; color: #94a3b8; font-size: 0.85rem;">
                        Score: <strong style="color:{color};">{entry.composite_score:.3f}</strong>
                        &nbsp;|&nbsp; MTTR: {entry.avg_mttr:.1f}
                        &nbsp;|&nbsp; SLA: {entry.sla_success_rate:.0f}%
                        &nbsp;|&nbsp; Wins: {entry.scenario_wins}/6
                    </div>
                </div>
                <div style="
                    margin-top: 8px;
                    height: 6px;
                    background: #2a3042;
                    border-radius: 3px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {score_bar_pct}%;
                        height: 100%;
                        background: linear-gradient(90deg, {color}, {color}88);
                        border-radius: 3px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Composite score bar chart ─────────────────────────────────────────
    st.markdown("### 📈 Composite Score Comparison")

    policies = [e.policy for e in entries]
    scores = [e.composite_score for e in entries]
    colors = [_POLICY_COLORS.get(p, "#64748b") for p in policies]

    fig_bar = go.Figure(go.Bar(
        x=scores,
        y=policies,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig_bar.update_layout(
        template=TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=300,
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis=dict(title="Composite Score", gridcolor=GRID, range=[0, 1.05]),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter", color="#94a3b8"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar chart by metric ─────────────────────────────────────────────
    st.markdown("### 🕸️ Multi-Metric Radar")
    radar_policies = [e for e in entries if e.rank <= 4]

    categories = ["MTTR", "SLA%", "Adv Suppr.", "Safety", "Revenue"]

    fig_radar = go.Figure()
    for e in radar_policies:
        color = _POLICY_COLORS.get(e.policy, "#64748b")
        # Normalize each metric to [0,1] for radar
        mttr_norm = 1.0 - (e.avg_mttr / 20.0)
        sla_norm = e.sla_success_rate / 100.0
        adv_norm = e.adversary_suppression_rate / 100.0
        safety_norm = 1.0 - (e.unsafe_action_rate / 100.0)
        rev_norm = min(1.0, e.total_revenue_saved_usd / 100_000.0)

        vals = [mttr_norm, sla_norm, adv_norm, safety_norm, rev_norm]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=e.policy,
            line=dict(color=color),
            fillcolor=f"{color}22",
        ))

    fig_radar.update_layout(
        template=TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID),
            bgcolor=BG,
        ),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        font=dict(family="Inter", color="#94a3b8"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
