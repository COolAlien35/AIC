# dashboard/components/postmortem_panel.py
"""
Postmortem Panel — displays auto-generated incident postmortem at episode end.

Shows:
  - Tabbed view: Executive / Engineering / Customer-safe / Checklist
  - Key decision timeline
  - Debate highlights
  - Download as markdown
"""
from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from aic.agents.comms_agent import CommsAgent, PostmortemReport


def render_postmortem_panel(
    trajectory: list[dict],
    final_health: float,
    mttr_steps: int,
    scenario_name: str,
    total_reward: float = 0.0,
) -> None:
    """
    Render the postmortem panel for a completed episode.

    Args:
        trajectory: Full step trajectory list.
        final_health: Final episode health score.
        mttr_steps: Steps until recovery.
        scenario_name: Active scenario name.
        total_reward: Total episode reward.
    """
    comms = CommsAgent()

    # Extract root cause from last available trace
    rca = None
    debate_rounds_dicts = []
    biz_severity = "P2"
    revenue_saved = 0.0

    for step_data in trajectory:
        trace = step_data.get("trace", {})
        if trace.get("root_cause_hypothesis"):
            rca = trace["root_cause_hypothesis"]
        if trace.get("business_impact_snapshot", {}).get("severity_level"):
            biz_severity = trace["business_impact_snapshot"]["severity_level"]
        if trace.get("debate_transcript"):
            debate_rounds_dicts.append({"criticisms": [], "supports": [], "security_vetoes": [], "debate_changed_selection": False})
            for ev in trace["debate_transcript"]:
                t = ev.get("type", "")
                if t == "criticism":
                    debate_rounds_dicts[-1]["criticisms"].append(ev)
                elif t == "support":
                    debate_rounds_dicts[-1]["supports"].append(ev)
                elif t == "veto":
                    debate_rounds_dicts[-1]["security_vetoes"].append(ev)

    # Revenue: assume healthy steps × $5k
    revenue_saved = sum(
        5000.0 * step_data.get("health", 0.0)
        for step_data in trajectory
    )

    # Generate postmortem
    report: PostmortemReport = comms.generate_postmortem(
        scenario_name=scenario_name,
        episode_traces=trajectory,
        final_health=final_health,
        mttr_steps=mttr_steps,
        root_cause_hypothesis=rca,
        business_severity=biz_severity,
        debate_rounds=debate_rounds_dicts,
        total_revenue_saved=revenue_saved,
    )

    # ── Header ────────────────────────────────────────────────────────────
    sla_icon = "✅" if report.sla_met else "❌"
    health_pct = f"{report.final_health * 100:.1f}%"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(16,185,129,0.05));
            border: 1px solid rgba(59,130,246,0.3);
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <h3 style="margin:0;color:#3b82f6;">📋 Auto-Generated Postmortem</h3>
            <p style="color:#94a3b8;margin:6px 0 0 0;font-size:0.85rem;">
                {report.scenario_name} · {report.severity} · 
                Health: {health_pct} · SLA: {sla_icon} · 
                Revenue Protected: ${report.revenue_impact_usd:,.0f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabbed view ────────────────────────────────────────────────────────
    tab_exec, tab_eng, tab_cust, tab_check, tab_raw = st.tabs([
        "👔 Executive", "⚙️ Engineering", "👥 Customer-Safe", "✅ Checklist", "📄 Raw Markdown"
    ])

    with tab_exec:
        st.markdown(f"### Executive Summary")
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-radius: 10px;
                padding: 16px;
                color: #cbd5e1;
                font-size: 0.95rem;
                line-height: 1.7;
            ">
                {report.executive_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**Key Business Metrics:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", report.severity)
        c2.metric("MTTR", f"{report.mttr_steps} steps")
        c3.metric("SLA", "Met ✅" if report.sla_met else "Breached ❌")
        c4.metric("Revenue Protected", f"${report.revenue_impact_usd:,.0f}")

        st.divider()
        st.markdown("**Root Cause:**")
        st.info(report.root_cause)

    with tab_eng:
        st.markdown("### Engineering Summary")
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-radius: 10px;
                padding: 16px;
                color: #cbd5e1;
                font-size: 0.85rem;
                line-height: 1.6;
            ">
                {report.engineering_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**Incident Timeline:**")
        for t in report.timeline[:8]:
            st.markdown(
                f"- **Step {t.step}** `[{t.agent}]`: {t.event} — *{t.impact}*"
            )

        if report.key_decisions:
            st.divider()
            st.markdown("**Key AIC Decisions:**")
            for kd in report.key_decisions:
                st.markdown(f"- {kd}")

        if report.debate_highlights:
            st.divider()
            st.markdown("**Agent Debate Highlights:**")
            for dh in report.debate_highlights:
                st.markdown(f"> {dh}")

        if report.contributing_factors:
            st.divider()
            st.markdown("**Contributing Factors:**")
            for cf in report.contributing_factors:
                st.markdown(f"- {cf}")

    with tab_cust:
        st.markdown("### Customer Communication Draft")
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-radius: 10px;
                padding: 16px;
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.7;
            ">
                {report.customer_safe_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"SLO Impact: {report.slo_impact}")

    with tab_check:
        st.markdown("### ✅ Remediation Checklist")
        for item in report.remediation_checklist:
            st.checkbox(item, key=f"check_{item[:30]}")

        st.divider()
        st.markdown("### 🛡️ Prevention Recommendations")
        for rec in report.prevention_recommendations:
            st.markdown(f"- {rec}")

    with tab_raw:
        md = report.to_markdown()
        st.code(md, language="markdown")
        st.download_button(
            label="⬇️ Download Postmortem.md",
            data=md,
            file_name=f"postmortem_{scenario_name.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )
