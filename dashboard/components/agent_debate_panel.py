# dashboard/components/agent_debate_panel.py
"""
Agent Debate Panel — shows the debate transcript for each step.

Displays:
  - Criticisms (orange)
  - Security vetoes (red)
  - Supports (green)
  - Whether debate changed the selection
"""
from __future__ import annotations

import streamlit as st


def render_debate_panel(trace: dict) -> None:
    """
    Render the agent debate panel for a single step trace.

    Args:
        trace: Step trace dict from trajectory. Expected keys:
               debate_transcript, commander_mode, commander_brief
    """
    debate = trace.get("debate_transcript") or []
    commander_brief = trace.get("commander_brief")
    commander_mode = trace.get("commander_mode", "fastest_recovery")

    # ── Commander brief ────────────────────────────────────────────────────
    if commander_brief:
        mode_colors = {
            "fastest_recovery": "#f59e0b",
            "safest_recovery": "#3b82f6",
            "protect_data": "#8b5cf6",
            "minimize_user_impact": "#10b981",
            "contain_compromise": "#ef4444",
        }
        color = mode_colors.get(commander_mode, "#94a3b8")
        st.markdown(
            f"""
            <div style="
                border-left: 3px solid {color};
                background: rgba(255,255,255,0.03);
                border-radius: 0 8px 8px 0;
                padding: 8px 12px;
                margin-bottom: 12px;
                font-size: 0.85rem;
                color: {color};
            ">
                {commander_brief}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Debate transcript ──────────────────────────────────────────────────
    if not debate:
        st.caption("✅ No debate activity this step — agents aligned.")
        return

    st.markdown(f"**{len(debate)} debate event(s) this step:**")

    for event in debate:
        event_type = event.get("type", "criticism")
        text = event.get("text") or event.get("reason", "")

        if event_type == "veto":
            icon = "🔒"
            color = "#ef4444"
            bg = "rgba(239,68,68,0.08)"
            label = "SECURITY VETO"
        elif event_type == "criticism":
            icon = "⚠️"
            color = "#f59e0b"
            bg = "rgba(245,158,11,0.08)"
            severity = event.get("severity", "medium")
            label = f"CHALLENGE ({severity.upper()})"
        else:  # support
            icon = "✅"
            color = "#10b981"
            bg = "rgba(16,185,129,0.08)"
            label = "SUPPORTS"

        critic = event.get("critic") or event.get("supporter", "")
        target = event.get("target_agent", "")

        st.markdown(
            f"""
            <div style="
                background: {bg};
                border-left: 3px solid {color};
                border-radius: 0 8px 8px 0;
                padding: 8px 12px;
                margin-bottom: 6px;
                font-size: 0.8rem;
            ">
                <span style="color:{color};font-weight:700;">{icon} {label}</span>
                <span style="color:#64748b;"> — {critic} → {target}</span><br>
                <span style="color:#cbd5e1;">{text[:200]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Check if debate changed selection
    debate_changed = trace.get("debate_changed_selection", False)
    if debate_changed:
        st.warning("⚡ **Debate changed the agent selection this step!**")


def render_debate_summary_panel(episode_traces: list[dict]) -> None:
    """Render a summary of debate activity across the full episode."""
    total_criticisms = 0
    total_vetoes = 0
    changed_steps = 0

    for step_data in episode_traces:
        trace = step_data.get("trace", {})
        debate = trace.get("debate_transcript") or []
        for ev in debate:
            if ev.get("type") == "veto":
                total_vetoes += 1
            elif ev.get("type") == "criticism":
                total_criticisms += 1
        if trace.get("debate_changed_selection"):
            changed_steps += 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Criticisms", total_criticisms)
    with c2:
        st.metric("Security Vetoes", total_vetoes, delta_color="off")
    with c3:
        st.metric("Selection Changes", changed_steps,
                  help="Steps where debate overrode the pre-debate top pick")
