#!/usr/bin/env python3
# dashboard/app.py
"""
AIC War Room Dashboard — 5-tab Streamlit application.

Tabs:
  🚨 Mission Control — world state, trust, traces, reward, debate
  🏆 Leaderboard     — arena benchmark scoreboard
  🎮 Judge Challenge  — interactive scenario runner
  💼 Business Impact  — revenue/SLO guardian
  📋 Postmortem       — auto-generated incident docs

Usage:
    streamlit run dashboard/app.py
"""
import pickle
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.utils.constants import (
    METRIC_TARGETS, METRIC_FAULT_INIT, SLA_STEPS, ALL_AGENTS, AGENT_ADV,
    R3_CORRECT_OVERRIDE, R3_CORRECT_TRUST, R3_WRONG_OVERRIDE, R3_WRONG_TRUST,
    R4_MAX_PER_STEP, R4_MIN_PER_STEP, INITIAL_TRUST,
)

# New component imports
try:
    from dashboard.components.leaderboard_panel import render_leaderboard_panel
    from dashboard.components.judge_challenge_mode import render_judge_challenge_panel
    from dashboard.components.business_impact_panel import render_business_impact_panel
    from dashboard.components.postmortem_panel import render_postmortem_panel
    from dashboard.components.agent_debate_panel import render_debate_panel, render_debate_summary_panel
    from dashboard.components.topology_viz import render_topology_map
    from dashboard.components.impact_viz import extract_timeline_events, render_war_room_timeline
    from aic.utils.war_room_utils import project_metrics_to_topology_state
    _NEW_COMPONENTS = True
except ImportError:
    _NEW_COMPONENTS = False

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIC — War Room",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────
@st.cache_data
def load_trajectories(path: str):
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def get_demo_trajectory():
    """Generate minimal demo data if pkl files are missing."""
    from aic.training.config import TrainingConfig
    from aic.training.train import run_episode
    from aic.utils.seeding import make_episode_rng, get_adversary_cycle
    from aic.agents.db_agent import DBAgent
    from aic.agents.infra_agent import InfraAgent
    from aic.agents.app_agent import AppAgent
    from aic.agents.adversarial_agent import AdversarialAgent
    from aic.agents.orchestrator_agent import OrchestratorAgent

    config = TrainingConfig(num_episodes=1, use_llm_agents=False)
    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    cycle = get_adversary_cycle(make_episode_rng(0, 42))
    adv = AdversarialAgent(cycle, db)
    orch = OrchestratorAgent(adv, use_llm=False)
    result = run_episode(0, config, orch, db, infra, app)
    return {0: result}


ASSETS = Path(__file__).parent / "assets"
trained_data = load_trajectories(str(ASSETS / "trained_trajectories.pkl"))
untrained_data = load_trajectories(str(ASSETS / "untrained_trajectories.pkl"))

if trained_data is None:
    trained_data = get_demo_trajectory()
if untrained_data is None:
    untrained_data = get_demo_trajectory()

# ── Emerald Color Palette ───────────────────────────────────────────────
AGENT_COLORS = {
    "db_agent": "#34d399",
    "infra_agent": "#14b8a6",
    "app_agent": "#fbbf24",
    "adversarial_agent": "#fb7185",
}
AGENT_ICONS = {
    "db_agent": "🗄️",
    "infra_agent": "⚙️",
    "app_agent": "📱",
    "adversarial_agent": "🎭",
}

PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR = "#0c0f0a"
CARD_COLOR = "#161e14"
GRID_COLOR = "#1e2b1a"

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controls")

    mode = st.radio(
        "Agent Mode",
        ["Trained", "Untrained"],
        index=0,
        help="Toggle between trained (trust-updating) and untrained (frozen trust) agent.",
    )

    data = trained_data if mode == "Trained" else untrained_data
    available_episodes = sorted(data.keys())

    episode = st.select_slider(
        "Episode",
        options=available_episodes,
        value=available_episodes[0],
    )

    ep_data = data[episode]
    trajectory = ep_data["trajectory"]

    step = st.slider("Step", 0, SLA_STEPS - 1, 0, key="step_slider")

    st.divider()

    # Auto-play
    autoplay = st.checkbox("▶️ Auto-Play Episode", value=False)

    if st.button("⏮️ Reset to Step 0"):
        st.session_state.step_slider = 0
        st.rerun()

    st.divider()
    st.markdown("### 📊 Episode Summary")
    st.metric("Total Reward", f"{ep_data['total_reward']:+.1f}")
    st.metric("Final Health", f"{ep_data['final_health']:.3f}")
    st.metric("R2 (SLA Bonus)", f"{ep_data['r2_bonus']:.1f}")

# ── Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 1.2rem 0 0.8rem 0; animation: fadeInUp 0.6s ease-out both;">
        <div style="
            display: inline-block;
            background: rgba(52, 211, 153, 0.08);
            border: 1px solid rgba(52, 211, 153, 0.15);
            border-radius: 20px;
            padding: 4px 16px;
            margin-bottom: 12px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            color: #34d399;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        ">WAR ROOM · LIVE</div>
        <h1 style="
            background: linear-gradient(135deg, #059669, #10b981, #34d399, #6ee7b7);
            background-size: 300% 100%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.4rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0;
            line-height: 1.2;
        ">Adaptive Incident Choreographer</h1>
        <p style="color: #6b7a68; font-size: 0.88rem; margin-top: 8px; font-family: 'Inter', sans-serif; font-weight: 400; letter-spacing: 0.02em;">
            Autonomous Incident Command · Multi-Agent Trust Calibration
        </p>
        <div style="width: 60px; height: 2px; background: linear-gradient(90deg, #059669, #34d399); margin: 12px auto 0; border-radius: 1px;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main tab navigation ──────────────────────────────────────────────────
tab_mc, tab_lb, tab_jc, tab_biz, tab_pm = st.tabs([
    "🚨 Mission Control",
    "🏆 Leaderboard",
    "🎮 Judge Challenge",
    "💼 Business Impact",
    "📋 Postmortem",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Mission Control
# ════════════════════════════════════════════════════════════════════════
with tab_mc:
    step_data = trajectory[step]
    health = step_data["health"]
    health_color = "#34d399" if health > 0.5 else "#fbbf24" if health > 0.3 else "#fb7185"

    # Status bar
    cols_status = st.columns([2, 2, 2, 2, 2])
    with cols_status[0]:
        st.markdown(f"**Mode:** `{mode}`")
    with cols_status[1]:
        st.markdown(f"**Episode:** `{episode}`")
    with cols_status[2]:
        st.markdown(f"**Step:** `{step}/{SLA_STEPS}`")
    with cols_status[3]:
        st.markdown(f"**Health:** <span style='color:{health_color}'>`{health:.3f}`</span>", unsafe_allow_html=True)
    with cols_status[4]:
        sla_remaining = SLA_STEPS - step
        st.markdown(f"**SLA Remaining:** `{sla_remaining}` steps")

    st.divider()

    # ── Commander brief (if available) ─────────────────────────────────
    trace = step_data.get("trace", {})
    commander_brief = trace.get("commander_brief")
    if commander_brief and _NEW_COMPONENTS:
        commander_mode = trace.get("commander_mode", "fastest_recovery")
        mode_colors = {
            "fastest_recovery": "#fbbf24",
            "safest_recovery": "#34d399",
            "protect_data": "#a78bfa",
            "minimize_user_impact": "#14b8a6",
            "contain_compromise": "#fb7185",
        }
        cmd_color = mode_colors.get(commander_mode, "#6b7a68")
        st.markdown(
            f"""<div style="border-left: 3px solid {cmd_color}; background: rgba(52, 211, 153, 0.04);
                border-radius: 0 12px 12px 0; padding: 10px 14px; margin-bottom: 14px;
                font-size: 0.82rem; color: {cmd_color}; backdrop-filter: blur(8px);
                animation: slideInLeft 0.4s ease-out both;">
                {commander_brief}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Main layout ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4, 3, 3])

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN 1 — World State & Agent Cards
    # ═══════════════════════════════════════════════════════════════════
    with col1:
        st.markdown(f"### 🌐 World State — Step {step}")

        metrics = step_data["metrics"]

        # 4×3 grid of metrics
        metric_names = list(METRIC_TARGETS.keys())
        for row_start in range(0, len(metric_names), 3):
            row_metrics = metric_names[row_start:row_start + 3]
            mcols = st.columns(len(row_metrics))
            for i, m_name in enumerate(row_metrics):
                with mcols[i]:
                    current = metrics.get(m_name, 0.0)
                    target = METRIC_TARGETS[m_name]
                    if target == 0.0:
                        delta_val = -current
                        delta_str = f"{-current:.1f}"
                    else:
                        pct_off = (current - target) / target * 100
                        delta_val = -pct_off
                        delta_str = f"{-pct_off:.0f}%"

                    st.metric(
                        label=m_name.replace("_", " ").title(),
                        value=f"{current:.1f}",
                        delta=delta_str,
                        delta_color="normal" if abs(current - target) / max(target, 1e-6) < 0.1 else "inverse",
                    )

        st.divider()

        # Agent recommendation cards
        st.markdown("### 🤖 Agent Recommendations")

        trust_scores = step_data["trust_scores"]

        for agent_name in ALL_AGENTS:
            trust = trust_scores.get(agent_name, INITIAL_TRUST)
            icon = AGENT_ICONS.get(agent_name, "🔹")
            color = AGENT_COLORS.get(agent_name, "#6b7280")

            is_adversary = agent_name == AGENT_ADV
            border_style = f"border-left: 3px solid {color};"
            if is_adversary and trust < 0.4:
                border_style = f"border-left: 3px solid #fb7185; background: rgba(251, 113, 133, 0.04);"

            # Get recommendation from trace
            action_text = trace.get("action_taken", "N/A") if not is_adversary else "See trace for details"

            with st.container():
                st.markdown(
                    f"""<div style="{border_style} padding: 10px 14px; border-radius: 12px;
                        margin-bottom: 8px; background: rgba(22, 30, 20, 0.65);
                        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                        border: 1px solid rgba(52, 211, 153, 0.08);
                        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                        animation: fadeInUp 0.4s ease-out both;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <strong style="color: {color}; font-family: 'Inter', sans-serif;">{agent_name.replace('_', ' ').title()}</strong>
                        <span style="float: right; color: {'#fb7185' if trust < 0.4 else '#6b7a68'};
                            font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">
                            Trust: {trust:.2f}
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(trust, text=None)

                if is_adversary and trust < 0.4:
                    st.warning("⚠️ Low trust — recommendations may be adversarial")

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN 2 — Trust Evolution & Explanation Trace
    # ═══════════════════════════════════════════════════════════════════
    with col2:
        st.markdown("### 📈 Trust Evolution")

        trust_evo = ep_data["trust_evolution"]

        fig_trust = go.Figure()
        for agent_name in ALL_AGENTS:
            steps_list = [te["step"] for te in trust_evo]
            trust_vals = [te.get(agent_name, INITIAL_TRUST) for te in trust_evo]
            color = AGENT_COLORS.get(agent_name, "#6b7280")
            dash = "dot" if agent_name == AGENT_ADV else "solid"
            width = 3 if agent_name == AGENT_ADV else 2

            fig_trust.add_trace(go.Scatter(
                x=steps_list, y=trust_vals,
                name=agent_name.replace("_", " ").title(),
                line=dict(color=color, width=width, dash=dash),
                mode="lines+markers",
                marker=dict(size=4),
            ))

        fig_trust.add_vline(
            x=step, line_width=2, line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=f"Step {step}",
            annotation_position="top",
        )

        fig_trust.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            height=320,
            margin=dict(l=20, r=20, t=30, b=30),
            yaxis=dict(range=[0, 1.05], title="Trust Score", gridcolor=GRID_COLOR),
            xaxis=dict(title="Step", gridcolor=GRID_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            font=dict(family="Inter", color="#9ca89a"),
        )
        st.plotly_chart(fig_trust, use_container_width=True)

        st.divider()

        # Explanation trace viewer
        st.markdown(f"### 📋 Explanation Trace — Step {step}")

        if trace:
            with st.expander("📝 Action & Reasoning", expanded=True):
                st.markdown(f"**Action:** {trace.get('action_taken', 'N/A')}")
                st.markdown(f"**Reasoning:** {trace.get('reasoning', 'N/A')}")

            c1, c2 = st.columns(2)
            with c1:
                override = trace.get("override_applied", False)
                if override:
                    st.error("🔄 Override Applied")
                    reason = trace.get("override_reason", "")
                    if reason:
                        st.caption(reason)
                else:
                    st.success("✅ No Override")

            with c2:
                drift = trace.get("schema_drift_detected", False)
                if drift:
                    field = trace.get("schema_drift_field", "unknown")
                    st.warning(f"⚠️ Schema Drift: `{field}`")
                elif step_data.get("drift_active", False):
                    st.info("🔍 Drift active (undetected)")
                else:
                    st.success("✅ No Drift")

            with st.expander("🔮 Predicted vs Actual Impact"):
                predicted = trace.get("predicted_2step_impact", {})
                if predicted:
                    for metric, pred_val in predicted.items():
                        actual_val = 0.0
                        if step + 2 < len(trajectory):
                            future = trajectory[step + 2]["metrics"]
                            current_m = metrics.get(metric, 0.0)
                            actual_val = future.get(metric, current_m) - current_m

                        acc_color = "#34d399" if abs(pred_val - actual_val) < abs(pred_val) * 0.5 else "#fb7185"
                        st.markdown(
                            f"**{metric}**: predicted `{pred_val:+.1f}` → actual `{actual_val:+.1f}` "
                            f"<span style='color:{acc_color}'>{'✅' if abs(pred_val - actual_val) < abs(pred_val) * 0.5 else '❌'}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No predictions for this step.")

            # Agent Debate panel (inline in trace column)
            if _NEW_COMPONENTS and (trace.get("debate_transcript") or trace.get("commander_brief")):
                with st.expander("🗣️ Agent Debate", expanded=False):
                    render_debate_panel(trace)

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN 3 — Reward Curve & Simulator
    # ═══════════════════════════════════════════════════════════════════
    with col3:
        st.markdown("### 📊 Reward Curve")

        fig_reward = go.Figure()

        trained_eps = sorted(trained_data.keys())
        trained_rewards = [trained_data[ep]["total_reward"] for ep in trained_eps]
        fig_reward.add_trace(go.Scatter(
            x=list(trained_eps), y=trained_rewards,
            name="Trained (Trust Update)",
            line=dict(color="#34d399", width=3),
            mode="lines+markers", marker=dict(size=8, symbol="circle"),
        ))

        untrained_eps = sorted(untrained_data.keys())
        untrained_rewards = [untrained_data[ep]["total_reward"] for ep in untrained_eps]
        fig_reward.add_trace(go.Scatter(
            x=list(untrained_eps), y=untrained_rewards,
            name="Untrained (Frozen Trust)",
            line=dict(color="#fb7185", width=3, dash="dot"),
            mode="lines+markers", marker=dict(size=8, symbol="diamond"),
        ))

        fig_reward.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            height=300,
            margin=dict(l=20, r=20, t=30, b=30),
            yaxis=dict(title="Total Reward", gridcolor=GRID_COLOR),
            xaxis=dict(title="Episode", gridcolor=GRID_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            font=dict(family="Inter", color="#9ca89a"),
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        st.divider()

        # Step reward breakdown
        st.markdown(f"### 🎯 Step {step} Reward Breakdown")
        reward = step_data.get("reward", {})
        r1 = reward.get("r1", 0.0)
        r3 = reward.get("r3", 0.0)
        r4 = reward.get("r4", 0.0)
        total = reward.get("total", 0.0)

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("R1 (Health)", f"{r1:+.1f}", delta_color="off")
        with rc2:
            st.metric("R3 (Trust)", f"{r3:+.1f}",
                       delta="Good" if r3 > 0 else "Bad",
                       delta_color="normal" if r3 > 0 else "inverse")
        with rc3:
            st.metric("R4 (Explain)", f"{r4:+.1f}", delta_color="off")
        with rc4:
            st.metric("Total", f"{total:+.1f}", delta_color="off")

        st.divider()

        # Interactive reward simulator
        st.markdown("### 🧮 Reward Simulator")
        st.caption("Adjust parameters to understand the reward math.")

        sim_health = st.slider(
            "Health Recovery (R1 factor)",
            min_value=-1.0, max_value=0.0, value=-0.5, step=0.05, key="sim_health",
        )

        sim_trust_case = st.selectbox(
            "Trust Calibration (R3 case)",
            ["Correct Override (+15)", "Correct Trust (+5)",
             "Unnecessary Override (-10)", "Blind Trust (-20)"],
            index=0, key="sim_trust",
        )

        sim_accuracy = st.slider(
            "Prediction Accuracy (R4 factor)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="sim_accuracy",
        )

        sim_r1 = sim_health * 5
        trust_map = {
            "Correct Override (+15)": R3_CORRECT_OVERRIDE,
            "Correct Trust (+5)": R3_CORRECT_TRUST,
            "Unnecessary Override (-10)": R3_WRONG_OVERRIDE,
            "Blind Trust (-20)": R3_WRONG_TRUST,
        }
        sim_r3 = trust_map[sim_trust_case]
        sim_r4 = R4_MIN_PER_STEP + sim_accuracy * (R4_MAX_PER_STEP - R4_MIN_PER_STEP)
        sim_total = sim_r1 + sim_r3 + sim_r4
        total_color = "#34d399" if sim_total > 0 else "#fb7185"

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Sim R1", f"{sim_r1:+.1f}")
        with sc2:
            st.metric("Sim R3", f"{sim_r3:+.1f}")
        with sc3:
            st.metric("Sim R4", f"{sim_r4:+.1f}")

        st.markdown(
            f"<div style='text-align:center; padding:14px; background:rgba(22,30,20,0.65); "
            f"backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); "
            f"border:1px solid rgba(52,211,153,0.1); "
            f"border-radius:14px; margin-top:8px;'>"
            f"<span style='font-size:0.78rem; color:#6b7a68; font-family:JetBrains Mono,monospace; text-transform:uppercase; letter-spacing:0.08em;'>Simulated Total</span><br>"
            f"<span style='font-size:2rem; font-weight:700; color:{total_color}; font-family:JetBrains Mono,monospace;'>"
            f"{sim_total:+.1f}</span></div>",
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2)
        with p1:
            if st.button("📉 Untrained Preset"):
                st.session_state.sim_health = -0.8
                st.session_state.sim_trust = "Blind Trust (-20)"
                st.session_state.sim_accuracy = 0.2
                st.rerun()
        with p2:
            if st.button("📈 Trained Preset"):
                st.session_state.sim_health = -0.1
                st.session_state.sim_trust = "Correct Override (+15)"
                st.session_state.sim_accuracy = 0.8
                st.rerun()

    # Debate summary (full-width, inside Mission Control tab)
    if _NEW_COMPONENTS:
        with st.expander("🗣️ Episode Debate Summary", expanded=False):
            render_debate_summary_panel(trajectory)

    # Topology + War Room Timeline (full-width, inside Mission Control tab)
    if _NEW_COMPONENTS:
        topo_col, timeline_col = st.columns(2)
        with topo_col:
            # Service topology DAG colored by health
            rca_node = None
            rca_hyp = trace.get("root_cause_hypothesis")
            if rca_hyp and isinstance(rca_hyp, dict):
                scenario_name_lc = rca_hyp.get("scenario_name", "").lower()
                if "cache" in scenario_name_lc:
                    rca_node = "cache"
                elif "db" in scenario_name_lc or "schema" in scenario_name_lc:
                    rca_node = "db"
                elif "queue" in scenario_name_lc:
                    rca_node = "queue"
                elif "credential" in scenario_name_lc or "security" in scenario_name_lc:
                    rca_node = "app"
                elif "regional" in scenario_name_lc or "network" in scenario_name_lc:
                    rca_node = "gateway"
            topo_state = project_metrics_to_topology_state(metrics, rca_node)
            fig_topo = render_topology_map(topo_state, root_cause_node=rca_node, height=350)
            st.plotly_chart(fig_topo, use_container_width=True)

        with timeline_col:
            # War room incident timeline
            events = extract_timeline_events(trajectory[:step + 1])
            fig_timeline = render_war_room_timeline(events, height=350)
            st.plotly_chart(fig_timeline, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Leaderboard
# ════════════════════════════════════════════════════════════════════════
with tab_lb:
    if _NEW_COMPONENTS:
        render_leaderboard_panel(
            arena_path=str(Path(__file__).parent.parent / "logs" / "arena_results.json")
        )
    else:
        st.warning("Leaderboard components not loaded. Check imports.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Judge Challenge
# ════════════════════════════════════════════════════════════════════════
with tab_jc:
    if _NEW_COMPONENTS:
        render_judge_challenge_panel()
    else:
        st.warning("Judge Challenge components not loaded. Check imports.")

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — Business Impact
# ════════════════════════════════════════════════════════════════════════
with tab_biz:
    if _NEW_COMPONENTS:
        ep_scenario = ep_data.get("scenario_name", "")
        render_business_impact_panel(
            trajectory=trajectory, step=step, scenario_name=ep_scenario,
        )
    else:
        st.warning("Business Impact components not loaded. Check imports.")

# ════════════════════════════════════════════════════════════════════════
# TAB 5 — Postmortem
# ════════════════════════════════════════════════════════════════════════
with tab_pm:
    if _NEW_COMPONENTS:
        ep_scenario = ep_data.get("scenario_name", "Unknown Scenario")
        ep_mttr = ep_data.get("mttr", SLA_STEPS)
        render_postmortem_panel(
            trajectory=trajectory,
            final_health=ep_data["final_health"],
            mttr_steps=ep_mttr,
            scenario_name=ep_scenario,
            total_reward=ep_data["total_reward"],
        )
    else:
        st.warning("Postmortem components not loaded. Check imports.")

# ── Auto-play logic ─────────────────────────────────────────────────────
if autoplay:
    current_step = st.session_state.get("step_slider", 0)
    if current_step < SLA_STEPS - 1:
        time.sleep(1.5)
        st.session_state.step_slider = current_step + 1
        st.rerun()
    else:
        st.toast("✅ Episode complete!", icon="🎉")

# ── Footer ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; padding:12px;'>"
    "<div style='width:80px; height:1px; background:linear-gradient(90deg, transparent, rgba(52,211,153,0.3), transparent); margin:0 auto 12px;'></div>"
    "<span style='color:#3d4a3b; font-size:0.75rem; font-family:Inter,sans-serif; letter-spacing:0.03em;'>"
    "Adaptive Incident Choreographer · Autonomous Command Arena<br>"
    "Gymnasium · Pydantic · Plotly · Streamlit"
    "</span></div>",
    unsafe_allow_html=True,
)
