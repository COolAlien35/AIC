# dashboard/components/judge_challenge_mode.py
"""
Judge Challenge Mode — interactive scenario runner for demos.

Judges pick:
  - Scenario
  - Adversary strength (0–1)
  - Business priority
  - Safety strictness
  - Policies to compare

Then hit "▶ Run Challenge" to see a live side-by-side comparison.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from aic.utils.constants import SLA_STEPS, METRIC_TARGETS, ALL_AGENTS, AGENT_ADV
from aic.utils.seeding import make_episode_rng, get_adversary_cycle
from aic.env.world_state import WorldState
from aic.env.scenario_registry import ScenarioEngine, SCENARIO_REGISTRY
from aic.env.reward_engine import RewardEngine
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.app_agent import AppAgent
from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.evals.benchmark_suite import (
    HighestConfidencePolicy, MajorityVotePolicy, NoTrustOrchestratorPolicy,
)

BG = "#0c0f0a"
CARD = "#161e14"
GRID = "#1e2b1a"
TEMPLATE = "plotly_dark"

_POLICY_COLORS = {
    "AIC (Full Stack)": "#34d399",
    "AIC (No Trust)": "#14b8a6",
    "Highest Confidence": "#fbbf24",
    "Majority Vote": "#6b7a68",
    "No Trust Orch.": "#fb7185",
}

_SCENARIO_NAMES = {s.scenario_id: s.name for s in SCENARIO_REGISTRY.values()}
_SCENARIO_DESCRIPTIONS = {s.scenario_id: s.description for s in SCENARIO_REGISTRY.values()}


def _run_challenge_episode(
    scenario_id: int,
    policy_name: str,
    adversary_strength: float,
    seed: int = 77,
) -> dict:
    """Run a single episode and return trajectory + summary."""
    rng = make_episode_rng(seed)
    engine = ScenarioEngine(scenario_id)
    ws = WorldState(rng)

    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    cycle = get_adversary_cycle(make_episode_rng(seed))
    adv = AdversarialAgent(cycle, correct_recommendation_provider=db)

    # Adversary strength hack: override ADV confidence
    adv._strength = adversary_strength  # informal; affects confidence weighting

    reward_engine = RewardEngine()

    health_curve = []
    metric_snapshots = []
    total_reward = 0.0
    prev_metrics = ws.snapshot()
    mttr = SLA_STEPS
    adversary_followed = 0
    unsafe_count = 0

    # Build policy
    if policy_name == "AIC (Full Stack)":
        orch = OrchestratorAgent(adv, use_llm=False)
        orch.mode = "trained"
        use_orch = True
    elif policy_name == "AIC (No Trust)":
        orch = OrchestratorAgent(adv, use_llm=False)
        orch.mode = "untrained"
        use_orch = True
    else:
        use_orch = False
        policy_map = {
            "Highest Confidence": HighestConfidencePolicy(),
            "Majority Vote": MajorityVotePolicy(),
            "No Trust Orch.": NoTrustOrchestratorPolicy(),
        }
        policy = policy_map.get(policy_name, HighestConfidencePolicy())

    for step in range(SLA_STEPS):
        try:
            faults = engine.get_contributions(step)
            db_obs = ws.get_db_observation()
            infra_obs = ws.get_infra_observation()
            app_obs = ws.get_app_observation()

            db_rec = db.recommend(db_obs, step)
            infra_rec = infra.recommend(infra_obs, step)
            app_rec = app.recommend(app_obs, step)
            adv_rec = adv.recommend({**db_obs, **infra_obs, **app_obs}, step)
            all_recs = [db_rec, infra_rec, app_rec, adv_rec]

            current = ws.snapshot()

            if use_orch:
                action, override = orch.decide(
                    step=step,
                    sla_remaining=SLA_STEPS - step,
                    sub_agent_recommendations=all_recs,
                    alert_summary=f"challenge:{_SCENARIO_NAMES[scenario_id]}",
                    prev_metrics=prev_metrics,
                    current_metrics=current,
                )
                selected_agent = orch._followed_agent
                action_deltas = action.action_deltas
                risk = action.explanation_trace.verifier_report.get("risk_score", 0) if action.explanation_trace.verifier_report else 0
            else:
                selected = policy.select(all_recs)
                selected_agent = selected.agent_name
                action_deltas = {m: -10.0 for m in selected.target_metrics}
                risk = selected.risk_score

            if selected_agent == AGENT_ADV:
                adversary_followed += 1
            if risk > 0.8:
                unsafe_count += 1

            ws.step(action_deltas, faults)

            snap = ws.snapshot()
            health = ws.get_health_score()
            health_curve.append(health)
            metric_snapshots.append(snap)

            rr = reward_engine.compute_step_reward(
                step=step, metrics=snap, prev_metrics=prev_metrics,
                override_applied=False,
                adversary_was_correct=adv.was_correct_at_step(step),
                predicted_2step_impact={m: -5.0 for m in action_deltas.keys()},
                reasoning="challenge",
                lock_penalty=0.0,
            )
            total_reward += rr["total"]
            prev_metrics = snap

            if health > 0.5 and mttr == SLA_STEPS:
                mttr = step + 1

        except Exception:
            health_curve.append(health_curve[-1] if health_curve else 0.0)
            metric_snapshots.append(prev_metrics)

    final_health = ws.get_health_score()
    return {
        "policy": policy_name,
        "scenario": _SCENARIO_NAMES[scenario_id],
        "health_curve": health_curve,
        "metric_snapshots": metric_snapshots,
        "total_reward": total_reward,
        "mttr": mttr,
        "final_health": final_health,
        "sla_met": final_health > 0.5,
        "adversary_followed": adversary_followed,
        "unsafe_count": unsafe_count,
    }


def render_judge_challenge_panel() -> None:
    """Render the full Judge Challenge Mode panel."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(20,184,166,0.04));
            border: 1px solid rgba(52,211,153,0.2);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            animation: fadeInUp 0.5s ease-out both;
        ">
            <h2 style="margin: 0; background: linear-gradient(135deg, #059669, #34d399);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-family: 'Outfit', sans-serif;">
                🎮 Judge Challenge Mode
            </h2>
            <p style="color: #6b7a68; margin: 8px 0 0 0; font-family: 'Inter', sans-serif;">
                Pick any scenario, crank up the adversary, and watch AIC race against baselines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Configuration ─────────────────────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        scenario_id = st.selectbox(
            "📋 Scenario",
            options=sorted(SCENARIO_REGISTRY.keys()),
            format_func=lambda sid: f"{SCENARIO_REGISTRY[sid].name} ({SCENARIO_REGISTRY[sid].severity})",
            key="jc_scenario",
        )
        st.caption(f"*{_SCENARIO_DESCRIPTIONS[scenario_id][:150]}…*")

        adversary_strength = st.slider(
            "🎭 Adversary Strength",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1,
            help="Higher = adversary gives more misleading recommendations",
            key="jc_adv_strength",
        )

    with col_cfg2:
        policies_to_run = st.multiselect(
            "🤖 Policies to Compare",
            options=["AIC (Full Stack)", "AIC (No Trust)", "Highest Confidence", "Majority Vote", "No Trust Orch."],
            default=["AIC (Full Stack)", "Highest Confidence", "No Trust Orch."],
            key="jc_policies",
        )

        seed = st.number_input(
            "🎲 Seed", min_value=0, max_value=9999, value=42, step=1,
            help="RNG seed for reproducibility",
            key="jc_seed",
        )

    # ── Run button ────────────────────────────────────────────────────────
    st.divider()

    if not policies_to_run:
        st.warning("Select at least one policy to compare.")
        return

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "▶ Run Challenge",
            type="primary",
            use_container_width=True,
            key="jc_run",
        )
    with col_info:
        if run_clicked:
            st.info(f"Running {len(policies_to_run)} policies × scenario '{_SCENARIO_NAMES[scenario_id]}'…")

    if not run_clicked:
        st.caption("👆 Configure above then click **Run Challenge** to race policies live.")
        return

    # ── Run episodes ──────────────────────────────────────────────────────
    results = []
    progress = st.progress(0.0, text="Running challenge…")

    for i, policy_name in enumerate(policies_to_run):
        progress.progress((i + 0.5) / len(policies_to_run), text=f"Running {policy_name}…")
        result = _run_challenge_episode(
            scenario_id=scenario_id,
            policy_name=policy_name,
            adversary_strength=adversary_strength,
            seed=int(seed),
        )
        results.append(result)
        progress.progress((i + 1) / len(policies_to_run), text=f"✓ {policy_name}")

    progress.empty()

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown(f"### 🏁 Results: {_SCENARIO_NAMES[scenario_id]}")

    # Winner callout
    winner = max(results, key=lambda r: (r["sla_met"], r["final_health"], -r["mttr"]))
    st.success(
        f"🏆 **Winner: {winner['policy']}** — "
        f"Health={winner['final_health']:.1%}, "
        f"MTTR={winner['mttr']} steps, "
        f"SLA={'✅' if winner['sla_met'] else '❌'}"
    )

    # Summary cards
    summary_cols = st.columns(len(results))
    for col, r in zip(summary_cols, results):
        with col:
            color = _POLICY_COLORS.get(r["policy"], "#64748b")
            health_color = "#34d399" if r["sla_met"] else "#fb7185"
            st.markdown(
                f"""
                <div style="
                    border: 1px solid {color};
                    border-radius: 14px;
                    padding: 14px;
                    text-align: center;
                    background: rgba(22, 30, 20, 0.65);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                    animation: fadeInUp 0.4s ease-out both;
                ">
                    <div style="color:{color};font-weight:600;font-size:0.85rem;font-family:'Inter',sans-serif;">{r['policy']}</div>
                    <div style="color:{health_color};font-size:1.8rem;font-weight:700;margin:6px 0;font-family:'JetBrains Mono',monospace;">
                        {r['final_health']:.0%}
                    </div>
                    <div style="color:#6b7a68;font-size:0.72rem;font-family:'JetBrains Mono',monospace;">
                        MTTR: {r['mttr']} | SLA: {'✅' if r['sla_met'] else '❌'}<br>
                        Adv followed: {r['adversary_followed']} | Unsafe: {r['unsafe_count']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Health curve comparison
    st.markdown("### 📈 Health Curve Comparison")
    fig = go.Figure()
    for r in results:
        color = _POLICY_COLORS.get(r["policy"], "#64748b")
        fig.add_trace(go.Scatter(
            x=list(range(len(r["health_curve"]))),
            y=r["health_curve"],
            name=r["policy"],
            line=dict(color=color, width=2),
            mode="lines",
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="#fbbf24",
                  annotation_text="SLA threshold", annotation_position="right")

    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=320,
        margin=dict(l=20, r=20, t=20, b=30),
        yaxis=dict(title="System Health", range=[0, 1.05], gridcolor=GRID),
        xaxis=dict(title="Step", gridcolor=GRID),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        font=dict(family="Inter", color="#9ca89a"),
    )
    st.plotly_chart(fig, use_container_width=True)
