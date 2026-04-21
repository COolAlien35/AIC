#!/usr/bin/env python3
"""
AIC — Hugging Face Space Demo (Gradio)

Interactive step-through of the Adaptive Incident Choreographer environment.
Shows observation, metrics, reward, health, and done status at each step.

Usage:
    python app.py
    # Opens Gradio UI at http://localhost:7860
"""
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio>=4.0.0")

from aic.env.aic_environment import AICEnvironment
from aic.env.reward_engine import compute_r1
from aic.utils.constants import METRIC_TARGETS, METRIC_FAULT_INIT, SLA_STEPS

# ── Global state ────────────────────────────────────────────────────────
_env = None
_obs = None
_step_history = []


def create_env(episode_id: int, seed: int, fault_mode: str):
    """Create and reset a new AIC environment."""
    global _env, _obs, _step_history
    _env = AICEnvironment(
        episode_id=int(episode_id),
        base_seed=int(seed),
        fault_mode=fault_mode,
    )
    _obs = _env.reset()
    _step_history = []

    metrics = _env.world_state.snapshot()
    health = _env.world_state.get_health_score()

    status = (
        f"✅ Environment created and reset.\n"
        f"Episode: {episode_id} | Seed: {seed} | Fault: {fault_mode}\n"
        f"SLA Steps: {SLA_STEPS} | Health: {health:.3f}"
    )

    metrics_table = _format_metrics(metrics)
    obs_json = json.dumps(_obs, indent=2, default=str)

    return status, metrics_table, obs_json, "", "[]"


def step_env(action_text: str):
    """Step the environment with a natural-language action."""
    global _env, _obs, _step_history

    if _env is None:
        return "⚠️ Create an environment first!", "", "", "", "[]"

    if _env.done:
        return "🏁 Episode is done. Reset to continue.", "", "", "", json.dumps(_step_history, indent=2)

    if not action_text.strip():
        action_text = "No action taken — observing system state."

    obs, reward, done, info = _env.step(action_text)
    _obs = obs

    metrics = _env.world_state.snapshot()
    health = info.get("health", 0.0)
    r1 = compute_r1(metrics)

    step_record = {
        "step": info["step"],
        "action": action_text[:100],
        "reward": round(r1, 3),
        "health": round(health, 3),
        "done": done,
        "sla_ok": info.get("is_within_sla", False),
    }
    _step_history.append(step_record)

    done_marker = "🏁 EPISODE COMPLETE" if done else ""
    sla_status = "✅ SLA OK" if info.get("is_within_sla") else "❌ SLA BREACH"

    status = (
        f"Step {info['step']}/{SLA_STEPS} | "
        f"Health: {health:.3f} | "
        f"R1 Reward: {r1:+.3f} | "
        f"{sla_status} {done_marker}"
    )

    metrics_table = _format_metrics(metrics)
    obs_json = json.dumps(obs, indent=2, default=str)
    history_json = json.dumps(_step_history, indent=2)

    return status, metrics_table, obs_json, "", history_json


def auto_step():
    """Step with a default no-op action."""
    return step_env("Observe current system state and assess severity.")


def _format_metrics(metrics: dict) -> str:
    """Format metrics into a readable table string."""
    lines = [f"{'Metric':<25} {'Current':>10} {'Target':>10} {'Status':>10}"]
    lines.append("-" * 58)
    for name in sorted(metrics.keys()):
        current = metrics[name]
        target = METRIC_TARGETS.get(name, 0.0)
        if target == 0.0:
            status = "✅" if current <= 0.5 else "❌"
        else:
            pct_off = abs(current - target) / target * 100
            status = "✅" if pct_off <= 10 else ("⚠️" if pct_off <= 30 else "❌")
        lines.append(f"{name:<25} {current:>10.1f} {target:>10.1f} {status:>10}")
    return "\n".join(lines)


# ── Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="AIC — Adaptive Incident Choreographer",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    ),
) as demo:
    gr.Markdown(
        """
        # 🚨 Adaptive Incident Choreographer
        ### Multi-Agent Trust Calibration Under Adversarial Conditions

        Step through a simulated production incident.
        Observe how metrics evolve, health degrades, and rewards accumulate.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Environment Setup")
            episode_input = gr.Number(label="Episode ID", value=0, precision=0)
            seed_input = gr.Number(label="Random Seed", value=42, precision=0)
            fault_input = gr.Dropdown(
                label="Fault Mode",
                choices=[
                    "cascading_failure",
                    "memory_leak",
                    "db_connection_saturation",
                    "network_storm",
                ],
                value="cascading_failure",
            )
            create_btn = gr.Button("🔄 Create & Reset Environment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### ▶️ Step Control")
            action_input = gr.Textbox(
                label="Action (natural language)",
                placeholder="e.g. Drain connection pool to 40% and enable queuing...",
                lines=2,
            )
            with gr.Row():
                step_btn = gr.Button("⏩ Step with Action", variant="primary")
                auto_btn = gr.Button("👁️ Observe (No-Op Step)")

    gr.Markdown("---")
    status_output = gr.Textbox(label="📊 Status", lines=2, interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌐 Metrics")
            metrics_output = gr.Textbox(
                label="Current Metrics vs Targets",
                lines=16,
                interactive=False,
                show_copy_button=True,
            )
        with gr.Column():
            gr.Markdown("### 📋 Observation")
            obs_output = gr.Textbox(
                label="Orchestrator Observation (JSON)",
                lines=16,
                interactive=False,
                show_copy_button=True,
            )

    with gr.Accordion("📜 Step History (JSON)", open=False):
        history_output = gr.Textbox(
            label="All Steps",
            lines=10,
            interactive=False,
            show_copy_button=True,
        )

    hidden = gr.Textbox(visible=False)

    # Wire up buttons
    create_btn.click(
        fn=create_env,
        inputs=[episode_input, seed_input, fault_input],
        outputs=[status_output, metrics_output, obs_output, hidden, history_output],
    )
    step_btn.click(
        fn=step_env,
        inputs=[action_input],
        outputs=[status_output, metrics_output, obs_output, hidden, history_output],
    )
    auto_btn.click(
        fn=auto_step,
        inputs=[],
        outputs=[status_output, metrics_output, obs_output, hidden, history_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
