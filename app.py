#!/usr/bin/env python3
"""
AIC — Hugging Face Space Demo (Gradio)

Interactive step-through of the Adaptive Incident Choreographer environment.
Shows observation, metrics, reward, health, and done status at each step.
Includes trained model toggle for live GRPO model comparison.

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

# ── Trained Model Support ───────────────────────────────────────────────
TRAINED_MODEL = None
TRAINED_TOKENIZER = None
_MODEL_STATUS = "⚠️ Using baseline (no trained model loaded)"


def load_trained_model():
    global TRAINED_MODEL, TRAINED_TOKENIZER, _MODEL_STATUS
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        export_path = "exports/aic-orchestrator-trained"
        if not Path(export_path).exists():
            # Try GRPO checkpoint
            export_path = "checkpoints/grpo"
        if not Path(export_path).exists():
            _MODEL_STATUS = "⚠️ No trained model found — using baseline"
            return False

        TRAINED_MODEL = AutoModelForCausalLM.from_pretrained(
            export_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        TRAINED_TOKENIZER = AutoTokenizer.from_pretrained(export_path)
        TRAINED_MODEL.eval()
        _MODEL_STATUS = f"✅ Trained model loaded from {export_path}"
        return True
    except Exception as e:
        _MODEL_STATUS = f"⚠️ Could not load trained model: {e}"
        return False


def get_model_decision(obs: dict, use_trained: bool = False) -> dict:
    """Get decision from trained model or fallback to baseline."""
    if use_trained and TRAINED_MODEL is not None:
        try:
            import torch
            from aic.training.prompting import build_orchestrator_prompt

            prompt = build_orchestrator_prompt(obs)
            inputs = TRAINED_TOKENIZER(prompt, return_tensors="pt", max_length=1024, truncation=True)
            with torch.no_grad():
                out = TRAINED_MODEL.generate(
                    **inputs.to(TRAINED_MODEL.device),
                    max_new_tokens=256, temperature=0.3, do_sample=True,
                    pad_token_id=TRAINED_TOKENIZER.eos_token_id
                )
            text = TRAINED_TOKENIZER.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            j = json.loads(text[text.find("{"):text.rfind("}")+1])
            return j
        except Exception:
            pass

    # Fallback: pick highest-confidence non-adversarial candidate
    candidates = obs.get("candidate_recommendations", [])
    safe = [c for c in candidates if c.get("agent_name") != "adversarial_agent"]
    if safe:
        best = max(safe, key=lambda c: c.get("confidence", 0))
        return {
            "selected_recommendation_id": best.get("recommendation_id", 0),
            "override_adversary": len(safe) < len(candidates),
            "reasoning": f"Baseline: selected {best.get('agent_name', 'unknown')} with confidence {best.get('confidence', 0):.2f}",
        }
    return {"selected_recommendation_id": 0, "override_adversary": False, "reasoning": "Fallback"}


# ── Global state ────────────────────────────────────────────────────────
_env = None
_obs = None
_step_history = []


def create_env(episode_id: int, seed: int, fault_mode: str):
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
    return status, _format_metrics(metrics), json.dumps(_obs, indent=2, default=str), "", "[]"


def step_env(action_text: str, use_trained: bool = False):
    global _env, _obs, _step_history
    if _env is None:
        return "⚠️ Create an environment first!", "", "", "", "[]"
    if _env.done:
        return "🏁 Episode is done. Reset to continue.", "", "", "", json.dumps(_step_history, indent=2)

    if not action_text.strip():
        if use_trained and TRAINED_MODEL is not None:
            decision = get_model_decision(_obs, use_trained=True)
            obs, reward, done, info = _env.step(decision)
            action_text = f"[TRAINED MODEL] {decision.get('reasoning', '')[:80]}"
        else:
            action_text = "No action taken — observing system state."
            obs, reward, done, info = _env.step(action_text)
    else:
        obs, reward, done, info = _env.step(action_text)

    _obs = obs
    metrics = _env.world_state.snapshot()
    health = info.get("health", 0.0)
    r1 = compute_r1(metrics)
    step_record = {
        "step": info["step"], "action": action_text[:100],
        "reward": round(r1, 3), "health": round(health, 3),
        "done": done, "sla_ok": info.get("is_within_sla", False),
    }
    _step_history.append(step_record)
    done_marker = "🏁 EPISODE COMPLETE" if done else ""
    sla_status = "✅ SLA OK" if info.get("is_within_sla") else "❌ SLA BREACH"
    status = f"Step {info['step']}/{SLA_STEPS} | Health: {health:.3f} | R1: {r1:+.3f} | {sla_status} {done_marker}"
    return status, _format_metrics(metrics), json.dumps(obs, indent=2, default=str), "", json.dumps(_step_history, indent=2)


def auto_step(use_trained: bool = False):
    return step_env("", use_trained)


def _format_metrics(metrics: dict) -> str:
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


# Try loading trained model at startup
load_trained_model()

# ── Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="AIC — Adaptive Incident Choreographer",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
) as demo:
    gr.Markdown("""
    # 🚨 Adaptive Incident Choreographer
    ### Multi-Agent Trust Calibration Under Adversarial Conditions
    Step through a simulated production incident. Toggle the trained GRPO model to compare.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Environment Setup")
            episode_input = gr.Number(label="Episode ID", value=0, precision=0)
            seed_input = gr.Number(label="Random Seed", value=42, precision=0)
            fault_input = gr.Dropdown(
                label="Fault Mode",
                choices=[
                    "cascading_failure", "memory_leak",
                    "db_connection_saturation", "network_storm",
                ],
                value="cascading_failure",
            )
            create_btn = gr.Button("🔄 Create & Reset Environment", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### ▶️ Step Control")
            action_input = gr.Textbox(
                label="Action (natural language or leave empty for model)",
                placeholder="e.g. Drain connection pool to 40%...",
                lines=2,
            )
            with gr.Row():
                use_trained_toggle = gr.Checkbox(label="🧠 Use Trained GRPO Model", value=False)
                model_status = gr.Textbox(value=_MODEL_STATUS, label="Model Status", interactive=False)
            with gr.Row():
                step_btn = gr.Button("⏩ Step with Action", variant="primary")
                auto_btn = gr.Button("👁️ Auto Step (Model/Observe)")

    gr.Markdown("---")
    status_output = gr.Textbox(label="📊 Status", lines=2, interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌐 Metrics")
            metrics_output = gr.Textbox(label="Metrics vs Targets", lines=16, interactive=False, show_copy_button=True)
        with gr.Column():
            gr.Markdown("### 📋 Observation")
            obs_output = gr.Textbox(label="Observation (JSON)", lines=16, interactive=False, show_copy_button=True)

    with gr.Accordion("📜 Step History (JSON)", open=False):
        history_output = gr.Textbox(label="All Steps", lines=10, interactive=False, show_copy_button=True)

    hidden = gr.Textbox(visible=False)

    create_btn.click(fn=create_env, inputs=[episode_input, seed_input, fault_input],
                     outputs=[status_output, metrics_output, obs_output, hidden, history_output])
    step_btn.click(fn=step_env, inputs=[action_input, use_trained_toggle],
                   outputs=[status_output, metrics_output, obs_output, hidden, history_output])
    auto_btn.click(fn=auto_step, inputs=[use_trained_toggle],
                   outputs=[status_output, metrics_output, obs_output, hidden, history_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
