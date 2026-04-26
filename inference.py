#!/usr/bin/env python3
"""OpenEnv baseline inference script.

Emits structured stdout records with [START], [STEP], and [END] prefixes.
The policy uses the OpenAI client for action selection and falls back to a
deterministic safe action if the remote model call fails.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI

from aic.env.aic_environment import AICEnvironment
from aic.evals.openenv_tasks import TASKS, grade_task
from aic.schemas.actions import OrchestratorDecision


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or "missing"
    base_url = os.environ.get("API_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _prompt(obs: dict[str, Any]) -> str:
    candidates = obs.get("candidate_recommendations", [])
    return (
        "You are controlling a production incident response environment. "
        "Return strict JSON only with keys selected_recommendation_id, "
        "override_adversary, reasoning, predicted_2step_impact, "
        "schema_drift_detected, schema_drift_field.\n\n"
        f"Step: {obs.get('step')}\n"
        f"SLA remaining: {obs.get('sla_remaining_steps')}\n"
        f"Metrics: {json.dumps(obs.get('current_metrics', {}), sort_keys=True)}\n"
        f"Candidates: {json.dumps(candidates, sort_keys=True)}"
    )


def _fallback_action(obs: dict[str, Any], reason: str) -> dict[str, Any]:
    candidates = obs.get("candidate_recommendations", [])
    safe = [c for c in candidates if c.get("agent_name") != "adversarial_agent"]
    safe = sorted(safe, key=lambda c: (float(c.get("risk_score", 1.0)), -float(c.get("confidence", 0.0))))
    selected = safe[0] if safe else (candidates[0] if candidates else {"recommendation_id": 0})
    return OrchestratorDecision(
        selected_recommendation_id=int(selected.get("recommendation_id", 0)),
        override_adversary=True,
        reasoning=f"Deterministic fallback: {reason}",
        predicted_2step_impact=dict(selected.get("expected_impact", {}) or {}),
        schema_drift_detected=bool(obs.get("schema_drift_active", False)),
        schema_drift_field=obs.get("schema_drift_field"),
    ).model_dump()


def _model_action(client: OpenAI, model_name: str, obs: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": _prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return OrchestratorDecision.model_validate(parsed).model_dump()
        return _fallback_action(obs, "model returned no JSON")
    except Exception as exc:
        return _fallback_action(obs, f"model call failed: {type(exc).__name__}")


def run_task(task_id: str, client: OpenAI, model_name: str) -> dict[str, Any]:
    task = TASKS[task_id]
    env = AICEnvironment(
        episode_id=task.scenario_id,
        base_seed=42,
        scenario_id=task.scenario_id,
        use_llm_agents=False,
        include_network=True,
        include_security=True,
        manage_trust_scores=False,
    )
    obs = env.reset(options={"scenario_id": task.scenario_id})
    total_reward = 0.0
    done = False
    step = 0
    adversary_selected = 0
    unsafe_actions = 0

    _emit("START", {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "model": model_name,
        "scenario_id": task.scenario_id,
    })

    while not done and step < 20:
        action = _model_action(client, model_name, obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if info.get("selected_agent") == "adversarial_agent":
            adversary_selected += 1
        verifier_report = info.get("verifier_report", {}) or {}
        if float(verifier_report.get("risk_score", 0.0) or 0.0) > 0.8:
            unsafe_actions += 1
        _emit("STEP", {
            "task_id": task.id,
            "step": step,
            "reward": round(float(reward), 6),
            "health": round(float(info.get("health", 0.0)), 6),
            "done": bool(done),
            "selected_recommendation_id": action.get("selected_recommendation_id"),
        })
        step += 1

    total_steps = max(1, step)
    episode = {
        "reward": total_reward,
        "final_health": env.world_state.get_health_score(),
        "success": env.world_state.get_health_score() >= task.target_health,
        "adversary_suppression": 1.0 - adversary_selected / total_steps,
        "unsafe_rate": unsafe_actions / total_steps,
    }
    graded = grade_task(task.id, episode)
    _emit("END", {
        "task_id": task.id,
        "score": graded["score"],
        "success": graded["success"],
        "raw_reward": round(float(total_reward), 6),
        "final_health": round(float(episode["final_health"]), 6),
    })
    return {"task": task.id, **graded, "raw_reward": total_reward}


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    client = _client()
    results = [run_task(task_id, client, model_name) for task_id in TASKS]
    avg = sum(float(r["score"]) for r in results) / max(1, len(results))
    _emit("END", {
        "task_id": "all",
        "score": round(avg, 6),
        "success": bool(avg >= 0.5),
        "num_tasks": len(results),
    })


if __name__ == "__main__":
    main()

