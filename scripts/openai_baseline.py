#!/usr/bin/env python3
"""OpenAI baseline policy for the AIC OpenEnv environment.

This is the rubric-mandated **OpenAI API client baseline**. Judges run it
with their own ``OPENAI_API_KEY`` to produce comparable per-task scores
without ever needing to train anything.

What it does:
  1. Spins up ``AICEnvironment`` pinned to each task's scenario.
  2. At each step, builds a structured system + user prompt describing the
     observation, candidate recommendations, trust scores, and action
     schema. Calls ``openai.chat.completions.create`` (default model
     ``gpt-4o-mini``).
  3. Parses the JSON response into ``OrchestratorDecision``. On parse
     failure or schema violation, falls back to the verifier's safe
     minimal action so a flaky LLM never crashes a benchmark run.
  4. Collects per-step traces, runs each task's grader from
     :mod:`aic.tasks`, and writes:
       - ``results/openai_baseline_scores.json``
       - ``results/openai_baseline_episodes.csv``
       - one row per (task, episode) with grader_score in [0, 1].

Usage::

    pip install openai
    export OPENAI_API_KEY=sk-...
    python scripts/openai_baseline.py --episodes 3
    python scripts/openai_baseline.py --model gpt-4o --episodes 5
    python scripts/openai_baseline.py --tasks db_pool_recovery,canary_blackout

The script is intentionally cost-aware: it caps total OpenAI calls at
``max_calls`` (default 200) and prints a running cost estimate at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.env.aic_environment import AICEnvironment
from aic.schemas.actions import OrchestratorDecision
from aic.tasks import TASKS, grade_episode

SYSTEM_PROMPT = """You are AIC, an autonomous incident-response orchestrator.
At each step you receive a structured observation (current metrics, candidate
recommendations from specialist agents, trust scores, scenario context).
You MUST emit a single JSON object that conforms to the OrchestratorDecision
schema below and nothing else. Do not include markdown fences, prose, or
explanations outside the JSON.

OrchestratorDecision schema:
{
  "selected_recommendation_id": int,        // id from candidate_recommendations
  "override_adversary": bool,               // true if you reject adversary recs
  "reasoning": str (>= 10 chars),
  "predicted_2step_impact": dict[str, float],
  "schema_drift_detected": bool,
  "schema_drift_field": str | null
}

Hard rules:
- Never select a recommendation from agent_name == "adversarial_agent" unless
  you also set override_adversary=false (you should almost never do this).
- If schema_drift_active is true, set schema_drift_detected=true.
- Prefer recommendations that target the most-degraded metrics.
- When in doubt, pick the verifier's safe minimal action (low risk_score).
"""


def _build_user_prompt(obs: dict) -> str:
    """Compress the observation into a deterministic prompt the LLM can act on."""
    cands = obs.get("candidate_recommendations", []) or []
    cand_lines = []
    for c in cands:
        cand_lines.append(
            f"  id={c.get('recommendation_id')} agent={c.get('agent_name')} "
            f"action={c.get('action')!r} confidence={c.get('confidence', 0):.2f} "
            f"risk={c.get('risk_score', 0):.2f} blast={c.get('blast_radius', 0):.2f} "
            f"targets={c.get('target_metrics', [])}"
        )
    metrics = obs.get("current_metrics", {}) or {}
    metric_lines = [f"  {k}={v}" for k, v in sorted(metrics.items())]
    trust_lines = [f"  {k}={v:.2f}" for k, v in (obs.get("current_trust_scores", {}) or {}).items()]

    drift_active = bool(obs.get("schema_drift_active", False))
    return (
        "ALERT SUMMARY:\n"
        f"{obs.get('alert_summary_text', '(none)')}\n\n"
        f"step={obs.get('step', 0)} sla_remaining={obs.get('sla_remaining_steps', 0)} "
        f"scenario_id={obs.get('scenario_id')} "
        f"schema_drift_active={drift_active}\n\n"
        "CURRENT METRICS:\n" + "\n".join(metric_lines) + "\n\n"
        "TRUST SCORES:\n" + "\n".join(trust_lines) + "\n\n"
        "CANDIDATE RECOMMENDATIONS:\n" + "\n".join(cand_lines) + "\n\n"
        "Emit the OrchestratorDecision JSON now."
    )


def _safe_minimal_action(obs: dict) -> dict:
    cands = obs.get("candidate_recommendations", []) or []
    verifier_id = next(
        (
            int(c.get("recommendation_id", 0))
            for c in cands
            if c.get("agent_name") == "recovery_verifier"
        ),
        int(cands[-1].get("recommendation_id", 0)) if cands else 0,
    )
    return {
        "selected_recommendation_id": verifier_id,
        "override_adversary": True,
        "reasoning": "Fallback: parse failure on LLM response, deferring to verifier.",
        "predicted_2step_impact": {},
        "schema_drift_detected": bool(obs.get("schema_drift_active", False)),
        "schema_drift_field": obs.get("schema_drift_field"),
    }


def _parse_llm_action(content: str, obs: dict) -> dict:
    """Best-effort JSON extraction + schema validation."""
    if not content:
        return _safe_minimal_action(obs)
    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return _safe_minimal_action(obs)
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        decision = OrchestratorDecision.model_validate(parsed)
        return decision.model_dump()
    except Exception:
        return _safe_minimal_action(obs)


def _run_episode_with_openai(
    client,
    task_scenario_id: int,
    episode_seed: int,
    model: str,
    call_counter: list[int],
    max_calls: int,
) -> list[dict]:
    env = AICEnvironment(
        episode_id=episode_seed,
        base_seed=episode_seed,
        fault_mode="cascading_failure",
        manage_trust_scores=True,
        scenario_id=task_scenario_id,
    )
    obs = env.reset(options={"scenario_id": task_scenario_id})

    trace: list[dict] = [
        {
            "step": 0,
            "info": {
                "current_metrics": dict(obs.get("current_metrics", {})),
                "health": float(env.world_state.get_health_score()),
                "is_within_sla": False,
                "verifier_report": {"approved": True},
                "adversary_present": True,
                "adversary_selected": False,
                "adversary_overridden": False,
            },
        }
    ]
    done = False
    safety = 0
    while not done and safety < 64:
        safety += 1
        if call_counter[0] >= max_calls:
            action = _safe_minimal_action(obs)
        else:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(obs)},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )
                call_counter[0] += 1
                action = _parse_llm_action(resp.choices[0].message.content, obs)
            except Exception as exc:
                print(f"    [warn] OpenAI call failed: {exc}; using safe action")
                action = _safe_minimal_action(obs)
        obs, reward, done, info = env.step(action)
        trace.append(
            {
                "step": int(info.get("step", safety)),
                "reward": float(reward),
                "info": dict(info),
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    parser.add_argument("--tasks", default="all", help="comma-separated task ids or 'all'")
    parser.add_argument("--max-calls", type=int, default=200,
                        help="hard cap on OpenAI API calls (cost guard)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY env var is required. Get a key at "
            "https://platform.openai.com/api-keys and re-run."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            f"openai SDK not installed: {exc}. Run `pip install openai` first."
        )

    client = OpenAI(api_key=api_key)

    task_ids = sorted(TASKS.keys()) if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]
    for tid in task_ids:
        if tid not in TASKS:
            raise SystemExit(f"Unknown task_id {tid!r}; known: {sorted(TASKS.keys())}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    call_counter = [0]
    rows: list[dict[str, Any]] = []

    print(f"\n=== OpenAI baseline ({args.model}) ===")
    print(f"  tasks   : {task_ids}")
    print(f"  episodes per task: {args.episodes}")
    print(f"  hard cap : {args.max_calls} OpenAI calls\n")

    started = time.time()
    for task_id in task_ids:
        task = TASKS[task_id]
        print(f"[bench] task={task_id} (scenario={task.scenario_id}, difficulty={task.difficulty})")
        for ep in range(args.episodes):
            seed = args.seed + (task.scenario_id * 100) + ep
            trace = _run_episode_with_openai(
                client, task.scenario_id, seed,
                args.model, call_counter, args.max_calls,
            )
            score = grade_episode(task_id, trace)
            rows.append(
                {
                    "policy": f"openai_baseline_{args.model}",
                    "task_id": task_id,
                    "difficulty": task.difficulty,
                    "scenario_id": task.scenario_id,
                    "episode_seed": seed,
                    "episode_steps": len(trace) - 1,
                    "grader_score": round(score, 4),
                    "openai_calls": call_counter[0],
                }
            )
            print(f"    ep{ep} (seed={seed}) score={score:.4f} calls={call_counter[0]}")

    elapsed = time.time() - started
    summary = {
        "model": args.model,
        "tasks": task_ids,
        "episodes_per_task": args.episodes,
        "rows": rows,
        "totals": {
            "total_openai_calls": call_counter[0],
            "elapsed_seconds": round(elapsed, 1),
            "approx_cost_usd_4o_mini": round(call_counter[0] * 0.00015, 4),
        },
        "note": (
            "Run reproduces the rubric-mandated OpenAI baseline column. "
            "Scores come from aic.tasks.* graders (deterministic 0.0-1.0)."
        ),
    }
    (out / "openai_baseline_scores.json").write_text(json.dumps(summary, indent=2))

    import csv

    with open(out / "openai_baseline_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[ok] wrote {out / 'openai_baseline_scores.json'}")
    print(f"[ok] wrote {out / 'openai_baseline_episodes.csv'}")
    print(f"     total OpenAI calls: {call_counter[0]}")
    print(f"     elapsed:           {elapsed:.1f}s")


if __name__ == "__main__":
    main()
