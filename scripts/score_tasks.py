#!/usr/bin/env python3
"""Score every registered task against every available policy.

This is the rubric-mandated **0.0-1.0 task grader** report. It:
  * loads the 3 tasks from ``aic.tasks``
  * runs N episodes per (task, policy) pair against ``AICEnvironment``
    pinned to the task's scenario
  * collects per-step traces (current_metrics, verifier_report, adversary
    diagnostics) and feeds them to the task's grader
  * writes ``results/benchmark_by_task_grader.csv`` (per-(policy,task) rows)
    and ``results/benchmark_summary_normalized.csv`` (per-policy means)

Policies:
  * ``baseline_frozen``    - NoTrustOrchestratorPolicy (no learning)
  * ``baseline_adaptive``  - simple OrchestratorAgent w/ adaptive trust
  * ``random_safe``        - always picks the verifier's safe minimal action
  * ``trained_grpo``       - optional, only if a checkpoint is available

The trained_grpo column is gated by ``--checkpoint-path``. When the
checkpoint is absent (CPU dev box) we skip it and document the gap in the
output; the live HF Space + Colab notebook + ``logs/grpo_progress.jsonl``
are the canonical training-evidence artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from aic.env.aic_environment import AICEnvironment
from aic.tasks import TASKS, grade_episode
from aic.schemas.actions import OrchestratorDecision
from aic.utils.constants import AGENT_ADV, AGENT_VERIFIER


# ─────────────────────── policies ────────────────────────────────────────


class FrozenTrustPolicy:
    """Deterministic baseline: always picks the highest-confidence non-adversarial
    candidate. Equivalent to ``HighestConfidence`` but with a hard adversary
    veto. Approximates the original ``baseline_frozen`` from the legacy benchmark
    while keeping the loop simple."""

    name = "baseline_frozen"

    def select_action(self, obs: dict) -> dict:
        candidates = obs.get("candidate_recommendations", []) or []
        non_adv = [c for c in candidates if c.get("agent_name") != AGENT_ADV]
        if not non_adv:
            non_adv = candidates
        chosen = max(non_adv, key=lambda c: float(c.get("confidence", 0.0))) if non_adv else None
        chosen_id = int(chosen.get("recommendation_id", 0)) if chosen else 0
        return {
            "selected_recommendation_id": chosen_id,
            "override_adversary": True,
            "reasoning": "Frozen-trust baseline: pick top-confidence non-adversarial recommendation.",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }


class AdaptiveTrustPolicy:
    """Adaptive baseline: weights confidence by current trust score."""

    name = "baseline_adaptive"

    def select_action(self, obs: dict) -> dict:
        candidates = obs.get("candidate_recommendations", []) or []
        trust = obs.get("current_trust_scores", {}) or {}
        non_adv = [c for c in candidates if c.get("agent_name") != AGENT_ADV]
        if not non_adv:
            non_adv = candidates
        def _score(c: dict) -> float:
            agent = c.get("agent_name", "")
            return float(c.get("confidence", 0.0)) * float(trust.get(agent, 0.5))
        chosen = max(non_adv, key=_score) if non_adv else None
        chosen_id = int(chosen.get("recommendation_id", 0)) if chosen else 0
        return {
            "selected_recommendation_id": chosen_id,
            "override_adversary": True,
            "reasoning": "Adaptive baseline: argmax confidence * current_trust_score.",
            "predicted_2step_impact": {},
            "schema_drift_detected": bool(obs.get("schema_drift_active", False)),
            "schema_drift_field": obs.get("schema_drift_field"),
        }


class RandomSafePolicy:
    """Trivial baseline: always picks the verifier's safe minimal action."""

    name = "random_safe"

    def select_action(self, obs: dict) -> dict:
        candidates = obs.get("candidate_recommendations", []) or []
        verifier_id = next(
            (
                int(c.get("recommendation_id", 0))
                for c in candidates
                if c.get("agent_name") == AGENT_VERIFIER
            ),
            int(candidates[-1].get("recommendation_id", 0)) if candidates else 0,
        )
        return {
            "selected_recommendation_id": verifier_id,
            "override_adversary": True,
            "reasoning": "Always pick the verifier's safe minimal action.",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }


# ─────────────────────── episode loop ────────────────────────────────────


def _run_episode(policy, task_scenario_id: int, episode_seed: int) -> list[dict]:
    """Run one episode and return a step trace suitable for graders."""
    env = AICEnvironment(
        episode_id=episode_seed,
        base_seed=episode_seed,
        fault_mode="cascading_failure",
        use_llm_agents=False,
        manage_trust_scores=True,
        scenario_id=task_scenario_id,
    )
    obs = env.reset(options={"scenario_id": task_scenario_id})

    trace: list[dict] = []
    initial_metrics = obs.get("current_metrics", {})
    trace.append(
        {
            "step": 0,
            "info": {
                "current_metrics": dict(initial_metrics),
                "health": float(env.world_state.get_health_score()),
                "is_within_sla": bool(env.world_state.is_within_sla()),
                "verifier_report": {"approved": True},
                "adversary_present": True,
                "adversary_selected": False,
                "adversary_overridden": False,
            },
        }
    )

    done = False
    safety_counter = 0
    while not done and safety_counter < 64:
        safety_counter += 1
        action_dict = policy.select_action(obs)
        try:
            decision = OrchestratorDecision.model_validate(action_dict)
            action = decision.model_dump()
        except Exception:
            action = action_dict
        obs, reward, done, info = env.step(action)
        trace.append(
            {
                "step": int(info.get("step", safety_counter)),
                "reward": float(reward),
                "info": dict(info),
            }
        )

    return trace


# ─────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3,
                        help="episodes per (policy, task) pair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--policies",
        default="baseline_frozen,baseline_adaptive,random_safe",
        help="comma-separated subset of policies to run",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_policies = {
        "baseline_frozen": FrozenTrustPolicy(),
        "baseline_adaptive": AdaptiveTrustPolicy(),
        "random_safe": RandomSafePolicy(),
    }
    requested = [p.strip() for p in args.policies.split(",") if p.strip()]
    policies = [all_policies[p] for p in requested if p in all_policies]
    if not policies:
        raise SystemExit(f"No valid policies requested. Known: {sorted(all_policies)}")

    rows = []
    print(f"\n=== Task grader benchmark ===")
    print(f"  policies: {[p.name for p in policies]}")
    print(f"  tasks   : {sorted(TASKS.keys())}")
    print(f"  episodes per (policy, task): {args.episodes}")
    print()

    for policy in policies:
        print(f"[bench] policy={policy.name}")
        for task_id, task in sorted(TASKS.items()):
            scores = []
            for ep in range(args.episodes):
                seed = args.seed + (task.scenario_id * 100) + ep
                trace = _run_episode(policy, task.scenario_id, seed)
                score = grade_episode(task_id, trace)
                scores.append(score)
                print(
                    f"    {task_id:24s} ep{ep} (seed={seed}, scenario={task.scenario_id}) "
                    f"score={score:.4f}"
                )
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            success_rate = float(np.mean([s >= task.success_threshold for s in scores]))
            rows.append(
                {
                    "policy": policy.name,
                    "task_id": task_id,
                    "difficulty": task.difficulty,
                    "scenario_id": task.scenario_id,
                    "scenario_name": getattr(task, "title", task_id),
                    "num_episodes": args.episodes,
                    "mean_score_0_1": round(mean_score, 4),
                    "std_score_0_1": round(std_score, 4),
                    "success_rate": round(success_rate, 4),
                    "success_threshold": task.success_threshold,
                }
            )

    df = pd.DataFrame(rows)
    by_grader = out / "benchmark_by_task_grader.csv"
    df.to_csv(by_grader, index=False)
    print(f"\n[ok] wrote {by_grader}")

    pivot = (
        df.pivot_table(
            index="policy", columns="task_id", values="mean_score_0_1", aggfunc="mean"
        )
        .round(4)
    )
    pivot["mean"] = pivot.mean(axis=1).round(4)
    pivot = pivot.reset_index()
    summary_norm = out / "benchmark_summary_normalized.csv"
    pivot.to_csv(summary_norm, index=False)
    print(f"[ok] wrote {summary_norm}")

    manifest = {
        "policies": [p.name for p in policies],
        "tasks": [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty,
                "scenario_id": t.scenario_id,
                "title": t.title,
                "success_threshold": t.success_threshold,
            }
            for t in TASKS.values()
        ],
        "episodes_per_pair": args.episodes,
        "seed": args.seed,
        "source_files": {
            "graders": [
                "aic/tasks/task_db_pool_recovery.py",
                "aic/tasks/task_canary_blackout.py",
                "aic/tasks/task_adversarial_misroute.py",
            ],
        },
    }
    (out / "normalized_score_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote {out / 'normalized_score_manifest.json'}")
    print()
    print("=== summary (mean grader score in [0,1]) ===")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
