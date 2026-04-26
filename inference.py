#!/usr/bin/env python3
"""Single-file inference entrypoint for the AIC OpenEnv environment.

This is the "baseline inference script" expected by the hackathon rubric and
copied into the submission bundle by ``scripts/build_submission_bundle.py``.

Behaviour:

  * Loads a trained policy if available:
      - ``--checkpoint exports/``  (merged model dir)
      - ``--checkpoint checkpoints/grpo/`` (LoRA adapter dir)
      - ``--hf-repo <user>/<repo>`` to pull the trained adapter from the Hub
    If none of those produce a usable model, falls back to the FrozenTrust
    baseline (``scripts.score_tasks.FrozenTrustPolicy``) so the script always
    runs end-to-end on a CPU dev box.
  * Runs one episode against ``AICEnvironment`` for each registered task.
  * Computes the rubric-mandated 0.0-1.0 grader score from
    :mod:`aic.tasks` and prints a summary.

Usage::

    python inference.py                                     # CPU-safe fallback
    python inference.py --checkpoint exports                # local merged model
    python inference.py --hf-repo KINGKK007/aic-grpo-qwen   # download adapter
    python inference.py --tasks db_pool_recovery --episodes 2

Outputs (when ``--out results/`` is set, default):

  * ``results/inference_summary.json``   - per-task scores + policy used
  * ``results/inference_traces.json``    - full per-step traces
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from aic.env.aic_environment import AICEnvironment
from aic.tasks import TASKS, grade_episode
from aic.schemas.actions import OrchestratorDecision

# Reuse the lean baselines from scripts.score_tasks so we have a deterministic
# CPU fallback when no checkpoint is available.
from scripts.score_tasks import FrozenTrustPolicy, AdaptiveTrustPolicy  # noqa: E402


def _load_trained_policy(checkpoint: str | None, hf_repo: str | None):
    """Try to load a trained model; return None if it can't be loaded."""
    if not (checkpoint or hf_repo):
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if hf_repo and not checkpoint:
            from huggingface_hub import snapshot_download

            ckpt_dir = snapshot_download(repo_id=hf_repo)
        else:
            ckpt_dir = checkpoint

        ckpt_path = Path(ckpt_dir)
        is_adapter = (ckpt_path / "adapter_config.json").exists()
        device_map = {"": 0} if torch.cuda.is_available() else None
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        if is_adapter:
            from peft import PeftModel

            with open(ckpt_path / "adapter_config.json") as f:
                cfg = json.load(f)
            base_name = str(cfg.get("base_model_name_or_path", ""))
            tok_dir = (
                str(ckpt_path) if (ckpt_path / "tokenizer_config.json").exists() else base_name
            )
            tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, str(ckpt_path))
        else:
            tok = AutoTokenizer.from_pretrained(str(ckpt_path), use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_path),
                device_map=device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        model.eval()
        print(f"[ok] trained policy loaded from {ckpt_dir}")
        return TrainedPolicy(model, tok)
    except Exception as exc:
        print(f"[warn] could not load trained policy: {exc}")
        return None


class TrainedPolicy:
    name = "trained_grpo"

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def select_action(self, obs: dict) -> dict:
        from aic.training.prompting import render_chat_prompt

        prompt = render_chat_prompt(self.tokenizer, obs)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=192,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        try:
            start = completion.find("{")
            end = completion.rfind("}") + 1
            if start >= 0 and end > start:
                return OrchestratorDecision.model_validate_json(
                    completion[start:end]
                ).model_dump()
        except Exception:
            pass
        return FrozenTrustPolicy().select_action(obs)


def _run_episode(policy, scenario_id: int, seed: int) -> list[dict]:
    env = AICEnvironment(
        episode_id=seed, base_seed=seed,
        fault_mode="cascading_failure",
        manage_trust_scores=True, scenario_id=scenario_id,
    )
    obs = env.reset(options={"scenario_id": scenario_id})
    trace = [
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
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)
        trace.append(
            {"step": int(info.get("step", safety)), "reward": float(reward), "info": dict(info)}
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="local merged model dir or LoRA adapter dir")
    parser.add_argument("--hf-repo", default=None,
                        help="HF Hub repo id of the trained adapter (e.g. KINGKK007/aic-grpo-qwen)")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trained = _load_trained_policy(args.checkpoint, args.hf_repo)
    policy = trained if trained is not None else FrozenTrustPolicy()
    print(f"[info] policy = {policy.name}")

    if args.tasks == "all":
        task_ids = sorted(TASKS.keys())
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    summary: dict[str, Any] = {"policy": policy.name, "tasks": {}}
    traces: dict[str, list] = {}

    for task_id in task_ids:
        task = TASKS[task_id]
        scores = []
        ep_traces = []
        for ep in range(args.episodes):
            seed = args.seed + (task.scenario_id * 100) + ep
            trace = _run_episode(policy, task.scenario_id, seed)
            score = grade_episode(task_id, trace)
            scores.append(score)
            ep_traces.append({"seed": seed, "score": score, "steps": len(trace) - 1})
            print(f"[run] {task_id:24s} ep{ep} (seed={seed}) score={score:.4f}")
        summary["tasks"][task_id] = {
            "difficulty": task.difficulty,
            "scenario_id": task.scenario_id,
            "mean_score": round(sum(scores) / len(scores), 4),
            "episodes": ep_traces,
        }
        traces[task_id] = ep_traces

    (out / "inference_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "inference_traces.json").write_text(json.dumps(traces, indent=2))
    print(f"\n[ok] wrote {out / 'inference_summary.json'}")
    print(f"[ok] wrote {out / 'inference_traces.json'}")


if __name__ == "__main__":
    main()
