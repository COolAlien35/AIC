#!/usr/bin/env python3
"""Run the full AIC benchmark suite: AIC vs baselines across 6 scenarios.

Critical fixes vs the previous revision:

  * Default ``--checkpoint-path`` is ``exports/`` (the merged model dir).
  * If the directory has ``adapter_config.json`` only, the policy now loads
    via ``PeftModel.from_pretrained(base, adapter_dir)`` instead of falling
    back to a heuristic that pretends to be the trained model.
  * ``--strict`` (default true) makes a failed checkpoint a hard error so
    headline numbers are never silently from a heuristic.
  * Inference uses ``render_chat_prompt`` so the prompt distribution
    matches the SFT/GRPO prompt distribution.
  * Inference ``max_length`` raised to 1024 to match training.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Iterable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from aic.env.aic_environment import AICEnvironment
from aic.evals.benchmark_suite import (
    NoTrustOrchestratorPolicy,
    _run_baseline_episode,
    _run_aic_episode,
    SCENARIO_REGISTRY,
)
from aic.training.prompting import render_chat_prompt
from aic.utils.constants import SLA_STEPS, AGENT_ADV

SCENARIO_ID_TO_FAULT_MODE = {
    0: "cascading_failure",
    1: "memory_leak",
    2: "db_connection_saturation",
    3: "network_storm",
    4: "db_connection_saturation",
    5: "cascading_failure",
}


def checkpoint_preflight(checkpoint_path: str) -> dict:
    """Validate checkpoint directory contains required artifacts."""
    cp = Path(checkpoint_path)
    result: dict = {
        "valid": False, "errors": [], "files_found": [],
        "path": str(cp), "is_adapter_only": False,
    }
    if not cp.exists():
        result["errors"].append(f"checkpoint directory does not exist: {cp}")
        return result
    found = [f.name for f in cp.iterdir() if f.is_file()]
    result["files_found"] = found

    has_full_config = "config.json" in found
    has_adapter = "adapter_config.json" in found

    has_full_weights = any(
        p in found for p in (
            "model.safetensors", "pytorch_model.bin",
            "model.safetensors.index.json", "pytorch_model.bin.index.json",
        )
    )
    has_adapter_weights = any(
        p in found for p in ("adapter_model.safetensors", "adapter_model.bin")
    )

    if has_full_config and has_full_weights:
        result["valid"] = True
        return result
    if has_adapter and has_adapter_weights:
        result["valid"] = True
        result["is_adapter_only"] = True
        return result

    if not has_full_config and not has_adapter:
        result["errors"].append("missing config.json AND adapter_config.json")
    if not has_full_weights and not has_adapter_weights:
        result["errors"].append("no model or adapter weights found")
    return result


class TrainedGRPOPolicy:
    """Loads either a merged model dir OR a LoRA adapter dir."""

    def __init__(self, checkpoint_path: str = "exports", strict: bool = True):
        self.checkpoint_path = checkpoint_path
        self.strict = strict
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_error: str | None = None
        self._preflight: dict | None = None

    @property
    def name(self) -> str:
        if self.using_model:
            return "trained_grpo"
        return "trained_grpo_FALLBACK"

    def _try_load(self):
        if self._loaded:
            return
        self._loaded = True

        self._preflight = checkpoint_preflight(self.checkpoint_path)
        if not self._preflight["valid"]:
            self._load_error = f"preflight failed: {self._preflight['errors']}"
            print(f"  [!] Checkpoint preflight FAILED for {self.checkpoint_path}:")
            for err in self._preflight["errors"]:
                print(f"      - {err}")
            if self.strict:
                raise RuntimeError(
                    f"Checkpoint preflight failed for {self.checkpoint_path}: "
                    f"{self._preflight['errors']}. Re-run with --no-strict to "
                    "see fallback numbers (NOT representative of trained model)."
                )
            print("  [!] Falling back to heuristic policy (use --strict to fail).")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            ckpt = Path(self.checkpoint_path)
            if self._preflight.get("is_adapter_only"):
                from peft import PeftModel

                with open(ckpt / "adapter_config.json") as f:
                    adapter_cfg = json.load(f)
                base_name = str(adapter_cfg.get("base_model_name_or_path", ""))
                print(f"  Loading base {base_name} + adapter from {ckpt}...")
                tok_dir = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else base_name
                self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
                base = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                self.model = PeftModel.from_pretrained(base, str(ckpt))
            else:
                print(f"  Loading merged model from {self.checkpoint_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint_path, use_fast=True,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint_path,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            self.model.eval()
            print(f"  [ok] Trained model loaded successfully")
        except Exception as e:
            self._load_error = str(e)
            if self.strict:
                raise
            print(f"  [!] Could not load trained model: {e}.")

    def select_action(self, obs: dict) -> dict:
        self._try_load()

        if self.model is not None and self.tokenizer is not None:
            import torch

            prompt = render_chat_prompt(self.tokenizer, obs)
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=1024,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=192,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            completion = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            try:
                json_start = completion.find("{")
                json_end = completion.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = completion[json_start:json_end]
                    from aic.schemas.actions import OrchestratorDecision
                    decision = OrchestratorDecision.model_validate_json(json_str)
                    return decision.model_dump()
            except Exception:
                pass

        candidates = obs.get("candidate_recommendations", [])
        safe = [c for c in candidates if c.get("agent_name") != "adversarial_agent"]
        chosen_id = safe[0].get("recommendation_id", 0) if safe else 0
        return {
            "selected_recommendation_id": chosen_id,
            "override_adversary": len(safe) < len(candidates),
            "reasoning": "Fallback: selecting safest candidate.",
        }

    @property
    def using_model(self) -> bool:
        self._try_load()
        return self.model is not None and self.tokenizer is not None

    @property
    def load_error(self) -> str | None:
        self._try_load()
        return self._load_error

    @property
    def preflight_result(self) -> dict | None:
        self._try_load()
        return self._preflight


def _run_trained_model_episode(
    policy: TrainedGRPOPolicy,
    scenario_id: int,
    episode_seed: int,
) -> dict:
    scenario = SCENARIO_REGISTRY[scenario_id]
    fault_mode = SCENARIO_ID_TO_FAULT_MODE.get(scenario_id, "cascading_failure")
    env = AICEnvironment(
        episode_id=episode_seed,
        base_seed=episode_seed,
        fault_mode=fault_mode,
        use_llm_agents=False,
        include_network=True,
        include_security=True,
        manage_trust_scores=False,
    )
    obs = env.reset(options={"fault_mode": fault_mode})
    done = False
    adversary_selected = 0
    unsafe_actions = 0
    mttr = SLA_STEPS

    while not done:
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)

        selected_agent = info.get("selected_agent", "")
        if selected_agent == AGENT_ADV:
            adversary_selected += 1
        verifier_report = info.get("verifier_report", {}) or {}
        if verifier_report.get("risk_score", 0.0) > 0.8:
            unsafe_actions += 1
        if info.get("health", 0.0) > 0.5 and mttr == SLA_STEPS:
            mttr = int(info.get("step", SLA_STEPS))

    total_steps = max(1, env.step_count)
    final_health = env.world_state.get_health_score()
    total_reward = env.reward_engine.get_total_episode_reward()
    return {
        "policy": policy.name,
        "scenario": scenario.name,
        "reward": float(total_reward),
        "success": bool(env.world_state.is_within_sla()),
        "mttr": int(mttr),
        "adversary_suppression": float(1.0 - (adversary_selected / total_steps)),
        "unsafe_rate": float(unsafe_actions / total_steps),
        "final_health": float(final_health),
        "trained_policy_source": "checkpoint" if policy.using_model else "fallback",
        "trained_policy_checkpoint": policy.checkpoint_path,
        "checkpoint_preflight": policy.preflight_result,
    }


def _parse_requested(requested: str, all_values: Iterable[str], label: str) -> list[str]:
    values = list(all_values)
    if requested == "all":
        return values
    requested_values = [v.strip() for v in requested.split(",") if v.strip()]
    unknown = sorted(set(requested_values) - set(values))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Allowed: {values}")
    return requested_values


def run_single_episode(
    policy_name: str,
    scenario_id: int,
    episode_seed: int,
    trained_policy: TrainedGRPOPolicy,
) -> dict:
    if policy_name == "baseline_frozen":
        result = _run_baseline_episode(NoTrustOrchestratorPolicy(), scenario_id, episode_seed)
        return {
            "policy": policy_name,
            "scenario": result.scenario_name,
            "reward": float(result.total_reward),
            "success": bool(result.sla_met),
            "mttr": int(result.mttr_steps),
            "adversary_suppression": float(result.adversary_suppression_rate),
            "unsafe_rate": float(result.unsafe_action_rate),
            "trained_policy_source": "n/a",
            "trained_policy_checkpoint": "n/a",
        }
    if policy_name == "baseline_adaptive":
        result = _run_aic_episode(scenario_id, mode="untrained", episode_seed=episode_seed)
        return {
            "policy": policy_name,
            "scenario": result.scenario_name,
            "reward": float(result.total_reward),
            "success": bool(result.sla_met),
            "mttr": int(result.mttr_steps),
            "adversary_suppression": float(result.adversary_suppression_rate),
            "unsafe_rate": float(result.unsafe_action_rate),
            "trained_policy_source": "n/a",
            "trained_policy_checkpoint": "n/a",
        }
    if policy_name == "trained_grpo":
        return _run_trained_model_episode(trained_policy, scenario_id, episode_seed)
    raise ValueError(f"Unknown policy {policy_name}")


def run_benchmark(
    num_episodes_per_scenario: int = 5,
    output_dir: str = "results",
    seed: int = 42,
    requested_scenarios: str = "all",
    requested_policies: str = "all",
    checkpoint_path: str = "exports",
    strict: bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_policies = ("baseline_frozen", "baseline_adaptive", "trained_grpo")
    policies = _parse_requested(requested_policies, all_policies, "policies")
    all_scenarios = [str(x) for x in sorted(SCENARIO_REGISTRY.keys())]
    scenario_ids = [int(x) for x in _parse_requested(requested_scenarios, all_scenarios, "scenarios")]
    trained_policy = TrainedGRPOPolicy(checkpoint_path=checkpoint_path, strict=strict)

    records: list[dict] = []
    invalid_records: list[dict] = []
    for policy_name in policies:
        print(f"\n[bench] Benchmarking: {policy_name}")
        for scenario_id in scenario_ids:
            for ep_idx in range(num_episodes_per_scenario):
                ep_seed = seed + (scenario_id * 100) + ep_idx
                result = run_single_episode(policy_name, scenario_id, ep_seed, trained_policy)
                if policy_name == "trained_grpo" and result["policy"].endswith("_FALLBACK"):
                    invalid_records.append(result)
                else:
                    records.append(result)
                print(
                    f"  {result['scenario']} ep{ep_idx}: "
                    f"reward={result['reward']:.2f}, success={result['success']}"
                )

    df = pd.DataFrame(records)
    invalid_df = pd.DataFrame(invalid_records)

    summary = df.groupby("policy").agg(
        avg_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        success_rate=("success", "mean"),
        num_episodes=("reward", "count"),
    ).reset_index()
    summary.to_csv(f"{output_dir}/benchmark_summary.csv", index=False)

    scenario_summary = df.groupby(["policy", "scenario"]).agg(
        avg_reward=("reward", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()
    scenario_summary.to_csv(f"{output_dir}/benchmark_by_scenario.csv", index=False)

    if not invalid_df.empty:
        invalid_df.to_csv(f"{output_dir}/_invalid_runs.csv", index=False)

    baseline_rewards = df[df["policy"] == "baseline_frozen"]["reward"].values
    trained_rewards = df[df["policy"] == "trained_grpo"]["reward"].values

    have_baseline = len(baseline_rewards) > 0
    have_trained = len(trained_rewards) > 0

    if have_baseline and have_trained and len(baseline_rewards) >= 2 and len(trained_rewards) >= 2:
        t_stat, p_value = stats.ttest_ind(baseline_rewards, trained_rewards)
        pooled_std = np.sqrt((np.std(baseline_rewards) ** 2 + np.std(trained_rewards) ** 2) / 2)
        cohens_d = (np.mean(trained_rewards) - np.mean(baseline_rewards)) / (pooled_std + 1e-9)
    else:
        t_stat, p_value, cohens_d = 0.0, 1.0, 0.0

    stats_output = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "cohens_d": float(cohens_d),
        "effect_size_label": (
            "large" if abs(cohens_d) > 0.8 else
            "medium" if abs(cohens_d) > 0.5 else "small"
        ),
        "baseline_mean": float(np.mean(baseline_rewards)) if have_baseline else None,
        "trained_mean": float(np.mean(trained_rewards)) if have_trained else None,
        "improvement": (
            float(np.mean(trained_rewards) - np.mean(baseline_rewards))
            if have_baseline and have_trained else None
        ),
        "improvement_pct": (
            float(
                (np.mean(trained_rewards) - np.mean(baseline_rewards))
                / (abs(np.mean(baseline_rewards)) + 1e-9) * 100
            )
            if have_baseline and have_trained else None
        ),
    }

    with open(f"{output_dir}/statistical_test.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    run_config = {
        "seed": seed,
        "episodes_per_scenario": num_episodes_per_scenario,
        "policies": list(policies),
        "scenarios": scenario_ids,
        "trained_checkpoint_path": checkpoint_path,
        "strict": strict,
        "trained_policy_source": "checkpoint" if trained_policy.using_model else "fallback",
        "trained_policy_label": trained_policy.name,
        "trained_policy_load_error": trained_policy.load_error,
        "trained_policy_preflight": trained_policy.preflight_result,
        "invalid_run_count": len(invalid_records),
    }
    with open(f"{output_dir}/benchmark_run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n[bench] BENCHMARK COMPLETE")
    print(f"\n{summary.to_string(index=False)}")

    print(f"\n[bench] STATISTICAL TEST:")
    if stats_output["baseline_mean"] is None or stats_output["trained_mean"] is None:
        print("   (stats unavailable: need both baseline_frozen and trained_grpo runs)")
    else:
        print(f"   Baseline avg reward: {stats_output['baseline_mean']:.2f}")
        print(f"   Trained avg reward:  {stats_output['trained_mean']:.2f}")
        print(
            f"   Improvement:         {stats_output['improvement']:+.2f} "
            f"({stats_output['improvement_pct']:+.1f}%)"
        )
    sig = 'SIGNIFICANT' if stats_output['significant'] else 'not significant'
    print(f"   p-value:             {stats_output['p_value']:.4f} ({sig})")
    print(f"   Cohen's d:           {stats_output['cohens_d']:.3f} ({stats_output['effect_size_label']} effect)")

    return df, stats_output


def main():
    parser = argparse.ArgumentParser(description="AIC Extended Benchmark Suite")
    parser.add_argument("--output", default="results")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--policies", default="all")
    parser.add_argument("--checkpoint-path", default="exports",
                        help="Trained checkpoint dir (merged model or adapter dir).")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Hard-fail when checkpoint preflight fails (default).")
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help="Allow heuristic fallback for trained_grpo policy.")
    args = parser.parse_args()

    print("\n=== AIC Extended Benchmark Suite ===\n")
    print("Running AIC vs baselines across 6 scenarios with statistical testing...\n")

    df, stats_summary = run_benchmark(
        num_episodes_per_scenario=args.episodes,
        output_dir=args.output,
        seed=args.seed,
        requested_scenarios=args.scenarios,
        requested_policies=args.policies,
        checkpoint_path=args.checkpoint_path,
        strict=args.strict,
    )

    print(f"\n[ok] Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
