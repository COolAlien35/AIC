#!/usr/bin/env python3
# scripts/run_final_benchmark.py
"""
Run the full AIC benchmark suite: AIC vs baselines across 6 scenarios.

Includes:
- 3 baseline policies + trained GRPO policy
- 5 episodes per scenario × 6 scenarios = 30 episodes per policy
- Statistical significance testing (t-test + Cohen's d)
- Per-scenario breakdown

Usage:
    python scripts/run_final_benchmark.py
    python scripts/run_final_benchmark.py --episodes 5 --output results/benchmark_summary.csv
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from aic.evals.benchmark_suite import (
    NoTrustOrchestratorPolicy,
    _run_baseline_episode,
    _run_aic_episode,
    SCENARIO_REGISTRY,
)


class TrainedGRPOPolicy:
    """The policy we actually trained. This is the star of the show."""

    name = "trained_grpo"

    def __init__(self, checkpoint_path: str = "checkpoints/grpo"):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _try_load(self):
        """Attempt to load the trained model. Gracefully degrade if missing."""
        if self._loaded:
            return
        self._loaded = True

        cp = Path(self.checkpoint_path)
        if not cp.exists():
            print(f"  ⚠️  No trained checkpoint at {cp}. Using fallback policy.")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"  Loading trained model from {self.checkpoint_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.model.eval()
            print(f"  ✅ Trained model loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Could not load trained model: {e}. Using fallback.")

    def select_action(self, obs: dict) -> dict:
        """Generate a decision from the trained model or fall back to heuristic."""
        self._try_load()

        if self.model is not None and self.tokenizer is not None:
            from aic.training.prompting import build_orchestrator_prompt
            import torch

            prompt = build_orchestrator_prompt(obs)
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=1024
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            completion = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
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

        # Graceful fallback — always return valid action
        candidates = obs.get("candidate_recommendations", [])
        safe = [c for c in candidates if c.get("agent_name") != "adversarial_agent"]
        chosen_id = safe[0].get("recommendation_id", 0) if safe else 0
        return {
            "selected_recommendation_id": chosen_id,
            "override_adversary": len(safe) < len(candidates),
            "reasoning": "Fallback: selecting safest candidate.",
        }


def run_single_episode(policy_name: str, scenario_id: int, episode_seed: int) -> dict:
    """Run one benchmark episode for a named policy."""
    if policy_name == "baseline_frozen":
        result = _run_baseline_episode(NoTrustOrchestratorPolicy(), scenario_id, episode_seed)
    elif policy_name == "baseline_adaptive":
        result = _run_aic_episode(scenario_id, mode="untrained", episode_seed=episode_seed)
    elif policy_name == "trained_grpo":
        result = _run_aic_episode(scenario_id, mode="trained", episode_seed=episode_seed)
    else:
        raise ValueError(f"Unknown policy {policy_name}")

    return {
        "policy": policy_name,
        "scenario": result.scenario_name,
        "reward": float(result.total_reward),
        "success": bool(result.sla_met),
        "mttr": int(result.mttr_steps),
        "adversary_suppression": float(result.adversary_suppression_rate),
        "unsafe_rate": float(result.unsafe_action_rate),
    }


def run_benchmark(
    num_episodes_per_scenario: int = 5,
    output_dir: str = "results",
    seed: int = 42,
):
    """Run 3-policy benchmark with significance testing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    policies = ("baseline_frozen", "baseline_adaptive", "trained_grpo")
    scenario_ids = sorted(SCENARIO_REGISTRY.keys())

    records = []
    for policy_name in policies:
        print(f"\n🔄 Benchmarking: {policy_name}")
        for scenario_id in scenario_ids:
            for ep_idx in range(num_episodes_per_scenario):
                ep_seed = seed + (scenario_id * 100) + ep_idx
                result = run_single_episode(policy_name, scenario_id, ep_seed)
                records.append(result)
                print(
                    f"  {result['scenario']} ep{ep_idx}: "
                    f"reward={result['reward']:.2f}, success={result['success']}"
                )

    df = pd.DataFrame(records)

    # Create summary by policy
    summary = df.groupby("policy").agg(
        avg_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        success_rate=("success", "mean"),
        num_episodes=("reward", "count"),
    ).reset_index()

    summary.to_csv(f"{output_dir}/benchmark_summary.csv", index=False)

    # Per-scenario breakdown
    scenario_summary = df.groupby(["policy", "scenario"]).agg(
        avg_reward=("reward", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()
    scenario_summary.to_csv(f"{output_dir}/benchmark_by_scenario.csv", index=False)

    baseline_rewards = df[df["policy"] == "baseline_frozen"]["reward"].values
    trained_rewards = df[df["policy"] == "trained_grpo"]["reward"].values

    t_stat, p_value = stats.ttest_ind(baseline_rewards, trained_rewards)
    pooled_std = np.sqrt((np.std(baseline_rewards) ** 2 + np.std(trained_rewards) ** 2) / 2)
    cohens_d = (np.mean(trained_rewards) - np.mean(baseline_rewards)) / (pooled_std + 1e-9)

    stats_output = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "cohens_d": float(cohens_d),
        "effect_size_label": (
            "large" if abs(cohens_d) > 0.8 else
            "medium" if abs(cohens_d) > 0.5 else "small"
        ),
        "baseline_mean": float(np.mean(baseline_rewards)),
        "trained_mean": float(np.mean(trained_rewards)),
        "improvement": float(np.mean(trained_rewards) - np.mean(baseline_rewards)),
        "improvement_pct": float(
            (np.mean(trained_rewards) - np.mean(baseline_rewards))
            / (abs(np.mean(baseline_rewards)) + 1e-9) * 100
        ),
    }

    with open(f"{output_dir}/statistical_test.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    # Print results
    print(f"\n📊 BENCHMARK COMPLETE")
    print(f"\n{summary.to_string(index=False)}")

    print(f"\n📈 STATISTICAL TEST:")
    print(f"   Baseline avg reward: {stats_output['baseline_mean']:.2f}")
    print(f"   Trained avg reward:  {stats_output['trained_mean']:.2f}")
    print(f"   Improvement:         {stats_output['improvement']:+.2f} ({stats_output['improvement_pct']:+.1f}%)")
    sig = '✅ SIGNIFICANT' if stats_output['significant'] else '⚠️ not significant'
    print(f"   p-value:             {stats_output['p_value']:.4f} ({sig})")
    print(f"   Cohen's d:           {stats_output['cohens_d']:.3f} ({stats_output['effect_size_label']} effect)")

    return df, stats_output


def main():
    parser = argparse.ArgumentParser(description="AIC Extended Benchmark Suite")
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per scenario")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--scenarios", default="all",
        help="Comma-separated scenario list or 'all'",
    )
    parser.add_argument(
        "--policies", default="all",
        help="Comma-separated policy list or 'all'",
    )
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print("\n[bold cyan]═══ AIC Extended Benchmark Suite ═══[/bold cyan]\n")
        console.print("Running AIC vs baselines across 6 scenarios with statistical testing...\n")
    except ImportError:
        print("\n═══ AIC Extended Benchmark Suite ═══\n")

    df, stats = run_benchmark(
        num_episodes_per_scenario=args.episodes,
        output_dir=args.output,
        seed=args.seed,
    )

    print(f"\n✅ Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
