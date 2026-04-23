#!/usr/bin/env python3
"""
Mac-safe benchmark runner: compare two deterministic orchestrator policies.

Outputs:
  - logs/eval/policy_benchmark.jsonl (per-episode raw results)
  - results/benchmark_summary.csv    (aggregated table)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.training.config import TrainingConfig
from aic.training.train import run_episode
from aic.utils.constants import ALL_AGENTS, INITIAL_TRUST
from aic.utils.seeding import get_adversary_cycle, make_episode_rng


@dataclass
class EpisodeBenchmarkRow:
    policy: str
    episode_id: int
    total_reward: float
    final_health: float
    r2_bonus: float
    mttr: int
    success: bool
    override_rate: float


class FrozenTrustOrchestrator(OrchestratorAgent):
    """Disables any trust learning by resetting trust every step."""

    def _update_trust_scores(self, step, prev_metrics, current_metrics) -> None:  # type: ignore[override]
        self.trust_scores = {a: INITIAL_TRUST for a in ALL_AGENTS}


def _override_rate(trajectory: list[dict]) -> float:
    if not trajectory:
        return 0.0
    overrides = sum(1 for t in trajectory if t.get("override_applied"))
    return overrides / len(trajectory)


def run_benchmark(
    *,
    num_episodes: int = 10,
    base_seed: int = 42,
    fault_mode: str = "cascading_failure",
    output_jsonl: str = "logs/eval/policy_benchmark.jsonl",
    output_csv: str = "results/benchmark_summary.csv",
) -> tuple[list[EpisodeBenchmarkRow], dict[str, dict]]:
    Path("logs/eval").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        num_episodes=1,
        base_seed=base_seed,
        fault_mode=fault_mode,
        use_llm_agents=False,
        log_dir="logs",
    )
    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)

    # Shared adversary seed schedule, per-episode, via run_episode()
    dummy_cycle = get_adversary_cycle(make_episode_rng(0, base_seed))
    adv = AdversarialAgent(dummy_cycle, db)

    policies: list[tuple[str, OrchestratorAgent]] = []
    frozen = FrozenTrustOrchestrator(adv, use_llm=False)
    frozen.mode = "untrained"
    policies.append(("baseline_frozen_trust", frozen))

    trained = OrchestratorAgent(adv, use_llm=False)
    trained.mode = "trained"
    policies.append(("baseline_adaptive_trust", trained))

    rows: list[EpisodeBenchmarkRow] = []
    episode_offset = 10_000
    for idx in range(num_episodes):
        episode_id = episode_offset + idx
        for policy_name, orch in policies:
            result = run_episode(
                episode_id=episode_id,
                config=config,
                orchestrator=orch,
                db=db,
                infra=infra,
                app=app,
            )
            row = EpisodeBenchmarkRow(
                policy=policy_name,
                episode_id=episode_id,
                total_reward=float(result["total_reward"]),
                final_health=float(result["final_health"]),
                r2_bonus=float(result.get("r2_bonus", 0.0)),
                mttr=int(result.get("mttr", 0)),
                success=bool(result["final_health"] > 0.5),
                override_rate=float(_override_rate(result.get("trajectory", []))),
            )
            rows.append(row)

    # Write JSONL raw
    with open(output_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    # Aggregate summary
    by_policy: dict[str, list[EpisodeBenchmarkRow]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, []).append(r)

    summary: dict[str, dict] = {}
    for policy, rs in by_policy.items():
        n = max(1, len(rs))
        summary[policy] = {
            "episodes": len(rs),
            "avg_total_reward": sum(r.total_reward for r in rs) / n,
            "avg_final_health": sum(r.final_health for r in rs) / n,
            "success_rate": sum(1 for r in rs if r.success) / n,
            "avg_mttr": sum(r.mttr for r in rs) / n,
            "avg_override_rate": sum(r.override_rate for r in rs) / n,
        }

    # Write CSV summary
    fieldnames = [
        "policy",
        "episodes",
        "avg_total_reward",
        "avg_final_health",
        "success_rate",
        "avg_mttr",
        "avg_override_rate",
    ]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for policy, s in sorted(summary.items()):
            w.writerow({"policy": policy, **s})

    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--fault_mode", default="cascading_failure")
    p.add_argument("--output_jsonl", default="logs/eval/policy_benchmark.jsonl")
    p.add_argument("--output_csv", default="results/benchmark_summary.csv")
    args = p.parse_args()

    run_benchmark(
        num_episodes=args.num_episodes,
        base_seed=args.base_seed,
        fault_mode=args.fault_mode,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
    )
    print(f"✅ wrote {args.output_jsonl} and {args.output_csv}")


if __name__ == "__main__":
    main()

