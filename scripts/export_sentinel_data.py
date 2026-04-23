#!/usr/bin/env python3
"""Export dashboard trajectory data for Sentinel React UI."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.training.config import TrainingConfig
from aic.training.train import run_episode
from aic.utils.constants import ALL_AGENTS, INITIAL_TRUST
from aic.utils.seeding import get_adversary_cycle, make_episode_rng


class FrozenTrustOrchestrator(OrchestratorAgent):
    def _update_trust_scores(self, step, prev_metrics, current_metrics):
        self.trust_scores = {a: INITIAL_TRUST for a in ALL_AGENTS}


def _demo_episode(mode: str, episode_id: int) -> dict:
    config = TrainingConfig(num_episodes=1, use_llm_agents=False)
    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    cycle = get_adversary_cycle(make_episode_rng(episode_id, config.base_seed))
    adv = AdversarialAgent(cycle, db)
    orch = FrozenTrustOrchestrator(adv, use_llm=False) if mode == "untrained" else OrchestratorAgent(adv, use_llm=False)
    return run_episode(episode_id, config, orch, db, infra, app)


def _load_mode_data(root: Path, mode: str) -> dict[int, dict]:
    assets = root / "dashboard" / "assets"
    file_name = "trained_trajectories.pkl" if mode == "trained" else "untrained_trajectories.pkl"
    source = assets / file_name
    if source.exists():
        with source.open("rb") as fp:
            return pickle.load(fp)
    return {0: _demo_episode(mode, 0)}


def _serialize_mode(mode_data: dict[int, dict], episode_id: int) -> dict:
    available = sorted(mode_data.keys())
    selected = episode_id if episode_id in mode_data else available[0]
    return {
        "available_episodes": available,
        "selected_episode_id": selected,
        "episodes": {str(ep): mode_data[ep] for ep in available},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sentinel dashboard data JSON")
    parser.add_argument("--mode", choices=["trained", "untrained"], default="trained")
    parser.add_argument("--episode-id", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    trained = _load_mode_data(root, "trained")
    untrained = _load_mode_data(root, "untrained")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "dashboard/assets/*.pkl or generated demo",
        "mode": args.mode,
        "modes": {
            "trained": _serialize_mode(trained, args.episode_id),
            "untrained": _serialize_mode(untrained, args.episode_id),
        },
    }
    out_path = root / "sentinel---incident-command-center" / "public" / "data" / "dashboard-data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported Sentinel data -> {out_path}")


if __name__ == "__main__":
    main()
