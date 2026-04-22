"""Generate SFT examples from the heuristic orchestrator interacting with AICEnvironment."""
from __future__ import annotations

import json
from pathlib import Path

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.network_agent import NetworkAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.agents.security_agent import SecurityAgent
from aic.env.aic_environment import AICEnvironment
from aic.training.config import TrainingConfig
from aic.training.prompting import build_orchestrator_prompt, serialize_decision
from aic.training.rollout_env import make_structured_action, materialize_recommendations
from aic.utils.seeding import get_adversary_cycle, make_episode_rng


def generate_sft_dataset(config: TrainingConfig | None = None) -> Path:
    """Generate JSONL SFT records from heuristic orchestrator rollouts."""
    if config is None:
        config = TrainingConfig()

    output_path = Path(config.sft_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    net = NetworkAgent(use_llm=False)
    sec = SecurityAgent(use_llm=False)

    with open(output_path, "w") as f:
        for episode_id in range(config.sft_num_episodes):
            env = AICEnvironment(
                episode_id=episode_id,
                base_seed=config.base_seed,
                fault_mode=config.fault_mode,
                log_dir=config.log_dir,
                db_agent=db,
                infra_agent=infra,
                app_agent=app,
                net_agent=net,
                sec_agent=sec,
                include_network=True,
                include_security=True,
                manage_trust_scores=False,
            )
            obs = env.reset()

            cycle = get_adversary_cycle(make_episode_rng(episode_id, config.base_seed))
            adv = AdversarialAgent(cycle, db)
            orch = OrchestratorAgent(adv, use_llm=False)
            orch.mode = "trained"
            orch.reset()

            prev_metrics = obs["current_metrics"]
            done = False
            while not done:
                recommendations = materialize_recommendations(obs)
                action, override_applied = orch.decide(
                    step=obs["step"],
                    sla_remaining=obs["sla_remaining_steps"],
                    sub_agent_recommendations=recommendations,
                    alert_summary=obs["alert_summary_text"],
                    prev_metrics=prev_metrics,
                    current_metrics=obs["current_metrics"],
                )

                structured_action = make_structured_action(
                    obs,
                    action,
                    getattr(orch, "_followed_agent", None),
                    override_applied,
                )

                record = {
                    "prompt": build_orchestrator_prompt(obs),
                    "completion": serialize_decision(structured_action),
                    "episode_id": episode_id,
                    "step": obs["step"],
                    "fault_mode": config.fault_mode,
                }
                f.write(json.dumps(record) + "\n")

                env.trust_scores = orch.trust_scores.copy()
                obs, _reward, done, _info = env.step(structured_action)
                prev_metrics = obs["current_metrics"]

    return output_path


if __name__ == "__main__":
    path = generate_sft_dataset()
    print(path)
