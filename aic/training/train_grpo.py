"""TRL GRPO training entrypoint for verifiable single-step AIC rollouts."""
from __future__ import annotations

import json
from pathlib import Path

from aic.env.aic_environment import AICEnvironment
from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.modeling_unsloth import load_model_and_tokenizer
from aic.training.prompting import build_orchestrator_prompt


def generate_grpo_prompt_dataset(config: TrainingConfig | None = None) -> Path:
    """Generate prompt-only JSONL records for single-step GRPO training."""
    if config is None:
        config = TrainingConfig()

    path = Path(config.grpo_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for episode_id in range(config.sft_num_episodes):
            env = AICEnvironment(
                episode_id=episode_id,
                base_seed=config.base_seed,
                fault_mode=config.fault_mode,
                use_llm_agents=False,
                manage_trust_scores=False,
            )
            obs = env.reset()
            record = {
                "prompt": build_orchestrator_prompt(obs),
                "episode_id": episode_id,
                "base_seed": config.base_seed,
                "fault_mode": config.fault_mode,
            }
            f.write(json.dumps(record) + "\n")
    return path


def run_grpo(config: TrainingConfig | None = None) -> Path:
    """Run a single-step GRPO/RLVR loop when TRL is available."""
    if config is None:
        config = TrainingConfig()

    try:  # pragma: no cover - optional dependency
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise ImportError(
            "GRPO dependencies missing. Install `trl`, `datasets`, and model backends first."
        ) from exc

    dataset_path = Path(config.grpo_dataset_path)
    if not dataset_path.exists():
        dataset_path = generate_grpo_prompt_dataset(config)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    model, tokenizer, backend_info = load_model_and_tokenizer(config)

    def reward_func(completions, **kwargs):
        rewards = []
        for completion, episode_id, base_seed, fault_mode in zip(
            completions,
            kwargs.get("episode_id", []),
            kwargs.get("base_seed", []),
            kwargs.get("fault_mode", []),
        ):
            env = AICEnvironment(
                episode_id=int(episode_id),
                base_seed=int(base_seed),
                fault_mode=fault_mode,
                use_llm_agents=False,
                manage_trust_scores=False,
            )
            env.reset()
            try:
                decision = OrchestratorDecision.model_validate_json(completion)
                _obs, reward, _done, _info = env.step(decision.model_dump())
            except Exception:
                _obs, reward, _done, _info = env.step("invalid completion")
            rewards.append(float(reward))
        return rewards

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=GRPOConfig(
            output_dir=config.grpo_output_dir,
            per_device_train_batch_size=config.grpo_per_device_train_batch_size,
            gradient_accumulation_steps=config.grpo_gradient_accumulation_steps,
            num_generations=config.grpo_num_generations,
            max_steps=config.grpo_max_steps,
            logging_steps=1,
            report_to=[],
        ),
        train_dataset=dataset,
    )
    trainer.train()

    output_dir = Path(config.grpo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "grpo_metadata.json", "w") as f:
        json.dump({"dataset": str(dataset_path), **backend_info}, f, indent=2)
    return output_dir


if __name__ == "__main__":
    print(run_grpo())
