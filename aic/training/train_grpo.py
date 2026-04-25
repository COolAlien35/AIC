"""TRL GRPO training entrypoint for verifiable single-step AIC rollouts."""
from __future__ import annotations

import json
import time
from pathlib import Path

from aic.env.aic_environment import AICEnvironment
from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.modeling_unsloth import load_model_and_tokenizer
from aic.training.prompting import build_orchestrator_prompt
from aic.training.scenario_contract import (
    CANONICAL_SCENARIO_IDS,
    SCENARIO_TRAINING_META,
)


class AICProgressCallback:
    """Logs reward and loss at every step. Creates the reward curve we need for evidence.

    Compatible with both transformers TrainerCallback and standalone usage.
    """

    def __init__(self, log_path: str = "logs/grpo_progress.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.step_log: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {
            "step": state.global_step,
            "reward": logs.get("reward", logs.get("train/reward", 0)),
            "loss": logs.get("loss", logs.get("train/loss", 0)),
            "elapsed_minutes": (time.time() - self.start_time) / 60,
        }
        self.step_log.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Console progress
        print(f"  Step {entry['step']:4d} | reward={entry['reward']:+.4f} | "
              f"loss={entry['loss']:.4f} | elapsed={entry['elapsed_minutes']:.1f}m")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_log:
            return
        summary = {
            "total_steps": len(self.step_log),
            "initial_reward": self.step_log[0]["reward"],
            "final_reward": self.step_log[-1]["reward"],
            "reward_delta": self.step_log[-1]["reward"] - self.step_log[0]["reward"],
            "training_time_minutes": self.step_log[-1]["elapsed_minutes"],
        }
        out = Path(args.output_dir) / "training_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n🏁 Training complete. Reward delta: {summary['reward_delta']:+.4f}")
        print(f"   Summary saved to {out}")


def generate_grpo_prompt_dataset(config: TrainingConfig | None = None) -> Path:
    """Generate prompt-only JSONL records for single-step GRPO training.

    Uses the canonical 6-scenario registry (same as SFT) for distribution parity.
    """
    if config is None:
        config = TrainingConfig()

    path = Path(config.grpo_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_scenarios = len(CANONICAL_SCENARIO_IDS)
    episodes_per_scenario = max(1, config.sft_num_episodes // num_scenarios)

    with open(path, "w") as f:
        episode_id = 0
        for scenario_id in CANONICAL_SCENARIO_IDS:
            meta = SCENARIO_TRAINING_META[scenario_id]
            for _ in range(episodes_per_scenario):
                env = AICEnvironment(
                    episode_id=episode_id,
                    base_seed=config.base_seed,
                    fault_mode=meta.fault_injector_mode,
                    use_llm_agents=False,
                    manage_trust_scores=False,
                )
                obs = env.reset()
                record = {
                    "prompt": build_orchestrator_prompt(obs),
                    "episode_id": episode_id,
                    "base_seed": config.base_seed,
                    "scenario": meta.scenario_name,
                    "scenario_id": scenario_id,
                    "fault_mode": meta.fault_injector_mode,
                }
                f.write(json.dumps(record) + "\n")
                episode_id += 1
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

    # Try to import TrainerCallback for proper callback integration
    try:
        from transformers import TrainerCallback

        class _AICCallback(TrainerCallback, AICProgressCallback):
            def __init__(self, log_path: str = "logs/grpo_progress.jsonl"):
                TrainerCallback.__init__(self)
                AICProgressCallback.__init__(self, log_path)
    except ImportError:
        _AICCallback = AICProgressCallback  # type: ignore[misc]

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

    progress_callback = _AICCallback()

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
            logging_steps=5,
            save_steps=25,
            warmup_steps=10,
            report_to=[],
        ),
        train_dataset=dataset,
        callbacks=[progress_callback],
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
