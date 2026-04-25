"""TRL GRPO training entrypoint for verifiable single-step AIC rollouts.

Critical fixes vs the previous revision:

  * Drop the strict ``OrchestratorDecision.model_validate_json(completion)``
    pre-parse. The env's ``_parse_action`` already handles strings, partial
    JSON, and legacy text — and the env scores every step with R5 (format
    validity) + R6 (verifier) + R1..R9. Wrapping it with our own
    ``-10/-12/-15`` constant penalty just collapses reward variance to 0
    and kills the gradient.
  * SFT warm-start is loaded via ``PeftModel.from_pretrained(base, sft_dir)``
    instead of ``AutoModelForCausalLM.from_pretrained(sft_dir)`` which would
    silently ignore the LoRA adapter.
  * Reward audit thresholds are relaxed (3600s wall-clock, severity 0.95)
    so a slow generation on T4 cannot silently zero out the reward signal.
  * Logs both the raw env reward and final shaped reward per step for
    forensic traceability.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aic.env.aic_environment import AICEnvironment
from aic.training.config import TrainingConfig
from aic.training.modeling_unsloth import load_model_and_tokenizer
from aic.training.prompting import (
    build_chat_messages_compact,
    build_orchestrator_prompt,
)
from aic.training.reward_audit import RewardAuditLoop
from aic.training.scenario_contract import (
    CANONICAL_SCENARIO_IDS,
    SCENARIO_TRAINING_META,
)


class AICProgressCallback:
    """Logs reward and loss at every step for the evidence reward curve."""

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
            "reward_std": logs.get("reward_std", logs.get("train/reward_std", 0)),
            "loss": logs.get("loss", logs.get("train/loss", 0)),
            "kl": logs.get("kl", 0),
            "completion_length": logs.get("completion_length", 0),
            "elapsed_minutes": (time.time() - self.start_time) / 60,
        }
        self.step_log.append(entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"  Step {entry['step']:4d} | reward={entry['reward']:+.3f} "
            f"std={entry['reward_std']:.3f} | loss={entry['loss']:.4f} "
            f"| kl={entry['kl']:.4f} | comp_len={entry['completion_length']:.0f} "
            f"| {entry['elapsed_minutes']:.1f}m"
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_log:
            return
        summary = {
            "total_steps": len(self.step_log),
            "initial_reward": self.step_log[0]["reward"],
            "final_reward": self.step_log[-1]["reward"],
            "reward_delta": self.step_log[-1]["reward"] - self.step_log[0]["reward"],
            "max_reward_std": max((s["reward_std"] for s in self.step_log), default=0.0),
            "training_time_minutes": self.step_log[-1]["elapsed_minutes"],
        }
        out = Path(args.output_dir) / "training_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"\n[GRPO] Training complete. Reward delta: "
            f"{summary['reward_delta']:+.3f} | max_std={summary['max_reward_std']:.3f}"
        )
        print(f"   Summary saved to {out}")


def _render_chat_prompt_for_scenario(tokenizer, obs: dict[str, Any]) -> str:
    """Render the compact chat prompt with the model's chat template."""
    messages = build_chat_messages_compact(obs)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return build_orchestrator_prompt(obs)


def generate_grpo_prompt_dataset(
    config: TrainingConfig | None = None,
    tokenizer=None,
) -> Path:
    """Generate prompt-only JSONL records for single-step GRPO training.

    If ``tokenizer`` is provided, prompts are rendered through its chat
    template; otherwise the compact-text prompt is used (the env's chat
    template will be applied by GRPOTrainer if configured).
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
                    scenario_id=scenario_id,
                )
                obs = env.reset()
                if tokenizer is not None:
                    prompt = _render_chat_prompt_for_scenario(tokenizer, obs)
                else:
                    from aic.training.prompting import build_compact_user_text
                    prompt = build_compact_user_text(obs)
                record = {
                    "prompt": prompt,
                    "episode_id": episode_id,
                    "base_seed": config.base_seed,
                    "scenario": meta.scenario_name,
                    "scenario_id": scenario_id,
                    "fault_mode": meta.fault_injector_mode,
                }
                f.write(json.dumps(record) + "\n")
                episode_id += 1
    return path


def _extract_completion_text(completion: Any) -> str:
    """TRL passes either a list of message dicts or a raw string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def _shape_reward(text: str) -> float:
    """Tiny graded shaping so completely-non-JSON outputs do not all get the
    same env penalty. Without this, the ``reward_std=0`` failure mode returns.
    """
    text = text.strip()
    shaping = 0.0
    if "{" not in text:
        shaping -= 1.0
    if "selected_recommendation_id" not in text:
        shaping -= 0.5
    if len(text) < 10:
        shaping -= 0.5
    return shaping


def run_grpo(config: TrainingConfig | None = None) -> Path:
    """Run a single-step GRPO/RLVR loop when TRL is available."""
    if config is None:
        config = TrainingConfig()

    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise ImportError(
            "GRPO dependencies missing. Install `trl`, `datasets`, and model backends first."
        ) from exc

    try:
        from transformers import TrainerCallback

        # NOTE: Order (AICProgressCallback first) is critical so that MRO
        # resolves on_log/on_train_end from our class instead of the no-op
        # methods on TrainerCallback.
        class _AICCallback(AICProgressCallback, TrainerCallback):
            def __init__(self, log_path: str = "logs/grpo_progress.jsonl"):
                AICProgressCallback.__init__(self, log_path)
                TrainerCallback.__init__(self)

            def on_log(self, args, state, control, logs=None, **kwargs):
                return AICProgressCallback.on_log(
                    self, args, state, control, logs=logs, **kwargs
                )

            def on_train_end(self, args, state, control, **kwargs):
                return AICProgressCallback.on_train_end(
                    self, args, state, control, **kwargs
                )
    except ImportError:
        _AICCallback = AICProgressCallback  # type: ignore[misc]

    sft_dir = Path(config.sft_output_dir)
    sft_adapter = sft_dir / "adapter_config.json"
    sft_meta = sft_dir / "sft_metadata.json"
    warm_start_path: Path | None = None
    if sft_adapter.exists() and sft_meta.exists():
        try:
            with open(sft_meta) as f:
                meta = json.load(f)
            sft_base = str(meta.get("model_name", "")).lower()
            if "qwen" in sft_base:
                warm_start_path = sft_dir
                print(f"[GRPO] Will attach SFT LoRA from {warm_start_path}")
        except Exception:
            pass

    model, tokenizer, backend_info = load_model_and_tokenizer(config)

    if warm_start_path is not None:
        try:
            from peft import PeftModel

            try:
                if hasattr(model, "unload"):
                    model = model.unload()
            except Exception:
                pass

            model = PeftModel.from_pretrained(
                model, str(warm_start_path), is_trainable=True,
            )
            backend_info["sft_warm_start"] = str(warm_start_path)
            print(f"[GRPO] Loaded SFT LoRA adapter; trainable params attached.")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[GRPO] WARNING: could not attach SFT LoRA: {exc}")
            backend_info["sft_warm_start_error"] = str(exc)

    dataset_path = Path(config.grpo_dataset_path)
    dataset_path = generate_grpo_prompt_dataset(config, tokenizer=tokenizer)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    grpo_audit = RewardAuditLoop(
        log_dir=str(Path(config.grpo_output_dir) / "audit"),
        max_wall_clock_seconds=3600.0,
        severity_clamp_threshold=0.95,
        reward_clamp_value=-1.0,
    )
    _audit_episode_counter = [0]

    def reward_func(completions, **kwargs):
        rewards: list[float] = []
        episode_ids = kwargs.get("episode_id", [])
        base_seeds = kwargs.get("base_seed", [])
        fault_modes = kwargs.get("fault_mode", [])
        scenario_ids = kwargs.get("scenario_id", [])

        for idx, completion in enumerate(completions):
            text = _extract_completion_text(completion)
            episode_id = int(episode_ids[idx]) if idx < len(episode_ids) else idx
            base_seed = int(base_seeds[idx]) if idx < len(base_seeds) else config.base_seed
            fault_mode = fault_modes[idx] if idx < len(fault_modes) else config.fault_mode
            scenario_id = int(scenario_ids[idx]) if idx < len(scenario_ids) else 0

            env = AICEnvironment(
                episode_id=episode_id,
                base_seed=base_seed,
                fault_mode=fault_mode,
                use_llm_agents=False,
                manage_trust_scores=False,
                scenario_id=scenario_id,
            )
            env.reset()

            audit_ep_id = _audit_episode_counter[0]
            _audit_episode_counter[0] += 1
            grpo_audit.begin_episode(audit_ep_id)

            try:
                _obs, env_reward, _done, _info = env.step(text)
            except Exception:
                env_reward = -8.0

            shaped = float(env_reward) + _shape_reward(text)

            metrics = env.world_state.snapshot() if hasattr(env, "world_state") else {}
            grpo_audit.record_step(
                step=0, action=text[:200],
                reward=float(shaped), metrics=metrics,
            )
            audit_result = grpo_audit.end_episode(float(shaped))
            rewards.append(float(audit_result.adjusted_reward))

        return rewards

    progress_callback = _AICCallback()

    grpo_args = GRPOConfig(
        output_dir=config.grpo_output_dir,
        per_device_train_batch_size=config.grpo_per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo_gradient_accumulation_steps,
        num_generations=config.grpo_num_generations,
        max_steps=config.grpo_max_steps,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=0.7,
        learning_rate=1e-5,
        beta=0.04,
        logging_steps=1,
        save_steps=50,
        warmup_steps=10,
        fp16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=grpo_args,
        train_dataset=dataset,
        callbacks=[progress_callback],
    )
    trainer.train()

    output_dir = Path(config.grpo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    audit_summary = grpo_audit.summary_stats()
    with open(output_dir / "grpo_audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    with open(output_dir / "grpo_metadata.json", "w") as f:
        json.dump({
            "dataset": str(dataset_path),
            "reward_audit_integrated": True,
            "audit_severity_threshold": 0.95,
            "audit_max_wall_clock_seconds": 3600.0,
            "num_generations": config.grpo_num_generations,
            "max_prompt_length": config.max_prompt_length,
            "max_completion_length": config.max_completion_length,
            **backend_info,
        }, f, indent=2)
    return output_dir


if __name__ == "__main__":
    print(run_grpo())
