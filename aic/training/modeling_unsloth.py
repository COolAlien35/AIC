"""Model loading helpers with optional Unsloth acceleration."""
from __future__ import annotations

from aic.training.config import TrainingConfig


def load_model_and_tokenizer(config: TrainingConfig, max_seq_length: int | None = None):
    """Load model/tokenizer with Unsloth when available, else fall back to transformers+peft."""
    model_name = config.model_name  # Always use config — never hardcode
    max_seq_length = max_seq_length or (config.max_prompt_length + config.max_completion_length)

    if config.use_unsloth:
        try:  # pragma: no cover - optional dependency
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=config.load_in_4bit,
            )
            # Apply LoRA for efficient fine-tuning
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            return model, tokenizer, {"model_name": model_name, "backend": "unsloth"}
        except Exception:
            pass  # Fall through to standard transformers

    # Standard transformers fallback (slower but works everywhere)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Transformers is required to load the policy model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto"} if config.load_in_4bit else {}
    if config.load_in_4bit:
        load_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Apply LoRA via peft if available
    try:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    except ImportError:
        pass  # Train full model if peft not available

    return model, tokenizer, {"model_name": model_name, "backend": "transformers"}
