"""Model loading helpers with optional Unsloth acceleration."""
from __future__ import annotations

from aic.training.config import TrainingConfig


def load_model_and_tokenizer(config: TrainingConfig, max_seq_length: int | None = None):
    """Load model/tokenizer with Unsloth when available, else fall back."""
    max_seq_length = max_seq_length or (config.max_prompt_length + config.max_completion_length)

    if config.use_unsloth:
        try:  # pragma: no cover - optional dependency
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            return model, tokenizer, {"backend": "unsloth"}
        except Exception:
            pass

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Transformers is required to load the policy model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    return model, tokenizer, {"backend": "transformers"}
