"""Model loading helpers with optional Unsloth acceleration.

Always returns a tuple ``(model, tokenizer, info)`` where ``info`` contains the
backend used and any quantization details. Critical fixes vs the previous
revision:

  * Pass ``BitsAndBytesConfig`` via ``quantization_config`` instead of the
    deprecated ``load_in_4bit`` keyword (which transformers 4.46+ ignores with
    a warning and silently runs in fp32).
  * Set ``tokenizer.padding_side = "left"`` so batched generation (used by
    GRPO) does not blend pad tokens into the produced sequence.
  * Disable KV cache when gradient checkpointing is on (avoids OOM and a
    transformers warning).
  * Apply LoRA to all 7 Qwen projection layers, not just q/v, in both
    Unsloth and transformers paths.
  * Run ``prepare_model_for_kbit_training`` so layer norms stay in fp32 and
    gradient checkpointing actually works on 4-bit weights.
"""
from __future__ import annotations

from typing import Any

from aic.training.config import TrainingConfig

QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _build_bnb_config(load_in_4bit: bool):
    """Construct a BitsAndBytesConfig for fp16 compute (T4 has no bf16)."""
    if not load_in_4bit:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except Exception:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(
    config: TrainingConfig,
    max_seq_length: int | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load model/tokenizer with Unsloth when available, else transformers+peft."""
    model_name = config.model_name
    max_seq_length = max_seq_length or (
        config.max_prompt_length + config.max_completion_length
    )

    if config.use_unsloth:
        try:  # pragma: no cover - optional dependency
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=config.load_in_4bit,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=QWEN_TARGET_MODULES,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            try:
                model.config.use_cache = False
            except Exception:
                pass
            return model, tokenizer, {
                "model_name": model_name,
                "backend": "unsloth",
                "max_seq_length": max_seq_length,
            }
        except Exception:
            pass  # Fall through to transformers backend

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Transformers is required to load the policy model.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {"device_map": "auto"}
    bnb = _build_bnb_config(config.load_in_4bit)
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    else:
        try:
            import torch

            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if config.load_in_4bit:
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True,
            )
        except Exception:
            pass

    backend = "transformers"
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=QWEN_TARGET_MODULES,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        backend = "transformers+peft"
    except ImportError:
        pass

    return model, tokenizer, {
        "model_name": model_name,
        "backend": backend,
        "max_seq_length": max_seq_length,
        "quantization": "nf4-fp16-doublequant" if bnb is not None else "fp16",
    }
