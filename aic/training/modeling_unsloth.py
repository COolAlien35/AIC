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
from aic.training.ddp_utils import local_rank as _ddp_local_rank

QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _preferred_compute_dtype():
    """Return torch.bfloat16 when supported on the active device, else float16.

    T4 supports bf16 via emulation; bf16 is more numerically stable than fp16
    for LoRA + GRPO and avoids the loss-spike / loss=0.0 collapse patterns
    we have observed with fp16. Falls back to fp16 on hardware that has no
    bf16 path at all (very old GPUs, or CPU-only smoke).
    """
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    except Exception:
        return None


def _build_bnb_config(load_in_4bit: bool):
    """Construct a BitsAndBytesConfig with bf16 compute when supported."""
    if not load_in_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        return None
    compute_dtype = _preferred_compute_dtype()
    if compute_dtype is None:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
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

    # device_map={"":0} pins ALL layers to cuda:0. "auto" can split a 4-bit
    # model across CPU/GPU which then breaks TRL's reference-model preparation
    # with: "You can't train a model loaded in 4-bit on a different device".
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
        torch = None  # type: ignore[assignment]

    load_kwargs: dict[str, Any] = {
        "device_map": {"": _ddp_local_rank()} if has_cuda else None,
        "low_cpu_mem_usage": True,
    }
    bnb = _build_bnb_config(config.load_in_4bit and has_cuda)
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    elif has_cuda and torch is not None:
        load_kwargs["torch_dtype"] = _preferred_compute_dtype() or torch.float16

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

    compute_dtype = _preferred_compute_dtype()
    dtype_label = "bf16" if (
        compute_dtype is not None and str(compute_dtype).endswith("bfloat16")
    ) else "fp16"
    return model, tokenizer, {
        "model_name": model_name,
        "backend": backend,
        "max_seq_length": max_seq_length,
        "quantization": (
            f"nf4-{dtype_label}-doublequant" if bnb is not None else dtype_label
        ),
    }
