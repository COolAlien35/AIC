"""Supervised warm-start training for the orchestrator policy.

Critical fixes vs the previous revision:

  * Uses Qwen's native chat template via ``apply_chat_template`` so the
    model sees the same ChatML wrapping it was instruct-tuned on.
  * LoRA is mandatory (rank 16, alpha 32) on all 7 Qwen projection layers.
  * Padding side is forced to "left" so the labels mask matches the actual
    completion span when batched.
  * Writes a CORRECT ``sft_metadata.json`` containing the real base model
    name (e.g. ``Qwen/Qwen2.5-3B-Instruct``) so the GRPO warm-start gate
    actually triggers.
  * Runs a parse-rate smoke generation at the end of training and
    raises if it falls below ``min_parse_rate`` (default 0.7).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.modeling_unsloth import (
    QWEN_TARGET_MODULES,
    _build_bnb_config,
)


def _require_sft_dependencies():
    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "SFT dependencies missing. Install `datasets`, `transformers`, `peft`."
        ) from exc

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise ImportError("`peft` is required for SFT.") from exc

    return (
        load_dataset, AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments,
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    )


def _build_chat_text(tokenizer, prompt_text: str, completion_text: str) -> tuple[str, int]:
    """Render a (prompt, completion) pair through the chat template.

    Returns the full text and the token count of just the prompt portion
    (so we can mask labels for it).
    """
    user_msg = [
        {"role": "system", "content": (
            "You are the Adaptive Incident Choreographer orchestrator. "
            "Output strict JSON only."
        )},
        {"role": "user", "content": prompt_text},
    ]
    prompt_only = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full_text = prompt_only + completion_text + tokenizer.eos_token
    return full_text, len(prompt_ids)


def _make_tokenize_fn(tokenizer, config: TrainingConfig):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        full_text, prompt_len = _build_chat_text(
            tokenizer, example["prompt"], example["completion"],
        )
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=config.max_prompt_length + config.max_completion_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _tokenize


def _smoke_parse_rate(model, tokenizer, dataset, sample_n: int = 8) -> float:
    """Generate completions for ``sample_n`` random prompts and report parse rate."""
    try:
        import torch
    except Exception:
        return 0.0
    if len(dataset) == 0:
        return 0.0
    n = min(sample_n, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    model.eval()
    successes = 0
    device = next(model.parameters()).device
    for idx in indices:
        prompt_text = dataset[idx]["prompt"]
        rendered, _ = _build_chat_text(tokenizer, prompt_text, "")
        inputs = tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=1024,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=192,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        try:
            json_start = completion.find("{")
            json_end = completion.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                OrchestratorDecision.model_validate_json(completion[json_start:json_end])
                successes += 1
        except Exception:
            continue
    model.train()
    return successes / max(1, n)


def run_sft(config: TrainingConfig | None = None, min_parse_rate: float = 0.7) -> Path:
    """Run instruction-style SFT on prompt/completion JSONL data."""
    if config is None:
        config = TrainingConfig()

    (
        load_dataset, AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments,
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    ) = _require_sft_dependencies()

    dataset_path = Path(config.sft_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Run generate_sft_data.py first."
        )

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    val_path = dataset_path.parent / "val.jsonl"
    eval_dataset = None
    if val_path.exists():
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    load_kwargs: dict[str, Any] = {"device_map": "auto" if has_cuda else None}
    bnb = _build_bnb_config(config.load_in_4bit and has_cuda)
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    elif has_cuda:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if config.load_in_4bit and has_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)

    tokenize_fn = _make_tokenize_fn(tokenizer, config)
    tokenized_train = raw_dataset.map(
        tokenize_fn, remove_columns=raw_dataset.column_names,
    )
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_fn, remove_columns=eval_dataset.column_names,
        )

    args = TrainingArguments(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accumulation,
        learning_rate=config.sft_learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=50 if tokenized_eval is not None else None,
        report_to=[],
        no_cuda=not has_cuda,
        use_mps_device=False,
        fp16=has_cuda,
        gradient_checkpointing=has_cuda,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_total_limit=1,
        remove_unused_columns=False,
        optim="adamw_8bit" if has_cuda else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        ),
    )

    train_result = trainer.train()

    output_dir = Path(config.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    final_loss: float | None = None
    try:
        if train_result is not None and train_result.metrics:
            final_loss = float(train_result.metrics.get("train_loss"))
    except Exception:
        final_loss = None

    parse_rate = 0.0
    parse_error: str | None = None
    try:
        parse_rate = _smoke_parse_rate(model, tokenizer, raw_dataset, sample_n=8)
    except Exception as exc:
        parse_error = str(exc)

    metadata = {
        "dataset": str(dataset_path),
        "dataset_size": len(raw_dataset),
        "model_name": config.model_name,
        "final_loss": final_loss,
        "used_cuda": has_cuda,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": QWEN_TARGET_MODULES,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "parse_rate": parse_rate,
        "parse_smoke_n": 8,
        "parse_smoke_error": parse_error,
    }
    with open(output_dir / "sft_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if parse_rate < min_parse_rate:
        raise RuntimeError(
            f"SFT parse-rate gate failed: got {parse_rate:.2f} < required "
            f"{min_parse_rate:.2f}. Inspect {output_dir}/sft_metadata.json. "
            "Common causes: dataset too small, prompt template mismatch, or "
            "max_completion_length too low for the JSON schema."
        )

    return output_dir


if __name__ == "__main__":
    print(run_sft())
