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
import os
import random
from pathlib import Path
from typing import Any

from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.ddp_utils import is_main_process, local_rank as _ddp_local_rank, wait_for_everyone
from aic.training.modeling_unsloth import (
    QWEN_TARGET_MODULES,
    _build_bnb_config,
    _preferred_compute_dtype,
)


def _write_prompt_completion_jsonl(dataset_path: Path) -> Path:
    """Write a temporary JSONL containing only ``prompt`` and ``completion``.

    The SFT generator emits rich metadata (scenario, drift, difficulty, optional
    negative-sample fields). Mixing that into ``datasets.load_dataset("json",
    ...)`` triggers schema-cast failures because positive and negative rows
    expose different ``metadata`` substructures. Since SFT only uses the two
    text fields, we strip everything else into a sibling file and load that.
    """
    cleaned_path = dataset_path.parent / "_prompt_completion_only.jsonl"
    with dataset_path.open() as src, cleaned_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dst.write(json.dumps({
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("completion", "")),
            }) + "\n")
    return cleaned_path


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
        prompt_text, _ = _build_chat_text(tokenizer, example["prompt"], "")
        completion_text = str(example["completion"]) + tokenizer.eos_token

        # Tokenize prompt and completion separately so long prompts cannot
        # consume the completion budget and silently zero out all labels.
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=config.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completion_text,
            truncation=True,
            max_length=config.max_completion_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + list(completion_ids)

        # Safety: if completion got fully truncated, keep at least EOS as a
        # supervised target to avoid all-masked batches and zero loss/grad.
        if not completion_ids and tokenizer.eos_token_id is not None:
            input_ids = prompt_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(prompt_ids)) + [tokenizer.eos_token_id]

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

    if is_main_process():
        _write_prompt_completion_jsonl(dataset_path)
    wait_for_everyone()
    cleaned_train_path = dataset_path.parent / "_prompt_completion_only.jsonl"
    raw_dataset = load_dataset(
        "json",
        data_files=str(cleaned_train_path),
        split="train",
    )

    val_path = dataset_path.parent / "val.jsonl"
    eval_dataset = None
    disable_eval = os.environ.get("AIC_SFT_DISABLE_EVAL", "").lower() in (
        "1", "true", "yes",
    )
    if val_path.exists() and not disable_eval:
        if is_main_process():
            _write_prompt_completion_jsonl(val_path)
        wait_for_everyone()
        cleaned_val_path = val_path.parent / "_prompt_completion_only.jsonl"
        eval_dataset = load_dataset(
            "json",
            data_files=str(cleaned_val_path),
            split="train",
        )
    elif val_path.exists() and disable_eval:
        print("[SFT] AIC_SFT_DISABLE_EVAL set - skipping eval split")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    # device_map={"":0} pins all layers to a single GPU. "auto" can split a
    # 4-bit model across CPU/GPU which then breaks TRL's reference-model
    # preparation in the GRPO step that consumes this checkpoint.
    load_kwargs: dict[str, Any] = {
        "device_map": {"": _ddp_local_rank()} if has_cuda else None,
        "low_cpu_mem_usage": True,
    }
    bnb = _build_bnb_config(config.load_in_4bit and has_cuda)
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    elif has_cuda:
        load_kwargs["torch_dtype"] = _preferred_compute_dtype() or torch.float16

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

    max_steps_env = os.environ.get("AIC_SFT_MAX_STEPS", "").strip()
    max_steps_override: int | None = None
    if max_steps_env:
        try:
            max_steps_override = int(max_steps_env)
            print(f"[SFT] AIC_SFT_MAX_STEPS={max_steps_override} - capping training")
        except ValueError:
            print(f"[SFT] WARNING: AIC_SFT_MAX_STEPS={max_steps_env!r} is not int - ignored")

    ta_kwargs: dict[str, Any] = dict(
        output_dir=config.sft_output_dir,
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
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=has_cuda,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_total_limit=1,
        remove_unused_columns=False,
        optim="adamw_8bit" if has_cuda else "adamw_torch",
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
    )
    if max_steps_override is not None:
        ta_kwargs["max_steps"] = max_steps_override
    else:
        ta_kwargs["num_train_epochs"] = config.sft_epochs

    args = TrainingArguments(**ta_kwargs)

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

    final_loss: float | None = None
    try:
        if train_result is not None and train_result.metrics:
            final_loss = float(train_result.metrics.get("train_loss"))
    except Exception:
        final_loss = None

    parse_rate = 0.0
    parse_error: str | None = None
    if is_main_process():
        try:
            parse_rate = _smoke_parse_rate(
                model, tokenizer, raw_dataset, sample_n=8,
            )
        except Exception as exc:
            parse_error = str(exc)
        _pr_file = output_dir / "_sft_parse_rate_sync.json"
        with open(_pr_file, "w") as _f:
            json.dump({"parse_rate": parse_rate, "error": parse_error}, _f)
    wait_for_everyone()
    if not is_main_process():
        _pr_file = output_dir / "_sft_parse_rate_sync.json"
        try:
            with open(_pr_file) as _f:
                _sync = json.load(_f)
            parse_rate = float(_sync.get("parse_rate", 0.0))
            e = _sync.get("error")
            parse_error = str(e) if e else None
        except Exception:
            pass
    try:
        if is_main_process() and (output_dir / "_sft_parse_rate_sync.json").exists():
            (output_dir / "_sft_parse_rate_sync.json").unlink()
    except Exception:
        pass
    wait_for_everyone()

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
    if is_main_process():
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        with open(output_dir / "sft_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    wait_for_everyone()

    def _cleanup_after_train() -> None:
        try:
            del trainer
            del model
        except Exception:
            pass
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    if parse_rate < min_parse_rate:
        _cleanup_after_train()
        raise RuntimeError(
            f"SFT parse-rate gate failed: got {parse_rate:.2f} < required "
            f"{min_parse_rate:.2f}. Inspect {output_dir}/sft_metadata.json. "
            "Common causes: dataset too small, prompt template mismatch, or "
            "max_completion_length too low for the JSON schema."
        )

    _cleanup_after_train()
    return output_dir


if __name__ == "__main__":
    print(run_sft())
