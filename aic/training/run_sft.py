"""Supervised warm-start training for the orchestrator policy."""
from __future__ import annotations

import json
from pathlib import Path

from aic.training.config import TrainingConfig


def _require_sft_dependencies() -> tuple:
    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "SFT dependencies missing. Install `datasets`, `transformers`, and optionally `peft`."
        ) from exc

    try:  # pragma: no cover - optional
        from peft import LoraConfig, get_peft_model
    except Exception:
        LoraConfig = None
        get_peft_model = None

    return load_dataset, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, LoraConfig, get_peft_model


def run_sft(config: TrainingConfig | None = None) -> Path:
    """Run instruction-style SFT on prompt/completion JSONL data."""
    if config is None:
        config = TrainingConfig()

    (
        load_dataset,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        LoraConfig,
        get_peft_model,
    ) = _require_sft_dependencies()

    dataset_path = Path(config.sft_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Run generate_sft_data.py first."
        )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Load validation set if available
    val_path = Path(config.sft_dataset_path).parent / "val.jsonl"
    eval_dataset = None
    if val_path.exists():
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if config.use_peft_for_sft and get_peft_model is not None and LoraConfig is not None:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    def _tokenize(example):
        # Tokenize prompt and completion separately to compute mask boundary
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=config.max_prompt_length,
            add_special_tokens=True,
        )["input_ids"]

        completion_ids = tokenizer(
            example["completion"],
            truncation=True,
            max_length=config.max_completion_length,
            add_special_tokens=False,
        )["input_ids"]

        # Concatenate with newline separator token
        sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + sep_ids + completion_ids

        # Truncate to max total length
        max_total = config.max_prompt_length + config.max_completion_length
        input_ids = input_ids[:max_total]

        # Mask prompt tokens in labels (set to -100 so loss ignores them)
        prompt_len = len(prompt_ids) + len(sep_ids)
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized_dataset = dataset.map(_tokenize, remove_columns=dataset.column_names)
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(_tokenize, remove_columns=eval_dataset.column_names)

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    args = TrainingArguments(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accumulation,
        learning_rate=config.sft_learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=50 if tokenized_eval is not None else None,
        report_to=[],
        no_cuda=not has_cuda,
        use_mps_device=False,
        fp16=has_cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_result = trainer.train()

    output_dir = Path(config.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta_path = output_dir / "sft_metadata.json"
    final_loss = None
    try:
        if train_result is not None and train_result.metrics:
            final_loss = train_result.metrics.get("train_loss")
    except Exception:
        final_loss = None

    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": str(dataset_path),
                "dataset_size": len(dataset),
                "model_name": config.model_name,
                "final_loss": final_loss,
                "used_cuda": has_cuda,
            },
            f,
            indent=2,
        )

    return output_dir


if __name__ == "__main__":
    print(run_sft())
