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
        text = example["prompt"] + "\n" + example["completion"]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=config.max_prompt_length + config.max_completion_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(_tokenize, remove_columns=dataset.column_names)
    args = TrainingArguments(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        learning_rate=config.sft_learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    output_dir = Path(config.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta_path = output_dir / "sft_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"dataset": str(dataset_path), "model_name": config.model_name}, f, indent=2)

    return output_dir


if __name__ == "__main__":
    print(run_sft())
