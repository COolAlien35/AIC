#!/usr/bin/env python3
"""Run minimal SFT training (3 steps) to produce a real LoRA checkpoint."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import (AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2-0.5B-Instruct"
OUT = "checkpoints/sft"
DATA = "artifacts/sft/orchestrator_sft.jsonl"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"))
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("Loading data...")
ds = load_dataset("json", data_files=DATA, split="train")
def tok(ex):
    t = tokenizer(ex["prompt"][:300] + "\n" + ex["completion"][:200], truncation=True, max_length=256)
    t["labels"] = t["input_ids"].copy()
    return t
tds = ds.map(tok, remove_columns=ds.column_names)

print("Training 3 steps...")
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT, max_steps=3, per_device_train_batch_size=2,
        learning_rate=2e-5, logging_steps=1, save_steps=3,
        save_strategy="steps", report_to=[], dataloader_pin_memory=False,
    ),
    train_dataset=tds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
result = trainer.train()
print(f"Loss: {result.training_loss:.4f}")

Path(OUT).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)
with open(f"{OUT}/sft_metadata.json", "w") as f:
    json.dump({"model": MODEL, "steps": 3, "loss": result.training_loss, "dataset": DATA}, f)
print(f"✅ SFT checkpoint saved to {OUT}")
