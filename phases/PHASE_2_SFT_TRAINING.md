# 🟠 PHASE 2 — SFT TRAINING
> **Time: ~2 hours GPU | Checkpoint: `checkpoints/sft/`**  
> Prerequisite: Phase 1 complete, GPU confirmed, code mounted.

---

## CELL 5 — Generate SFT Data + Run Training

```python
import subprocess, sys, json

# Step 1: Generate SFT data (if not already uploaded with the zip)
result = subprocess.run(
    [sys.executable, "aic/training/generate_sft_data.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    raise RuntimeError("SFT data generation failed — check Phase 0 Fix 0.2")

# Step 2: Verify data quality
data = [json.loads(l) for l in open('artifacts/sft/orchestrator_sft.jsonl')]
print(f"✅ SFT examples: {len(data)}")
assert len(data) >= 500, f"Not enough data: {len(data)}"

scenarios = set(d['scenario'] for d in data)
assert len(scenarios) == 6, f"Missing scenarios: {scenarios}"
print(f"✅ All 6 scenarios covered: {scenarios}")

# Step 3: Run SFT training
result = subprocess.run(
    [sys.executable, "aic/training/run_sft.py"],
    capture_output=False  # Stream output live so you can monitor
)
assert result.returncode == 0, "SFT training FAILED — check logs"
```

---

## CELL 6 — Verify SFT Checkpoint

```python
import json
from pathlib import Path

assert Path("checkpoints/sft").exists(), "❌ No SFT checkpoint found!"

meta = json.loads(Path("checkpoints/sft/sft_metadata.json").read_text())

# Hard stop — wrong model means wasted GPU time
assert "tiny-gpt2" not in meta.get("model_name", ""), \
    "WRONG MODEL IN CHECKPOINT — Phase 0 Fix 0.1 was not applied"
assert "Qwen" in meta.get("model_name", "") or "Llama" in meta.get("model_name", ""), \
    f"Unexpected model name: {meta.get('model_name')}"

print(f"✅ SFT Checkpoint validated")
print(f"   Model:          {meta['model_name']}")
print(f"   Dataset size:   {meta.get('dataset_size', 'N/A')}")
print(f"   Training loss:  {meta.get('final_loss', 'N/A')}")
```

---

## OOM Recovery (If Out of Memory)

If you see a CUDA OOM error during SFT, run this config patch before retrying:

```python
from aic.training.config import TrainingConfig

config = TrainingConfig()
config.max_prompt_length = 512      # Halve sequence length
config.sft_batch_size = 1           # Minimum batch
config.sft_grad_accumulation = 16   # Compensate with gradient accumulation

# Then rerun:
# python aic/training/run_sft.py --config config
```

---

## What Good SFT Looks Like

- Loss should start high (~3.5–5.0) and trend downward
- Final loss around 0.8–1.5 is healthy
- If loss is NaN → lower learning rate by 10x
- If loss doesn't move → check that data is loading correctly

---

## ✅ Phase 2 Completion Criteria

- [ ] `artifacts/sft/orchestrator_sft.jsonl` exists with 500+ records
- [ ] `checkpoints/sft/` directory exists
- [ ] `checkpoints/sft/sft_metadata.json` contains `Qwen` (not `tiny-gpt2`)
- [ ] Training loss visible and trending downward during run

**→ Next: [PHASE_3_GRPO_TRAINING.md](PHASE_3_GRPO_TRAINING.md)**
