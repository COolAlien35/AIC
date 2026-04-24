# 🔵 PHASE 3 — GRPO TRAINING (The Main Event)
> **Time: 5–7 hours GPU | Checkpoint: `checkpoints/grpo/`**  
> This is the phase that proves RL works. Do NOT rush this.

---

## CELL 7 — Generate GRPO Prompts + Launch Training

```python
import subprocess, sys
from pathlib import Path
import json

# Step 1: Generate GRPO prompt dataset
result = subprocess.run(
    [sys.executable, "aic/training/train_grpo.py", "--generate-prompts-only"],
    capture_output=True, text=True
)
print(result.stdout[-2000:])

# Verify GRPO prompts exist
grpo_prompts = list(Path("artifacts/grpo").glob("*.jsonl"))
assert grpo_prompts, "No GRPO prompts generated!"
total = sum(sum(1 for _ in open(f)) for f in grpo_prompts)
print(f"✅ GRPO prompts: {total}")

# Step 2: Launch GRPO training (this will run for hours — let it go)
result = subprocess.run(
    [sys.executable, "aic/training/train_grpo.py",
     "--max_steps", "150",
     "--per_device_train_batch_size", "1",
     "--gradient_accumulation_steps", "8",
     "--warmup_steps", "10",
     "--save_steps", "25",    # Checkpoint every 25 steps (Colab disconnect safety)
     "--logging_steps", "5",
    ],
    capture_output=False      # Stream live output
)
```

---

## CELL 7b — Colab Disconnect Recovery

If Colab disconnects mid-training (it will try), run this to resume:

```python
import os
from pathlib import Path

# Find latest intermediate checkpoint
checkpoints = sorted(
    Path("checkpoints/grpo").glob("checkpoint-*"),
    key=lambda p: int(p.name.split("-")[1])
)

if checkpoints:
    latest = checkpoints[-1]
    print(f"Resuming from: {latest}")
    os.system(f"python aic/training/train_grpo.py --resume_from_checkpoint {latest}")
else:
    print("No intermediate checkpoint found. Must restart from step 0.")
    print("Mount Google Drive first to persist checkpoints across sessions.")
```

### Prevent Disconnect Loss (Recommended)

Before running CELL 7, mount Google Drive and set `output_dir` to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# In config or CLI:
# --output_dir /content/drive/MyDrive/aic_checkpoints/grpo
```

---

## CELL 7c — Live Monitor (Open in Parallel Tab)

Open a second Colab tab and run this to watch progress without interrupting training:

```python
import json, time
from pathlib import Path

log_path = Path("logs/grpo_progress.jsonl")
last_step = 0

while True:
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]

        if entries and entries[-1]["step"] > last_step:
            last_step = entries[-1]["step"]
            first_reward = entries[0]["reward"]
            curr_reward = entries[-1]["reward"]
            delta = curr_reward - first_reward

            print(f"Step {last_step:4d} | reward: {curr_reward:+.4f} | "
                  f"Δ from start: {delta:+.4f} | "
                  f"elapsed: {entries[-1]['elapsed_minutes']:.1f}m")

    time.sleep(30)
```

---

## Reading the Training Signal

### ✅ Good GRPO (expected trajectory)

| Steps | What You See |
|-------|-------------|
| 0–20 | Reward fluctuates wildly — this is exploration, it's normal |
| 20–60 | Reward trend starts going up — good sign |
| 60–120 | Stabilization with upward trend — great |
| 120–150 | Final convergence — save checkpoint |

### ❌ Bad GRPO (with fixes)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Reward stuck at exact same value every step | Reward function not being called | Verify `use_reward_audit=True` in config |
| Loss is NaN | Learning rate too high | Lower LR by 10x in config |
| OOM at step ~50 | Sequence too long | Add `--max_prompt_length 512` |
| Reward briefly improves then collapses | Reward hacking | Check audit logs in `logs/audit/` |

---

## ✅ Phase 3 Completion Criteria

- [ ] `checkpoints/grpo/` directory exists
- [ ] `logs/grpo_progress.jsonl` has 50+ entries
- [ ] `checkpoints/grpo/training_summary.json` shows a positive `reward_delta`
- [ ] Reward curve shows upward trend (even if noisy)

**→ Next: [PHASE_4_BENCHMARK.md](PHASE_4_BENCHMARK.md)**
