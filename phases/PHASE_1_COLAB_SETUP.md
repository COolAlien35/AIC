# 🟡 PHASE 1 — COLAB SETUP (The GPU Engine)
> **Time: 20 minutes | Prerequisite: Phase 0 must be fully verified**

---

## Before You Open Colab

- Phase 0 verification must have passed (all 4 checks green)
- Your code must be zipped OR your repo must be public on GitHub
- Target runtime: **T4 GPU (16GB VRAM)** — do not proceed on CPU

---

## CELL 1 — GPU Check

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Must show T4 (16GB) or A100 (40GB). Do not proceed on CPU.
```

---

## CELL 2 — Install Dependencies

```bash
%%bash
pip install -q unsloth
pip install -q trl transformers peft accelerate bitsandbytes
pip install -q scipy pandas

# If unsloth fails on this runtime:
# pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## CELL 3 — Upload & Mount Code

**Option A: Git clone (if repo is public)**

```python
!git clone https://github.com/YOUR_ORG/aic.git
%cd aic
!ls
```

**Option B: Upload zip**

```python
from google.colab import files
uploaded = files.upload()  # Upload aic.zip

import zipfile, os
with zipfile.ZipFile(list(uploaded.keys())[0], 'r') as z:
    z.extractall('/content/aic')

os.chdir('/content/aic')
!ls  # Verify structure
```

---

## CELL 4 — Critical Pre-Flight Check

```python
from aic.training.config import TrainingConfig

c = TrainingConfig()

# HARD STOP — if this fails, go back and fix Phase 0
assert 'Qwen' in c.model_name, f"STOP: Wrong model loaded: {c.model_name}"
assert c.use_reward_audit, "STOP: Reward audit is disabled"
assert c.grpo_max_steps >= 100, f"STOP: Too few GRPO steps: {c.grpo_max_steps}"

print(f"✅ Ready to train")
print(f"   Model:      {c.model_name}")
print(f"   GRPO steps: {c.grpo_max_steps}")
print(f"   Reward audit: {c.use_reward_audit}")
```

---

## ✅ Phase 1 Completion Criteria

- [ ] GPU detected (T4 or better)
- [ ] All pip installs completed without fatal errors
- [ ] Code is mounted at `/content/aic`
- [ ] Pre-flight check passes — `Qwen` model confirmed, no `tiny-gpt2`

**→ Next: [PHASE_2_SFT_TRAINING.md](PHASE_2_SFT_TRAINING.md)**
