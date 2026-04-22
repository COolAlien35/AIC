#!/usr/bin/env python3
"""Smoke test: generate SFT data, run SFT training, run GRPO training.

All with minimal configs (2 episodes, 1 epoch, 2 steps) to verify the full stack.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def step1_generate_sft_data():
    """Generate SFT data from 2 heuristic episodes."""
    print("=" * 60)
    print("STEP 1: Generating SFT data...")
    print("=" * 60)
    from aic.training.config import TrainingConfig
    from aic.training.generate_sft_data import generate_sft_dataset

    config = TrainingConfig(sft_num_episodes=2)
    path = generate_sft_dataset(config)
    with open(path) as f:
        lines = f.readlines()
    print(f"✅ SFT data generated: {path}")
    print(f"   Records: {len(lines)}")
    if lines:
        rec = json.loads(lines[0])
        print(f"   Keys: {list(rec.keys())}")
        print(f"   Prompt length: {len(rec.get('prompt', ''))}")
        print(f"   Completion length: {len(rec.get('completion', ''))}")
    return path


def step2_run_sft():
    """Run SFT training with minimal config."""
    print("\n" + "=" * 60)
    print("STEP 2: Running SFT training (1 epoch, tiny batch)...")
    print("=" * 60)
    from aic.training.config import TrainingConfig
    from aic.training.run_sft import run_sft

    config = TrainingConfig(
        sft_num_episodes=2,
        sft_epochs=1,
        sft_batch_size=1,
        sft_learning_rate=2e-5,
        model_name="Qwen/Qwen2-0.5B-Instruct",
        use_peft_for_sft=True,
        lora_r=4,
        lora_alpha=16,
        max_prompt_length=512,
        max_completion_length=128,
    )
    try:
        output_dir = run_sft(config)
        print(f"✅ SFT training complete: {output_dir}")
        print(f"   Files: {[f.name for f in Path(output_dir).iterdir()]}")
        return str(output_dir)
    except Exception as e:
        print(f"❌ SFT training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step3_generate_grpo_prompts():
    """Generate GRPO prompt dataset."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating GRPO prompt dataset...")
    print("=" * 60)
    from aic.training.config import TrainingConfig
    from aic.training.train_grpo import generate_grpo_prompt_dataset

    config = TrainingConfig(sft_num_episodes=2)
    path = generate_grpo_prompt_dataset(config)
    with open(path) as f:
        lines = f.readlines()
    print(f"✅ GRPO prompts generated: {path}")
    print(f"   Records: {len(lines)}")
    return str(path)


def step4_run_grpo():
    """Run GRPO training with minimal config."""
    print("\n" + "=" * 60)
    print("STEP 4: Running GRPO training (10 steps)...")
    print("=" * 60)
    from aic.training.config import TrainingConfig
    from aic.training.train_grpo import run_grpo

    config = TrainingConfig(
        sft_num_episodes=2,
        model_name="Qwen/Qwen2-0.5B-Instruct",
        grpo_max_steps=10,
        grpo_per_device_train_batch_size=1,
        grpo_gradient_accumulation_steps=1,
        grpo_num_generations=2,
        max_prompt_length=512,
        max_completion_length=128,
        use_unsloth=False,
        lora_r=4,
        lora_alpha=16,
    )
    try:
        output_dir = run_grpo(config)
        print(f"✅ GRPO training complete: {output_dir}")
        print(f"   Files: {[f.name for f in Path(output_dir).iterdir()]}")
        return str(output_dir)
    except Exception as e:
        print(f"❌ GRPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if not args or "sft-data" in args or "all" in args:
        step1_generate_sft_data()
    
    if "sft" in args or "all" in args:
        step2_run_sft()
    
    if "grpo-data" in args or "all" in args:
        step3_generate_grpo_prompts()
    
    if "grpo" in args or "all" in args:
        step4_run_grpo()
    
    if not args:
        print("\nUsage: python3 run_training_smoke.py [sft-data|sft|grpo-data|grpo|all]")
        print("Default: sft-data only")
