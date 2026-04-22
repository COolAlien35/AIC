#!/usr/bin/env python3
"""Validate model export: load LoRA adapter, merge, run inference, confirm end-to-end.

Usage:
    python eval/test_export.py --source checkpoints/sft
    python eval/test_export.py --source checkpoints/grpo --merge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.training.config import TrainingConfig
from aic.training.export_model import export_model, validate_export


def test_export(
    source_dir: str,
    output_dir: str | None = None,
    merge: bool = False,
    push_to_hub: bool = False,
    hub_repo: str = "",
) -> bool:
    """Full export validation pipeline."""
    config = TrainingConfig()
    source = Path(source_dir)
    output = Path(output_dir or config.export_dir)

    print(f"=== Model Export Validation ===")
    print(f"Source:  {source}")
    print(f"Output:  {output}")
    print(f"Merge:   {merge}")
    print()

    # Step 1: Check source exists
    if not source.exists():
        print(f"❌ FAILED: Source directory does not exist: {source}")
        print("  → Run SFT or GRPO training first to produce a checkpoint.")
        return False
    print(f"✅ Source directory exists: {list(source.iterdir())[:10]}")

    # Step 2: Export model (optionally merge adapters)
    try:
        exported = export_model(str(source), str(output), merge_adapters=merge)
        print(f"✅ Model exported to: {exported}")
    except Exception as e:
        print(f"❌ FAILED: Export error: {e}")
        return False

    # Step 3: Validate by running inference
    try:
        ok = validate_export(str(output))
        if ok:
            print("✅ Inference validation passed")
        else:
            print("❌ FAILED: Inference validation returned False")
            return False
    except Exception as e:
        print(f"❌ FAILED: Inference validation error: {e}")
        return False

    # Step 4: Optional push to HuggingFace Hub
    if push_to_hub and hub_repo:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Pushing to HuggingFace Hub: {hub_repo}")
            model = AutoModelForCausalLM.from_pretrained(str(output))
            tokenizer = AutoTokenizer.from_pretrained(str(output))
            model.push_to_hub(hub_repo)
            tokenizer.push_to_hub(hub_repo)
            print(f"✅ Pushed to {hub_repo}")
        except Exception as e:
            print(f"⚠️ Push to Hub failed (non-fatal): {e}")

    print()
    print("=== Export Validation Complete ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate AIC model export")
    parser.add_argument("--source", default="checkpoints/sft",
                        help="Source checkpoint directory")
    parser.add_argument("--output", default=None,
                        help="Output directory for exported model")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapters into base model")
    parser.add_argument("--push", action="store_true",
                        help="Push exported model to HuggingFace Hub")
    parser.add_argument("--hub-repo", default="",
                        help="HuggingFace Hub repo ID")
    args = parser.parse_args()

    success = test_export(
        source_dir=args.source,
        output_dir=args.output,
        merge=args.merge,
        push_to_hub=args.push,
        hub_repo=args.hub_repo,
    )
    sys.exit(0 if success else 1)
