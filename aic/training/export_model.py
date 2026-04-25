"""Model export helpers for adapter-based SFT / RL training.

The benchmark loader (``scripts/run_final_benchmark.py``) requires a directory
containing a real ``config.json`` plus model weights. After GRPO, the
checkpoint is just a LoRA adapter plus the tokenizer, so we have to merge it
back into the base model and write a self-contained directory.

``export_grpo_to_full`` does that and runs a 1-prompt smoke generation so we
catch a broken export before the benchmark blames the model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aic.training.config import TrainingConfig


def _resolve_base_model_name(adapter_dir: Path, fallback: str) -> str:
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        try:
            with open(cfg) as f:
                payload = json.load(f)
            base = payload.get("base_model_name_or_path")
            if base:
                return str(base)
        except Exception:
            pass
    return fallback


def export_grpo_to_full(
    grpo_dir: str | Path,
    output_dir: str | Path,
    base_model_name: str | None = None,
) -> Path:
    """Merge a GRPO LoRA adapter into the base model and save a full checkpoint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("transformers + peft are required to merge an adapter.") from exc

    grpo_dir = Path(grpo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not grpo_dir.exists():
        raise FileNotFoundError(f"GRPO checkpoint dir missing: {grpo_dir}")

    fallback = base_model_name or TrainingConfig().model_name
    base_name = _resolve_base_model_name(grpo_dir, fallback)
    print(f"[export] Merging adapter from {grpo_dir} into base {base_name}")

    has_cuda = torch.cuda.is_available()
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
    )
    model = PeftModel.from_pretrained(base, str(grpo_dir))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(output_dir), safe_serialization=True)

    tok_source = grpo_dir if (grpo_dir / "tokenizer_config.json").exists() else base_name
    tokenizer = AutoTokenizer.from_pretrained(str(tok_source), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(str(output_dir))

    smoke_ok = False
    smoke_error: str | None = None
    try:
        device = next(merged.parameters()).device
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Say {\"ok\":true}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = merged.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        smoke_ok = bool(tokenizer.decode(out[0], skip_special_tokens=True))
    except Exception as exc:  # pragma: no cover - best effort
        smoke_error = str(exc)

    metadata: dict[str, Any] = {
        "source_grpo_dir": str(grpo_dir),
        "base_model_name": base_name,
        "smoke_ok": smoke_ok,
        "smoke_error": smoke_error,
    }
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if not smoke_ok:
        raise RuntimeError(
            f"Export smoke generation failed: {smoke_error}. "
            f"See {output_dir}/export_metadata.json"
        )

    return output_dir


def validate_export(model_dir: str, prompt: str = "Select the safest candidate.") -> bool:
    """Smoke-test a saved model export by running one generation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ImportError("Transformers is required for export validation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    inputs = tokenizer(prompt, return_tensors="pt")
    _ = model.generate(**inputs, max_new_tokens=16)
    return True


# Backwards compat shim — older entrypoints import ``export_model``.
def export_model(source_dir: str, output_dir: str, merge_adapters: bool = True) -> Path:
    return export_grpo_to_full(source_dir, output_dir)


if __name__ == "__main__":
    cfg = TrainingConfig()
    print(export_grpo_to_full(cfg.grpo_output_dir, cfg.export_dir, cfg.model_name))
