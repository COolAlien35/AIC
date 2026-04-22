"""Model export helpers for adapter-based SFT / RL training."""
from __future__ import annotations

from pathlib import Path

from aic.training.config import TrainingConfig


def export_model(source_dir: str, output_dir: str, merge_adapters: bool = False) -> Path:
    """Export a trained model directory, optionally merging PEFT adapters."""
    try:  # pragma: no cover - optional dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise ImportError("Transformers is required for export.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source_dir)

    model = AutoModelForCausalLM.from_pretrained(source_dir)
    if merge_adapters and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    return output_path


def validate_export(model_dir: str, prompt: str = "Select the safest candidate recommendation.") -> bool:
    """Smoke-test a saved model export by running one generation."""
    try:  # pragma: no cover - optional dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise ImportError("Transformers is required for export validation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    inputs = tokenizer(prompt, return_tensors="pt")
    _ = model.generate(**inputs, max_new_tokens=16)
    return True


if __name__ == "__main__":
    cfg = TrainingConfig()
    print(export_model(cfg.sft_output_dir, cfg.export_dir))
