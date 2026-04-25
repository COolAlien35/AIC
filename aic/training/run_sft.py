"""Supervised warm-start training for the orchestrator policy.

Critical fixes vs the previous revision:

  * Uses Qwen's native chat template via ``apply_chat_template`` so the
    model sees the same ChatML wrapping it was instruct-tuned on.
  * LoRA is mandatory (rank 16, alpha 32) on all 7 Qwen projection layers.
  * Padding side is forced to "left" so the labels mask matches the actual
    completion span when batched.
  * Writes a CORRECT ``sft_metadata.json`` containing the real base model
    name (e.g. ``Qwen/Qwen2.5-3B-Instruct``) so the GRPO warm-start gate
    actually triggers.
  * Runs a parse-rate smoke generation at the end of training and
    raises if it falls below ``min_parse_rate`` (default 0.7).
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from aic.schemas.actions import OrchestratorDecision
from aic.training.config import TrainingConfig
from aic.training.ddp_utils import is_main_process, local_rank as _ddp_local_rank, wait_for_everyone
from aic.training.modeling_unsloth import (
    QWEN_TARGET_MODULES,
    _build_bnb_config,
    _preferred_compute_dtype,
)


# region agent log
_AGENT_DEBUG_LOG = Path("/Users/pulkitpandey/Desktop/AIC/.cursor/debug-b030f6.log")
_AGENT_DEBUG_FALLBACK_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-b030f6.log"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    """Temporary Cursor debug instrumentation for session b030f6."""
    payload = {
        "sessionId": "b030f6",
        "runId": os.environ.get("AIC_AGENT_DEBUG_RUN_ID", "pre-fix"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, default=str) + "\n"
    for _path in (_AGENT_DEBUG_LOG, _AGENT_DEBUG_FALLBACK_LOG):
        try:
            _path.parent.mkdir(parents=True, exist_ok=True)
            with _path.open("a") as f:
                f.write(line)
        except Exception:
            pass
# endregion


def _write_prompt_completion_jsonl(dataset_path: Path) -> Path:
    """Write a temporary JSONL containing only ``prompt`` and ``completion``.

    The SFT generator emits rich metadata (scenario, drift, difficulty, optional
    negative-sample fields). Mixing that into ``datasets.load_dataset("json",
    ...)`` triggers schema-cast failures because positive and negative rows
    expose different ``metadata`` substructures. Since SFT only uses the two
    text fields, we strip everything else into a sibling file and load that.
    """
    cleaned_path = dataset_path.parent / "_prompt_completion_only.jsonl"
    with dataset_path.open() as src, cleaned_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dst.write(json.dumps({
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("completion", "")),
            }) + "\n")
    return cleaned_path


def _require_sft_dependencies():
    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "SFT dependencies missing. Install `datasets`, `transformers`, `peft`."
        ) from exc

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise ImportError("`peft` is required for SFT.") from exc

    return (
        load_dataset, AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments,
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    )


def _build_chat_text(tokenizer, prompt_text: str, completion_text: str) -> tuple[str, int]:
    """Render a (prompt, completion) pair through the chat template.

    Returns the full text and the token count of just the prompt portion
    (so we can mask labels for it).
    """
    user_msg = [
        {"role": "system", "content": (
            "You are the Adaptive Incident Choreographer orchestrator. "
            "Output strict JSON only."
        )},
        {"role": "user", "content": prompt_text},
    ]
    prompt_only = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full_text = prompt_only + completion_text + tokenizer.eos_token
    return full_text, len(prompt_ids)


def _make_tokenize_fn(tokenizer, config: TrainingConfig):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        prompt_text, _ = _build_chat_text(tokenizer, example["prompt"], "")
        completion_text = str(example["completion"]) + tokenizer.eos_token

        # Tokenize prompt and completion separately so long prompts cannot
        # consume the completion budget and silently zero out all labels.
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=config.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completion_text,
            truncation=True,
            max_length=config.max_completion_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + list(completion_ids)

        # Safety: if completion got fully truncated, keep at least EOS as a
        # supervised target to avoid all-masked batches and zero loss/grad.
        if not completion_ids and tokenizer.eos_token_id is not None:
            input_ids = prompt_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(prompt_ids)) + [tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _tokenize


def _sft_completion_length(config: TrainingConfig) -> int:
    """SFT target JSONs are longer than GRPO rollout caps; keep them separate."""
    raw = os.environ.get("AIC_SFT_MAX_COMPLETION_LENGTH", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            print(f"[SFT] WARNING: AIC_SFT_MAX_COMPLETION_LENGTH={raw!r} is not int - ignored")
    return max(192, int(config.max_completion_length))


def _orchestrator_json_candidates(completion: str) -> list[str]:
    """Return JSON substrings to try for ``OrchestratorDecision`` validation."""
    t = completion.strip()
    if not t:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    _add(t)
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE):
        _add(m.group(1))
    i0, i1 = t.find("{"), t.rfind("}")
    if i0 >= 0 and i1 > i0:
        _add(t[i0 : i1 + 1])
    return out


def _smoke_parse_rate(
    model,
    tokenizer,
    dataset,
    sample_n: int = 8,
    *,
    max_completion_hint: int = 96,
) -> float:
    """Generate completions for ``sample_n`` random prompts and report parse rate.

    Uses **greedy** decoding: after a short SFT cap, sampling with temperature
    often yields 0/8 valid JSON by chance even when training loss moved, which
    falsely fails smoke gates.
    """
    try:
        import torch
    except Exception:
        return 0.0
    if len(dataset) == 0:
        return 0.0
    n = min(sample_n, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    model.eval()
    successes = 0
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    max_new = min(512, max(192, max_completion_hint * 3))
    # region agent log
    gen_cfg = getattr(model, "generation_config", None)
    _agent_debug_log(
        "A,D,E",
        "aic/training/run_sft.py:_smoke_parse_rate:start",
        "SFT parse probe starting",
        {
            "dataset_len": len(dataset),
            "sample_n": n,
            "indices": indices,
            "device": str(device),
            "model_class": type(model).__name__,
            "active_adapters": getattr(model, "active_adapters", None),
            "pad_id": pad_id,
            "eos_id": eos_id,
            "max_new_tokens": max_new,
            "generation_config": {
                "do_sample": getattr(gen_cfg, "do_sample", None),
                "temperature": getattr(gen_cfg, "temperature", None),
                "top_p": getattr(gen_cfg, "top_p", None),
                "top_k": getattr(gen_cfg, "top_k", None),
            },
        },
    )
    # endregion
    for idx in indices:
        prompt_text = dataset[idx]["prompt"]
        rendered, _ = _build_chat_text(tokenizer, prompt_text, "")
        inputs = tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=1024,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        sample_ok = False
        sample_error = None
        candidate_count = 0
        for cand in _orchestrator_json_candidates(completion):
            candidate_count += 1
            try:
                OrchestratorDecision.model_validate_json(cand)
                successes += 1
                sample_ok = True
                break
            except Exception as exc:
                sample_error = str(exc)[:500]
                continue
        # region agent log
        _agent_debug_log(
            "A,B",
            "aic/training/run_sft.py:_smoke_parse_rate:sample",
            "SFT parse probe sample result",
            {
                "idx": idx,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                "completion_len": len(completion),
                "completion_prefix": completion[:600],
                "candidate_count": candidate_count,
                "has_brace": "{" in completion and "}" in completion,
                "valid": sample_ok,
                "last_validation_error": sample_error,
            },
        )
        # endregion
    model.train()
    # region agent log
    _agent_debug_log(
        "A,B,D,E",
        "aic/training/run_sft.py:_smoke_parse_rate:end",
        "SFT parse probe finished",
        {"successes": successes, "n": n, "parse_rate": successes / max(1, n)},
    )
    # endregion
    return successes / max(1, n)


def run_sft(config: TrainingConfig | None = None, min_parse_rate: float = 0.7) -> Path:
    """Run instruction-style SFT on prompt/completion JSONL data."""
    if config is None:
        config = TrainingConfig()

    (
        load_dataset, AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments,
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    ) = _require_sft_dependencies()

    dataset_path = Path(config.sft_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Run generate_sft_data.py first."
        )

    if is_main_process():
        _write_prompt_completion_jsonl(dataset_path)
    wait_for_everyone()
    cleaned_train_path = dataset_path.parent / "_prompt_completion_only.jsonl"
    raw_dataset = load_dataset(
        "json",
        data_files=str(cleaned_train_path),
        split="train",
    )

    val_path = dataset_path.parent / "val.jsonl"
    eval_dataset = None
    disable_eval = os.environ.get("AIC_SFT_DISABLE_EVAL", "").lower() in (
        "1", "true", "yes",
    )
    if val_path.exists() and not disable_eval:
        if is_main_process():
            _write_prompt_completion_jsonl(val_path)
        wait_for_everyone()
        cleaned_val_path = val_path.parent / "_prompt_completion_only.jsonl"
        eval_dataset = load_dataset(
            "json",
            data_files=str(cleaned_val_path),
            split="train",
        )
    elif val_path.exists() and disable_eval:
        print("[SFT] AIC_SFT_DISABLE_EVAL set - skipping eval split")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    # device_map={"":0} pins all layers to a single GPU. "auto" can split a
    # 4-bit model across CPU/GPU which then breaks TRL's reference-model
    # preparation in the GRPO step that consumes this checkpoint.
    load_kwargs: dict[str, Any] = {
        "device_map": {"": _ddp_local_rank()} if has_cuda else None,
        "low_cpu_mem_usage": True,
    }
    bnb = _build_bnb_config(config.load_in_4bit and has_cuda)
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    elif has_cuda:
        load_kwargs["torch_dtype"] = _preferred_compute_dtype() or torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if config.load_in_4bit and has_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)

    sft_max_completion_length = _sft_completion_length(config)
    sft_tokenize_config = replace(
        config,
        max_completion_length=sft_max_completion_length,
    )
    if sft_max_completion_length != config.max_completion_length:
        print(
            f"[SFT] Using max_completion_length={sft_max_completion_length} "
            f"for SFT labels (base config={config.max_completion_length})."
        )

    tokenize_fn = _make_tokenize_fn(tokenizer, sft_tokenize_config)
    tokenized_train = raw_dataset.map(
        tokenize_fn, remove_columns=raw_dataset.column_names,
    )
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_fn, remove_columns=eval_dataset.column_names,
        )
    # region agent log
    try:
        sample_count = min(32, len(raw_dataset))
        completion_lengths: list[int] = []
        completion_truncated = 0
        non_masked_labels: list[int] = []
        for _i in range(sample_count):
            comp_ids = tokenizer(
                str(raw_dataset[_i]["completion"]) + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            completion_lengths.append(len(comp_ids))
            if len(comp_ids) > sft_tokenize_config.max_completion_length:
                completion_truncated += 1
            labels = tokenized_train[_i]["labels"]
            non_masked_labels.append(sum(1 for x in labels if int(x) != -100))
        _agent_debug_log(
            "C",
            "aic/training/run_sft.py:run_sft:tokenization_stats",
            "SFT tokenization and label stats before training",
            {
                "sample_count": sample_count,
                "max_completion_length": sft_tokenize_config.max_completion_length,
                "completion_lengths_min": min(completion_lengths) if completion_lengths else None,
                "completion_lengths_max": max(completion_lengths) if completion_lengths else None,
                "completion_lengths_avg": (
                    sum(completion_lengths) / len(completion_lengths)
                    if completion_lengths else None
                ),
                "completion_truncated": completion_truncated,
                "non_masked_labels_min": min(non_masked_labels) if non_masked_labels else None,
                "non_masked_labels_max": max(non_masked_labels) if non_masked_labels else None,
                "non_masked_labels_avg": (
                    sum(non_masked_labels) / len(non_masked_labels)
                    if non_masked_labels else None
                ),
            },
        )
    except Exception as exc:
        _agent_debug_log(
            "C",
            "aic/training/run_sft.py:run_sft:tokenization_stats_error",
            "Failed to collect SFT tokenization stats",
            {"error": str(exc)[:500]},
        )
    # endregion

    max_steps_env = os.environ.get("AIC_SFT_MAX_STEPS", "").strip()
    max_steps_override: int | None = None
    if max_steps_env:
        try:
            max_steps_override = int(max_steps_env)
            print(f"[SFT] AIC_SFT_MAX_STEPS={max_steps_override} - capping training")
        except ValueError:
            print(f"[SFT] WARNING: AIC_SFT_MAX_STEPS={max_steps_env!r} is not int - ignored")

    ta_kwargs: dict[str, Any] = dict(
        output_dir=config.sft_output_dir,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accumulation,
        learning_rate=config.sft_learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="steps" if tokenized_eval is not None else "no",
        eval_steps=50 if tokenized_eval is not None else None,
        report_to=[],
        no_cuda=not has_cuda,
        use_mps_device=False,
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=has_cuda,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        save_total_limit=1,
        remove_unused_columns=False,
        optim="adamw_8bit" if has_cuda else "adamw_torch",
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
    )
    if max_steps_override is not None:
        ta_kwargs["max_steps"] = max_steps_override
    else:
        ta_kwargs["num_train_epochs"] = config.sft_epochs

    args = TrainingArguments(**ta_kwargs)

    # transformers 4.46.x: ``Trainer._wrap_model`` enables ``nn.DataParallel`` when
    # ``args.n_gpu > 1`` unless ``model.is_loaded_in_8bit``. 4-bit QLoRA is not excluded,
    # so multi-GPU Spaces wrap PEFT+4bit in DataParallel and crash (scatter on kwargs).
    # ``TrainingArguments.__post_init__`` already ran ``_setup_devices`` and set
    # ``_n_gpu = torch.cuda.device_count()``; forcing ``_n_gpu = 1`` for non-DDP 4-bit
    # keeps training on a single visible device without DataParallel.
    _ws_raw = os.environ.get("WORLD_SIZE", "1") or "1"
    try:
        _ws_sft = int(_ws_raw)
    except ValueError:
        _ws_sft = 1
    if has_cuda and config.load_in_4bit and _ws_sft <= 1 and getattr(args, "_n_gpu", 1) > 1:
        print(
            f"[SFT] Forcing TrainingArguments._n_gpu=1 (was {args._n_gpu}) for 4-bit "
            "single-process training — transformers<=4.46 would otherwise use DataParallel."
        )
        args._n_gpu = 1

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        ),
    )

    train_result = trainer.train()

    output_dir = Path(config.sft_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_loss: float | None = None
    try:
        if train_result is not None and train_result.metrics:
            final_loss = float(train_result.metrics.get("train_loss"))
    except Exception:
        final_loss = None

    parse_rate = 0.0
    parse_error: str | None = None
    if is_main_process():
        try:
            parse_rate = _smoke_parse_rate(
                model,
                tokenizer,
                raw_dataset,
                sample_n=8,
                max_completion_hint=sft_tokenize_config.max_completion_length,
            )
        except Exception as exc:
            parse_error = str(exc)
        _pr_file = output_dir / "_sft_parse_rate_sync.json"
        with open(_pr_file, "w") as _f:
            json.dump({"parse_rate": parse_rate, "error": parse_error}, _f)
    wait_for_everyone()
    if not is_main_process():
        _pr_file = output_dir / "_sft_parse_rate_sync.json"
        try:
            with open(_pr_file) as _f:
                _sync = json.load(_f)
            parse_rate = float(_sync.get("parse_rate", 0.0))
            e = _sync.get("error")
            parse_error = str(e) if e else None
        except Exception:
            pass
    try:
        if is_main_process() and (output_dir / "_sft_parse_rate_sync.json").exists():
            (output_dir / "_sft_parse_rate_sync.json").unlink()
    except Exception:
        pass
    wait_for_everyone()

    metadata = {
        "dataset": str(dataset_path),
        "dataset_size": len(raw_dataset),
        "model_name": config.model_name,
        "final_loss": final_loss,
        "used_cuda": has_cuda,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": QWEN_TARGET_MODULES,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": sft_tokenize_config.max_completion_length,
        "base_config_max_completion_length": config.max_completion_length,
        "parse_rate": parse_rate,
        "parse_smoke_n": 8,
        "parse_smoke_error": parse_error,
    }
    # region agent log
    _agent_debug_log(
        "A,B,C,D,E",
        "aic/training/run_sft.py:run_sft:metadata",
        "SFT metadata before save",
        {
            "final_loss": final_loss,
            "parse_rate": parse_rate,
            "parse_error": parse_error,
            "used_cuda": has_cuda,
            "load_in_4bit": config.load_in_4bit,
            "output_dir": str(output_dir),
        },
    )
    # endregion
    if is_main_process():
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        with open(output_dir / "sft_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    wait_for_everyone()

    def _cleanup_after_train() -> None:
        try:
            del trainer
            del model
        except Exception:
            pass
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    if parse_rate < min_parse_rate:
        _cleanup_after_train()
        raise RuntimeError(
            f"SFT parse-rate gate failed: got {parse_rate:.2f} < required "
            f"{min_parse_rate:.2f}. Inspect {output_dir}/sft_metadata.json. "
            "Common causes: dataset too small, prompt template mismatch, or "
            "max_completion_length too low for the JSON schema."
        )

    _cleanup_after_train()
    return output_dir


if __name__ == "__main__":
    print(run_sft())
