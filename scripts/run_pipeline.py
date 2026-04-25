#!/usr/bin/env python3
"""End-to-end AIC training pipeline driver.

Subcommands
-----------
verify   nvidia-smi + dependency diagnostics + free disk + free VRAM
smoke    1-step SFT + 2-step GRPO; asserts reward_std>0 and parse_rate>=70%
full     SFT (1 epoch) + GRPO (200 steps) + export + benchmark + plots

Each stage writes ``logs/pipeline/stage_<name>.json`` so a downstream
notebook can render progress without re-running anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PIPELINE_LOG_DIR = REPO_ROOT / "logs" / "pipeline"
PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _stage_log(name: str, payload: dict[str, Any]) -> Path:
    out = PIPELINE_LOG_DIR / f"stage_{name}.json"
    payload["timestamp"] = time.time()
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(_args) -> int:
    _print_header("STAGE: verify")
    info: dict[str, Any] = {"stage": "verify"}

    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        print(out)
        info["nvidia_smi"] = out.splitlines()[:25]
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
        print(f"[!] nvidia-smi failed: {exc}")

    try:
        from aic.utils.dependency_diagnostics import print_dependency_diagnostics

        print_dependency_diagnostics()
        info["dependency_check"] = "ok"
    except Exception as exc:
        info["dependency_check_error"] = str(exc)
        print(f"[!] dependency diagnostics failed: {exc}")

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device"] = torch.cuda.get_device_name(0)
            info["torch_cuda_capability"] = list(torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info(0)
            info["torch_vram_free_gb"] = round(free / 1e9, 2)
            info["torch_vram_total_gb"] = round(total / 1e9, 2)
            print(
                f"[verify] CUDA: {info['torch_cuda_device']} "
                f"VRAM free {info['torch_vram_free_gb']} / {info['torch_vram_total_gb']} GB"
            )
    except Exception as exc:
        info["torch_error"] = str(exc)

    try:
        import shutil

        usage = shutil.disk_usage(str(REPO_ROOT))
        info["disk_free_gb"] = round(usage.free / 1e9, 2)
        info["disk_total_gb"] = round(usage.total / 1e9, 2)
        print(f"[verify] Disk free: {info['disk_free_gb']} / {info['disk_total_gb']} GB")
    except Exception as exc:
        info["disk_error"] = str(exc)

    info["status"] = "ok"
    _stage_log("verify", info)
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def _smoke_sft() -> dict[str, Any]:
    """Build a tiny SFT dataset (3 episodes) and run a single training step."""
    from aic.training.config import TrainingConfig
    from aic.training.generate_sft_data import generate_sft_dataset
    from aic.training.run_sft import run_sft

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_num_episodes=12,
        sft_epochs=1,
        sft_batch_size=1,
        sft_grad_accumulation=1,
        sft_learning_rate=2e-4,
        max_prompt_length=1024,
        max_completion_length=192,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=8,
        lora_alpha=16,
    )

    # --- relax data integrity gates that assume the full 600-row dataset ---
    try:
        from aic.training import data_integrity

        original_min = data_integrity.DataQualityGates.min_total_records
    except Exception:
        original_min = None

    print("[smoke] Generating tiny SFT dataset (12 episodes)...")
    try:
        dataset_path = generate_sft_dataset(cfg)
    except AssertionError:
        # Older generator hard-asserts >=400 rows. Bypass for smoke by writing
        # 12 short rows directly.
        print("[smoke] Heuristic generator asserted; using minimal generator path.")
        dataset_path = _minimal_sft_dataset(cfg)
    except Exception as exc:
        return {"ok": False, "error": f"sft_data: {exc}"}

    print(f"[smoke] Dataset at {dataset_path}")

    try:
        out = run_sft(cfg, min_parse_rate=0.0)  # smoke records the rate but does not abort
    except Exception as exc:
        return {"ok": False, "error": f"sft_train: {exc}\n{traceback.format_exc()}"}

    parse_rate = 0.0
    try:
        with open(Path(out) / "sft_metadata.json") as f:
            parse_rate = float(json.load(f).get("parse_rate", 0.0))
    except Exception:
        pass

    return {"ok": True, "sft_dir": str(out), "parse_rate": parse_rate}


def _minimal_sft_dataset(cfg) -> Path:
    """Fallback for smoke when the heuristic generator hard-asserts 400+."""
    from aic.training.generate_sft_data import generate_sft_dataset
    from aic.training import data_integrity

    data_integrity.DataQualityGates.min_total_records = 1
    return generate_sft_dataset(cfg)


def _smoke_grpo(sft_dir: str | None) -> dict[str, Any]:
    from aic.training.config import TrainingConfig
    from aic.training.train_grpo import run_grpo

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_output_dir=sft_dir or "checkpoints/sft",
        sft_num_episodes=12,  # generates ~12 prompts for the GRPO dataset
        grpo_max_steps=2,
        grpo_per_device_train_batch_size=1,
        grpo_gradient_accumulation_steps=1,
        grpo_num_generations=4,
        max_prompt_length=1024,
        max_completion_length=128,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=8,
        lora_alpha=16,
    )

    try:
        run_grpo(cfg)
    except Exception as exc:
        return {"ok": False, "error": f"grpo: {exc}\n{traceback.format_exc()}"}

    log_path = REPO_ROOT / "logs" / "grpo_progress.jsonl"
    max_std = 0.0
    rewards: list[float] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except Exception:
                continue
            max_std = max(max_std, float(entry.get("reward_std", 0.0)))
            rewards.append(float(entry.get("reward", 0.0)))

    return {
        "ok": True,
        "max_reward_std": max_std,
        "rewards": rewards,
        "log_path": str(log_path),
    }


def cmd_smoke(_args) -> int:
    _print_header("STAGE: smoke")
    if cmd_verify(_args) != 0:
        _stage_log("smoke", {"stage": "smoke", "status": "verify_failed"})
        return 1

    sft_result = _smoke_sft()
    print(f"[smoke] SFT: {sft_result}")
    if not sft_result["ok"]:
        _stage_log("smoke", {"stage": "smoke", "status": "sft_failed", **sft_result})
        return 1

    grpo_result = _smoke_grpo(sft_result["sft_dir"])
    print(f"[smoke] GRPO: ok={grpo_result['ok']} max_std={grpo_result.get('max_reward_std')}")
    if not grpo_result["ok"]:
        _stage_log(
            "smoke",
            {"stage": "smoke", "status": "grpo_failed", "sft": sft_result, "grpo": grpo_result},
        )
        return 1

    parse_rate = sft_result.get("parse_rate", 0.0)
    max_std = grpo_result.get("max_reward_std", 0.0)
    parse_ok = parse_rate >= 0.7
    std_ok = max_std > 0.0

    summary = {
        "stage": "smoke",
        "sft": sft_result,
        "grpo": grpo_result,
        "parse_rate_gate_passed": parse_ok,
        "reward_std_gate_passed": std_ok,
        "status": "ok" if (parse_ok and std_ok) else "gate_failed",
    }
    _stage_log("smoke", summary)

    if not (parse_ok and std_ok):
        print("\n[smoke] FAILED gates:")
        if not parse_ok:
            print(f"   parse_rate {parse_rate:.2f} < 0.70")
        if not std_ok:
            print(f"   max reward_std {max_std:.3f} == 0 (no GRPO learning signal)")
        print("\nDo NOT run --full until smoke gates pass.")
        return 2

    print("\n[smoke] All gates passed. Safe to run --full.")
    return 0


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------

def _full_sft() -> dict[str, Any]:
    from aic.training.config import TrainingConfig
    from aic.training.generate_sft_data import generate_sft_dataset
    from aic.training.run_sft import run_sft

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_num_episodes=120,
        sft_epochs=1,
        sft_batch_size=2,
        sft_grad_accumulation=8,
        sft_learning_rate=2e-4,
        max_prompt_length=1024,
        max_completion_length=192,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=16,
        lora_alpha=32,
    )

    print("[full] Generating SFT dataset (120 episodes)...")
    dataset_path = generate_sft_dataset(cfg)
    print(f"[full] Dataset at {dataset_path}")

    out = run_sft(cfg, min_parse_rate=0.7)
    return {"sft_dir": str(out)}


def _full_grpo(sft_dir: str) -> dict[str, Any]:
    from aic.training.config import TrainingConfig
    from aic.training.train_grpo import run_grpo

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_output_dir=sft_dir,
        sft_num_episodes=120,
        grpo_max_steps=200,
        grpo_per_device_train_batch_size=1,
        grpo_gradient_accumulation_steps=4,
        grpo_num_generations=4,
        max_prompt_length=1024,
        max_completion_length=192,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=16,
        lora_alpha=32,
    )
    out = run_grpo(cfg)
    return {"grpo_dir": str(out)}


def _full_export(grpo_dir: str) -> dict[str, Any]:
    from aic.training.export_model import export_grpo_to_full

    out = export_grpo_to_full(grpo_dir, "exports", base_model_name="Qwen/Qwen2.5-3B-Instruct")
    return {"exports_dir": str(out)}


def _full_benchmark() -> dict[str, Any]:
    from scripts.run_final_benchmark import run_benchmark

    df, stats = run_benchmark(
        num_episodes_per_scenario=5,
        output_dir="results",
        seed=42,
        requested_scenarios="all",
        requested_policies="all",
        checkpoint_path="exports",
        strict=True,
    )
    return {
        "summary_path": "results/benchmark_summary.csv",
        "stats": stats,
        "rows": len(df),
    }


def cmd_full(_args) -> int:
    _print_header("STAGE: full")

    smoke_log = PIPELINE_LOG_DIR / "stage_smoke.json"
    if smoke_log.exists():
        try:
            payload = json.loads(smoke_log.read_text())
            if payload.get("status") != "ok":
                print(
                    "[full] Smoke last run did not pass gates "
                    "(status="
                    f"{payload.get('status')}). Re-run `python scripts/run_pipeline.py smoke` first."
                )
                return 1
        except Exception:
            pass

    try:
        sft_result = _full_sft()
        _stage_log("sft", {"stage": "sft", "status": "ok", **sft_result})
    except Exception as exc:
        _stage_log(
            "sft",
            {"stage": "sft", "status": "failed", "error": str(exc),
             "trace": traceback.format_exc()},
        )
        print(f"[full] SFT failed: {exc}")
        return 2

    try:
        grpo_result = _full_grpo(sft_result["sft_dir"])
        _stage_log("grpo", {"stage": "grpo", "status": "ok", **grpo_result})
    except Exception as exc:
        _stage_log(
            "grpo",
            {"stage": "grpo", "status": "failed", "error": str(exc),
             "trace": traceback.format_exc()},
        )
        print(f"[full] GRPO failed: {exc}")
        return 3

    try:
        export_result = _full_export(grpo_result["grpo_dir"])
        _stage_log("export", {"stage": "export", "status": "ok", **export_result})
    except Exception as exc:
        _stage_log(
            "export",
            {"stage": "export", "status": "failed", "error": str(exc),
             "trace": traceback.format_exc()},
        )
        print(f"[full] Export failed: {exc}")
        return 4

    try:
        bench_result = _full_benchmark()
        _stage_log("benchmark", {"stage": "benchmark", "status": "ok", **bench_result})
    except Exception as exc:
        _stage_log(
            "benchmark",
            {"stage": "benchmark", "status": "failed", "error": str(exc),
             "trace": traceback.format_exc()},
        )
        print(f"[full] Benchmark failed: {exc}")
        return 5

    print("\n[full] Pipeline complete. See results/ and logs/pipeline/ for evidence.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AIC training pipeline driver")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("smoke")
    sub.add_parser("full")
    args = parser.parse_args()

    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "full":
        return cmd_full(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
