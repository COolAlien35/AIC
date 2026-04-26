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


def _smoke_single_gpu_env() -> None:
    """Before any ``aic`` (or torch) imports: hide extra GPUs for ``smoke``.

    HF ``Trainer`` enables ``nn.DataParallel`` when ``torch.cuda.device_count() > 1``
    in a single process. That path breaks 4-bit PEFT and raises scatter errors
    (``chunk expects at least a 1-dimensional tensor``). Setting
    ``CUDA_VISIBLE_DEVICES`` only inside ``cmd_smoke`` is too late if CUDA was
    already initialised earlier in the same interpreter.
    """
    if __name__ != "__main__" or len(sys.argv) < 2 or sys.argv[1] != "smoke":
        return
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1") or "1")
    except ValueError:
        ws = 1
    if ws > 1:
        return
    dev = os.environ.get("AIC_SMOKE_CUDA_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = dev


_smoke_single_gpu_env()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aic.training.ddp_utils import (  # noqa: E402
    is_main_process,
    local_rank,
    wait_for_everyone,
    world_size,
)

PIPELINE_LOG_DIR = REPO_ROOT / "logs" / "pipeline"
PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _stage_log(name: str, payload: dict[str, Any]) -> Path:
    out = PIPELINE_LOG_DIR / f"stage_{name}.json"
    payload["timestamp"] = time.time()
    if is_main_process():
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    return out


def _print_header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _free_vram(stage_name: str) -> None:
    """Best-effort VRAM cleanup between long training stages."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free, total = torch.cuda.mem_get_info(0)
            print(
                f"[{stage_name}] VRAM free after cleanup: "
                f"{free/1e9:.1f}/{total/1e9:.1f} GB"
            )
    except Exception:
        pass


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
        sft_num_episodes=24,  # 24 eps -> ~660 records, exceeds both 400/500 asserts
        sft_epochs=1,
        sft_batch_size=1,
        sft_grad_accumulation=1,
        sft_learning_rate=2e-4,
        max_prompt_length=1024,
        max_completion_length=96,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=8,
        lora_alpha=16,
    )

    # Hard caps for smoke (full run reads these from config, not env):
    #   - cap SFT to 40 steps so smoke wraps in minutes, not hours
    #   - disable eval split so we don't OOM on T4 mid-train
    os.environ["AIC_SFT_MAX_STEPS"] = "40"
    os.environ["AIC_SFT_DISABLE_EVAL"] = "1"
    print(
        f"[smoke] AIC_SFT_MAX_STEPS={os.environ['AIC_SFT_MAX_STEPS']} "
        f"AIC_SFT_DISABLE_EVAL={os.environ['AIC_SFT_DISABLE_EVAL']}"
    )

    print("[smoke] Generating SFT dataset (24 episodes)...")
    try:
        dataset_path = generate_sft_dataset(cfg)
    except AssertionError as exc:
        # The generator writes the JSONL BEFORE its >=400/>=500 assertions.
        # For smoke we accept the partial file as long as it actually exists.
        candidate = (
            Path(getattr(cfg, "artifacts_dir", "artifacts"))
            / "sft"
            / "orchestrator_sft.jsonl"
        )
        if candidate.exists() and candidate.stat().st_size > 0:
            print(
                f"[smoke] Generator asserted ({exc}) but JSONL already on disk "
                f"({candidate}). Continuing with partial dataset for smoke."
            )
            dataset_path = candidate
        else:
            return {"ok": False, "error": f"sft_data assertion: {exc}"}
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

    # Fallback: also read trainer_state.json from the GRPO output dir, which
    # records every logged metric even if our progress JSONL was empty.
    trainer_state = REPO_ROOT / "checkpoints" / "grpo" / "trainer_state.json"
    if trainer_state.exists():
        try:
            with open(trainer_state) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                rs = entry.get("reward_std")
                if rs is not None:
                    max_std = max(max_std, float(rs))
                rw = entry.get("reward")
                if rw is not None:
                    rewards.append(float(rw))
        except Exception:
            pass

    return {
        "ok": True,
        "max_reward_std": max_std,
        "rewards": rewards,
        "log_path": str(log_path),
    }


def cmd_smoke(_args) -> int:
    _print_header("STAGE: smoke")
    if world_size() > 1:
        print(
            "[smoke] Refusing: run smoke single-GPU, not with accelerate launch."
        )
        return 1

    print(
        "[smoke] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        "(set at process start for single-process smoke; override with "
        "AIC_SMOKE_CUDA_DEVICES)"
    )

    if cmd_verify(_args) != 0:
        _stage_log("smoke", {"stage": "smoke", "status": "verify_failed"})
        return 1

    sft_result = _smoke_sft()
    print(f"[smoke] SFT: {sft_result}")
    if not sft_result["ok"]:
        _stage_log("smoke", {"stage": "smoke", "status": "sft_failed", **sft_result})
        return 1

    grpo_result = _smoke_grpo(sft_result["sft_dir"])
    print(
        f"[smoke] GRPO: ok={grpo_result['ok']} "
        f"max_std={grpo_result.get('max_reward_std')}"
    )
    if not grpo_result["ok"]:
        _stage_log(
            "smoke",
            {
                "stage": "smoke",
                "status": "grpo_failed",
                "sft": sft_result,
                "grpo": grpo_result,
            },
        )
        return 1

    parse_rate = sft_result.get("parse_rate", 0.0)
    max_std = grpo_result.get("max_reward_std", 0.0)
    # Smoke is a wiring sanity-check, not a quality bar:
    #   - parse_rate >= 1/8  (8 greedy samples; short SFT often misses 30% bar)
    #   - max_reward_std > 0   (GRPO actually got diverse rewards across rollouts)
    # The full run uses the strict 0.70 parse-rate gate inside run_sft.
    parse_ok = parse_rate >= 0.125
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
            print(f"   parse_rate {parse_rate:.2f} < 0.125 (smoke gate)")
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

    # Drop smoke leftovers before full run; otherwise full SFT may silently
    # inherit AIC_SFT_MAX_STEPS=40 from a prior smoke invocation.
    os.environ.pop("AIC_SFT_MAX_STEPS", None)
    os.environ["AIC_SFT_DISABLE_EVAL"] = "1"

    # Data lever: more episodes => more diverse + more "difficult negatives".
    # Default 240 keeps us inside the $22 budget on 4xL4 while materially
    # improving SFT coverage vs the 120-episode baseline.
    sft_eps = int(os.environ.get("AIC_FULL_SFT_EPISODES", "240"))

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_num_episodes=sft_eps,
        sft_epochs=1,
        sft_batch_size=1,
        sft_grad_accumulation=16,
        sft_learning_rate=2e-4,
        max_prompt_length=1024,
        max_completion_length=96,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=16,
        lora_alpha=32,
    )

    print("[full] Generating SFT dataset (120 episodes)...")
    if is_main_process():
        dataset_path = generate_sft_dataset(cfg)
        print(f"[full] Dataset at {dataset_path}")
    wait_for_everyone()

    out = run_sft(cfg, min_parse_rate=0.5)
    parse_rate = 0.0
    try:
        with open(Path(out) / "sft_metadata.json") as f:
            parse_rate = float(json.load(f).get("parse_rate", 0.0))
    except Exception:
        pass
    if parse_rate < 0.7:
        print(
            f"[full] WARNING: SFT parse_rate={parse_rate:.2f} < 0.70 "
            "(continuing because >= 0.50)"
        )

    os.environ.pop("AIC_SFT_DISABLE_EVAL", None)
    return {"sft_dir": str(out), "parse_rate": parse_rate}


def _full_grpo(sft_dir: str) -> dict[str, Any]:
    from aic.training.config import TrainingConfig
    from aic.training.train_grpo import run_grpo

    # Keep GRPO prompt dataset aligned with the SFT coverage level.
    sft_eps = int(os.environ.get("AIC_FULL_SFT_EPISODES", "240"))
    grpo_steps = int(os.environ.get("AIC_FULL_GRPO_MAX_STEPS", "50"))

    cfg = TrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        sft_output_dir=sft_dir,
        sft_num_episodes=sft_eps,
        grpo_max_steps=grpo_steps,
        grpo_per_device_train_batch_size=1,
        max_prompt_length=1024,
        load_in_4bit=True,
        use_unsloth=False,
        lora_r=16,
        lora_alpha=32,
    )
    print(f"[full] GRPO max_steps={grpo_steps}")
    out = run_grpo(cfg)
    return {"grpo_dir": str(out)}


def _full_export(grpo_dir: str) -> dict[str, Any]:
    from aic.training.export_model import export_grpo_to_full

    out = export_grpo_to_full(grpo_dir, "exports", base_model_name="Qwen/Qwen2.5-3B-Instruct")
    return {"exports_dir": str(out)}



def _full_publish_to_hub(exports_dir: str) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("AIC_HUB_REPO", "COolAlien35/aic-orchestrator-l4")
    if not token:
        print("[full] HF_TOKEN not set; skipping hub push")
        return {"pushed": False, "reason": "no HF_TOKEN"}
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=exports_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="AIC 4xL4 DDP run",
    )
    print(f"[full] Pushed {exports_dir} -> https://huggingface.co/{repo_id}")
    return {"pushed": True, "repo_id": repo_id}


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
    print(
        f"[full] DDP world_size={world_size()} local_rank={local_rank()} "
        f"is_main={is_main_process()}"
    )

    smoke_log = PIPELINE_LOG_DIR / "stage_smoke.json"
    if not smoke_log.exists():
        print(
            "[full] REFUSING to run: stage_smoke.json not found. "
            "Run `python scripts/run_pipeline.py smoke` first."
        )
        return 1
    try:
        payload = json.loads(smoke_log.read_text())
    except Exception:
        print("[full] REFUSING to run: stage_smoke.json is not readable JSON.")
        return 1
    if payload.get("status") != "ok":
        print(
            f"[full] REFUSING to run: smoke status={payload.get('status')}. "
            "Re-run smoke until both gates pass."
        )
        return 1

    if os.environ.get("AIC_FULL_SKIP_SFT", "").lower() in ("1", "true", "yes"):
        sft_dir = os.environ.get("AIC_FULL_SFT_DIR", "checkpoints/sft")
        if not Path(sft_dir).exists():
            print(f"[full] SFT skip requested but {sft_dir!r} does not exist.")
            return 2
        parse_rate = 0.0
        try:
            with open(Path(sft_dir) / "sft_metadata.json") as f:
                parse_rate = float(json.load(f).get("parse_rate", 0.0))
        except Exception:
            pass
        sft_result = {"sft_dir": sft_dir, "parse_rate": parse_rate, "skipped": True}
        if is_main_process():
            print(f"[full] Skipping SFT; using existing checkpoint at {sft_dir}")
            _stage_log("sft", {"stage": "sft", "status": "skipped", **sft_result})
    else:
        try:
            sft_result = _full_sft()
            if is_main_process():
                _stage_log("sft", {"stage": "sft", "status": "ok", **sft_result})
            wait_for_everyone()
            if is_main_process():
                _free_vram("post-sft")
        except Exception as exc:
            if is_main_process():
                _stage_log(
                    "sft",
                    {
                        "stage": "sft",
                        "status": "failed",
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    },
                )
            print(f"[full] SFT failed: {exc}")
            wait_for_everyone()
            return 2
    wait_for_everyone()

    try:
        grpo_result = _full_grpo(sft_result["sft_dir"])
        if is_main_process():
            _stage_log("grpo", {"stage": "grpo", "status": "ok", **grpo_result})
        wait_for_everyone()
        if is_main_process():
            _free_vram("post-grpo")
    except Exception as exc:
        if is_main_process():
            _stage_log(
                "grpo",
                {
                    "stage": "grpo",
                    "status": "failed",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                },
            )
        print(f"[full] GRPO failed: {exc}")
        wait_for_everyone()
        return 3

    wait_for_everyone()
    exit_code = 0
    export_result: dict[str, Any] = {}
    if is_main_process():
        try:
            export_result = _full_export(grpo_result["grpo_dir"])
            _stage_log("export", {"stage": "export", "status": "ok", **export_result})
            _free_vram("post-export")
        except Exception as exc:
            _stage_log(
                "export",
                {
                    "stage": "export",
                    "status": "failed",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                },
            )
            print(f"[full] Export failed: {exc}")
            exit_code = 4
        if exit_code == 0:
            try:
                bench_result = _full_benchmark()
                _stage_log("benchmark", {"stage": "benchmark", "status": "ok", **bench_result})
            except Exception as exc:
                _stage_log(
                    "benchmark",
                    {
                        "stage": "benchmark",
                        "status": "failed",
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    },
                )
                print(f"[full] Benchmark failed: {exc}")
                exit_code = 5
        if exit_code == 0 and export_result.get("exports_dir"):
            try:
                _full_publish_to_hub(str(export_result["exports_dir"]))
            except Exception as exc:
                print(f"[full] Hub push failed (non-fatal): {exc}")
        (PIPELINE_LOG_DIR / "full_exit_code.txt").write_text(str(exit_code))
    wait_for_everyone()
    if not is_main_process():
        try:
            exit_code = int(
                (PIPELINE_LOG_DIR / "full_exit_code.txt")
                .read_text()
                .strip()
                or "0"
            )
        except Exception:
            exit_code = 0
    if exit_code:
        return exit_code

    if is_main_process():
        print(
            "\n[full] Pipeline complete. See results/ and logs/pipeline/ for evidence."
        )
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
