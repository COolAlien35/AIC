"""DDP-aware utilities for safely guarding file writes and inter-rank syncs.

All file writes outside of HF Trainer / TRL GRPOTrainer (which already handle
their own rank-0 gating) must be wrapped through these helpers, because under
``accelerate launch --num_processes=N`` every rank executes the entire script,
including dataset generation, metadata writes, audit logs, callback writes,
and post-train benchmark / export steps.
"""
from __future__ import annotations

import os


def local_rank() -> int:
    """Return this process's local GPU rank. Defaults to 0 outside DDP."""
    val = os.environ.get("LOCAL_RANK")
    if val is None or val == "":
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


def world_size() -> int:
    val = os.environ.get("WORLD_SIZE")
    if val is None or val == "":
        return 1
    try:
        return int(val)
    except ValueError:
        return 1


def is_main_process() -> bool:
    """True on the global rank-0 process. Falls back to RANK if accelerate
    is not initialised yet (e.g. before Trainer construction).
    """
    try:
        from accelerate import PartialState

        return bool(PartialState().is_main_process)
    except Exception:
        return os.environ.get("RANK", "0") == "0"


def wait_for_everyone() -> None:
    """Synchronise all ranks. No-op outside DDP."""
    try:
        from accelerate import PartialState

        PartialState().wait_for_everyone()
        return
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass
