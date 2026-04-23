from __future__ import annotations

import importlib
import platform
from importlib import metadata


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def print_dependency_diagnostics() -> None:
    """Print a quick, judge-friendly dependency and backend summary."""
    pkgs = [
        ("python", None),
        ("platform", None),
        ("openenv", "openenv"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("matplotlib", "matplotlib"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("unsloth", "unsloth"),
    ]

    print("Dependency diagnostics")
    print("-" * 60)
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print("-" * 60)
    for display, mod in pkgs[2:]:
        v = _version(display)
        ok = _can_import(mod or display)
        status = "OK" if ok else "MISSING"
        print(f"{display:<12} {status:<7} version={v}")

    # Torch backend info (best-effort; never crash)
    try:
        import torch

        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda = bool(torch.cuda.is_available())
        print("-" * 60)
        print(f"torch.backends: cuda={cuda} mps={mps}")
    except Exception:
        pass

