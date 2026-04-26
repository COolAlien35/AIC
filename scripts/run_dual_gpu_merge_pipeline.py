#!/usr/bin/env python3
"""Optional one-shot: build fixtures from newone → merge → plots."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild data/benchmark_ingest/run*_episodes_from_fixture.csv",
    )
    p.add_argument(
        "--no-newone-ref",
        action="store_true",
        help="Omit newone/statistical_test.json from merged provenance.",
    )
    args = p.parse_args()
    py = sys.executable
    if not args.skip_build:
        subprocess.check_call(
            [py, str(ROOT / "scripts/build_benchmark_fixtures.py")], cwd=ROOT
        )
    merge_cmd = [py, str(ROOT / "scripts/merge_dual_gpu_benchmarks.py")]
    if args.no_newone_ref:
        merge_cmd += ["--ref-newone-stats", str(ROOT / "__no_such_file__.json")]
    subprocess.check_call(merge_cmd, cwd=ROOT)
    subprocess.check_call([py, str(ROOT / "scripts/plot_benchmark_merged.py")], cwd=ROOT)
    print("[ok] Pipeline: build → merge → plots")


if __name__ == "__main__":
    main()
