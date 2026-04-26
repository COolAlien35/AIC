#!/usr/bin/env python3
"""Promote a completed GPU evidence run into the canonical results/ folder.

Run this LOCALLY, after `scripts/gpu_evidence_run.sh` finished on the GPU box
and you've pulled the resulting `evidence/gpu_run/run_<id>/` folder into the
repo. It verifies the manifest, then:

  - Copies the real `benchmark_summary.csv`, `benchmark_by_scenario.csv`,
    `statistical_test.json`, `benchmark_episodes.csv`, `benchmark_run_config.json`
    into `results/` so `results/` is now the authoritative source.
  - Quarantines the synthetic / jittered fixtures
    (`results/benchmark_merged/benchmark_episodes_long*.csv`,
     `data/benchmark_ingest/run*_episodes_from_fixture.csv`)
    by moving them under `archive/synthetic_pre_gpu/` so they cannot be
    confused with real data.
  - Writes `results/statistical_summary.md` with a one-page judge-ready
    summary that points to the evidence folder.

Usage:
    python scripts/finalize_gpu_evidence.py --run evidence/gpu_run/run_<id>
    python scripts/finalize_gpu_evidence.py --run <path> --quarantine
    python scripts/finalize_gpu_evidence.py --run <path> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_FILES = (
    "benchmark_summary.csv",
    "benchmark_by_scenario.csv",
    "benchmark_episodes.csv",
    "benchmark_run_config.json",
    "statistical_test.json",
)

SYNTHETIC_FILES = (
    "results/benchmark_merged/benchmark_episodes_long.csv",
    "results/benchmark_merged/benchmark_episodes_long_normalized.csv",
    "data/benchmark_ingest/run1_episodes_from_fixture.csv",
    "data/benchmark_ingest/run2_episodes_from_fixture.csv",
    "data/benchmark_ingest/fixtures_provenance.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(run_dir: Path) -> tuple[bool, list[str]]:
    manifest = run_dir / "MANIFEST.sha256"
    if not manifest.exists():
        return False, [f"missing {manifest}"]
    failures: list[str] = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            if not rel:
                digest, _, rel = line.partition(" ")
            rel = rel.lstrip("./").strip()
            target = run_dir / rel
            if not target.exists():
                failures.append(f"missing: {rel}")
                continue
            actual = sha256_file(target)
            if actual != digest:
                failures.append(f"hash mismatch: {rel}")
    return len(failures) == 0, failures


def copy_results(run_dir: Path, dest: Path, dry_run: bool) -> list[str]:
    src_results = run_dir / "results"
    if not src_results.exists():
        sys.exit(f"[err] {src_results} does not exist; nothing to promote.")
    dest.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for name in CANONICAL_FILES:
        s = src_results / name
        if not s.exists():
            print(f"  [skip] {s} not present in run bundle")
            continue
        d = dest / name
        moved.append(f"{s} -> {d}")
        if not dry_run:
            shutil.copy2(s, d)
    return moved


def quarantine(items: tuple[str, ...], dry_run: bool) -> list[str]:
    archive_dir = REPO_ROOT / "archive" / "synthetic_pre_gpu"
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
    moves: list[str] = []
    for rel in items:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = archive_dir / src.name
        moves.append(f"{src} -> {dst}")
        if not dry_run:
            shutil.move(str(src), dst)
    if not dry_run and moves:
        readme = archive_dir / "README.md"
        readme.write_text(
            "# Quarantined synthetic fixtures (pre-GPU)\n\n"
            "These files were produced by `scripts/build_benchmark_fixtures.py`,\n"
            "which jitters real per-scenario means with `np.random.normal(0, 0.3)`\n"
            "to fabricate per-episode rows. They are **not** real per-episode\n"
            "rewards. The real GPU benchmark superseded them; they are kept\n"
            "here only for traceability.\n\n"
            "Do not cite any statistic computed from these files in the\n"
            "submission — use `results/benchmark_*` and `evidence/gpu_run/`.\n"
        )
    return moves


def write_summary_md(stats_path: Path, run_dir: Path, dest: Path) -> None:
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text())
    summary_path = dest / "statistical_summary.md"
    sig = "**SIGNIFICANT**" if stats.get("significant") else "not significant"
    rel_run = run_dir.relative_to(REPO_ROOT)
    summary_path.write_text(
        f"""# Trained vs Baseline — Statistical Summary

_Generated by `scripts/finalize_gpu_evidence.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}._

Source: real per-episode benchmark run on GPU.
Provenance: [`{rel_run}`](../{rel_run}/PROVENANCE.md).

| metric | value |
|---|---|
| baseline_mean (frozen) | `{stats.get('baseline_mean')}` |
| trained_mean (GRPO)    | `{stats.get('trained_mean')}` |
| absolute improvement   | `{stats.get('improvement')}` |
| relative improvement   | `{stats.get('improvement_pct')}` % |
| t-statistic            | `{stats.get('t_statistic')}` |
| p-value                | `{stats.get('p_value')}` ({sig}) |
| Cohen's d              | `{stats.get('cohens_d')}` ({stats.get('effect_size_label')}) |

The per-episode CSV (`benchmark_episodes.csv`) and full console log
(`{rel_run}/full_console.log`) are committed alongside this run; every
artifact has a SHA-256 in `{rel_run}/MANIFEST.sha256` so any judge can
verify the chain of custody with one command:

```bash
cd {rel_run} && shasum -a 256 -c MANIFEST.sha256
```
"""
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Path to evidence/gpu_run/run_<id>/")
    p.add_argument("--results", default="results", help="Destination for canonical files")
    p.add_argument("--quarantine", action="store_true",
                   help="Move synthetic/jittered fixtures into archive/synthetic_pre_gpu/")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip MANIFEST.sha256 verification (not recommended)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without modifying any file")
    args = p.parse_args()

    run_dir = (REPO_ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    if not run_dir.exists():
        sys.exit(f"[err] run dir does not exist: {run_dir}")

    print(f"[finalize] using bundle: {run_dir}")

    if not args.no_verify:
        ok, failures = verify_manifest(run_dir)
        if not ok:
            print("[err] MANIFEST verification failed:")
            for f in failures:
                print(f"   - {f}")
            sys.exit(1)
        print("[finalize] MANIFEST.sha256 verified ✓")

    dest = (REPO_ROOT / args.results).resolve()
    moves = copy_results(run_dir, dest, args.dry_run)
    for m in moves:
        print(f"  copy  {m}")

    stats_path = dest / "statistical_test.json"
    if not args.dry_run:
        write_summary_md(stats_path, run_dir, dest)
        print(f"  wrote {dest / 'statistical_summary.md'}")

    if args.quarantine:
        q = quarantine(SYNTHETIC_FILES, args.dry_run)
        if q:
            print("[finalize] quarantined synthetic fixtures:")
            for m in q:
                print(f"  move  {m}")
        else:
            print("[finalize] nothing to quarantine (already clean)")

    print()
    print("[finalize] done. Now:")
    print(f"  cat {dest.relative_to(REPO_ROOT)}/statistical_summary.md")
    print("  # update README headline numbers, commit, push.")


if __name__ == "__main__":
    main()
