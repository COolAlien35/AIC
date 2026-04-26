#!/usr/bin/env python3
"""
Parse a pasted `run_final_benchmark` console log into a benchmark_episodes-style CSV.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# [bench] Benchmarking: baseline_frozen
BENCH_RE = re.compile(r"^\[bench\]\s+Benchmarking:\s+(\S+)")
#   Cache Stampede ep0: reward=-258.62, success=False
EP_RE = re.compile(
    r"^\s+(.+?)\s+ep(\d+):\s*reward=([-\d.]+)\s*,\s*success=(True|False)\s*$"
)
# STATISTICAL TEST lines (optional validation)
#    Baseline avg reward: -252.71
TRAINED_MEAN_RE = re.compile(
    r"Trained avg reward:\s*([-\d.]+)", re.IGNORECASE
)
BASELINE_MEAN_RE = re.compile(
    r"Baseline avg reward:\s*([-\d.]+)", re.IGNORECASE
)


def parse_paste_text(text: str) -> list[dict]:
    rows: list[dict] = []
    current_policy: str | None = None
    for line in text.splitlines():
        m = BENCH_RE.match(line.strip())
        if m:
            current_policy = m.group(1)
            continue
        m = EP_RE.match(line.rstrip())
        if m and current_policy:
            scen, ep_s, r_s, succ_s = m.groups()
            rows.append(
                {
                    "policy": current_policy,
                    "scenario": scen.strip(),
                    "episode_index": int(ep_s),
                    "reward": float(r_s),
                    "success": succ_s == "True",
                }
            )
    return rows


def _optional_extras_for_schema(rows: list[dict]) -> list[dict]:
    for r in rows:
        r.setdefault("mttr", float("nan"))
        r.setdefault("adversary_suppression", float("nan"))
        r.setdefault("unsafe_rate", float("nan"))
        r.setdefault("trained_policy_source", "n/a")
        r.setdefault("trained_policy_checkpoint", "n/a")
    return rows


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    records = _optional_extras_for_schema([dict(x) for x in records])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "policy" in df.columns:
        df = df.sort_values(
            by=["policy", "scenario", "episode_index"], kind="stable"
        ).reset_index(drop=True)
    return df


def extract_embedded_means(text: str) -> tuple[float | None, float | None]:
    """Optional baseline / trained means from a printed [bench] STATISTICAL TEST block."""
    b = b_mean = t_mean = None
    for line in text.splitlines():
        m = BASELINE_MEAN_RE.search(line)
        if m:
            b_mean = float(m.group(1))
        m = TRAINED_MEAN_RE.search(line)
        if m:
            t_mean = float(m.group(1))
    return b_mean, t_mean


def validate_against_means(
    df: pd.DataFrame,
    baseline_expected: float | None,
    trained_expected: float | None,
    rtol: float = 0.01,
) -> list[str]:
    issues: list[str] = []
    if df.empty or not baseline_expected or not trained_expected:
        return issues
    bf = df[df["policy"] == "baseline_frozen"]
    tr = df[df["policy"] == "trained_grpo"]
    if not bf.empty:
        got = float(bf["reward"].mean())
        if not np.isclose(got, baseline_expected, rtol=rtol, atol=0.5):
            issues.append(
                f"baseline_frozen mean {got:.4f} != embedded {baseline_expected} (tolerance {rtol})"
            )
    if not tr.empty:
        got = float(tr["reward"].mean())
        if not np.isclose(got, trained_expected, rtol=rtol, atol=0.5):
            issues.append(
                f"trained_grpo mean {got:.4f} != embedded {trained_expected} (tolerance {rtol})"
            )
    return issues


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("input", type=Path, help="Pasted .log / .txt file")
    p.add_argument("--out", type=Path, help="Output CSV (default: input with .csv)")
    p.add_argument(
        "--strict-embed",
        action="store_true",
        help="Fail if optional embedded means do not match recomputed (when present in text).",
    )
    args = p.parse_args()
    text = args.input.read_text(encoding="utf-8", errors="replace")
    records = parse_paste_text(text)
    if not records:
        print("No episode lines parsed. Check file format.", file=sys.stderr)
        sys.exit(1)
    df = to_dataframe(records)
    out = args.out or args.input.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    b_emb, t_emb = extract_embedded_means(text)
    issues: list[str] = []
    if args.strict_embed and b_emb and t_emb:
        issues = validate_against_means(df, b_emb, t_emb)
    if issues:
        for i in issues:
            print(f"[warn] {i}", file=sys.stderr)
        sys.exit(2)
    df.to_csv(out, index=False)
    print(f"[ok] Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
