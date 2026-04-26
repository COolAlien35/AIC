#!/usr/bin/env python3
"""Generate rubric-mandated training-evidence plots from a real GRPO run.

Reads ``logs/grpo_progress.jsonl`` (emitted by
``aic.training.train_grpo.AICProgressCallback``) and writes:

  * ``results/grpo_reward_curve.png`` - reward (and +/- std band) vs step
  * ``results/grpo_loss_curve.png``   - training loss vs step
  * ``results/grpo_kl_curve.png``     - KL vs step (sanity check)
  * ``results/grpo_training_summary.json`` - {total_steps, initial_reward,
    final_reward, reward_delta, max_reward_std, training_time_minutes}

Filters the trailing ``step:80, reward:0`` sentinel that the callback emits
on training-end, and de-duplicates per-step rows by keeping the first
occurrence of each ``step``.

Usage::

    python scripts/plot_grpo_progress.py
    python scripts/plot_grpo_progress.py --log logs/grpo_progress.jsonl --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DARK_BG = "#0C0F0A"
PANEL_BG = "#111610"
TEXT = "#E8EDE6"
SUBTEXT = "#9CA89A"
TICK = "#6B7A68"
GREEN = "#34D399"
ORANGE = "#F59E0B"
BLUE = "#60A5FA"
RED = "#F87171"


def _load_progress(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _clean(rows: list[dict]) -> list[dict]:
    """Drop the training-end sentinel (reward=0, loss=0, kl=0 row) and de-dup steps."""
    seen: dict[int, dict] = {}
    for r in rows:
        if not isinstance(r, dict) or "step" not in r:
            continue
        is_sentinel = (
            r.get("reward", 0) == 0
            and r.get("loss", 0) == 0
            and r.get("kl", 0) == 0
            and r.get("completion_length", 0) == 0
        )
        if is_sentinel:
            continue
        step = int(r["step"])
        if step not in seen:
            seen[step] = r
    return [seen[s] for s in sorted(seen.keys())]


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.spines["bottom"].set_color(TICK)
    ax.spines["left"].set_color(TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TICK)
    ax.grid(True, alpha=0.15, color=TICK, linestyle="--")
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)


def plot_reward_curve(rows: list[dict], out: Path) -> None:
    steps = np.array([r["step"] for r in rows])
    rewards = np.array([r.get("reward", 0.0) for r in rows], dtype=float)
    stds = np.array([r.get("reward_std", 0.0) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)

    ax.fill_between(
        steps,
        rewards - stds,
        rewards + stds,
        color=GREEN,
        alpha=0.18,
        label="reward +/- std",
    )
    ax.plot(steps, rewards, color=GREEN, linewidth=2.2, marker="o", markersize=4, label="reward (mean)")

    if len(rewards) >= 5:
        window = max(3, min(7, len(rewards) // 5))
        kernel = np.ones(window) / window
        smooth = np.convolve(rewards, kernel, mode="valid")
        smooth_steps = steps[window - 1 :]
        ax.plot(
            smooth_steps,
            smooth,
            color=ORANGE,
            linewidth=1.6,
            linestyle="--",
            label=f"moving avg (w={window})",
        )

    initial = float(rewards[0])
    final = float(rewards[-1])
    delta = final - initial
    ax.set_title(
        f"GRPO training reward (real run, {len(rewards)} steps)  -  "
        f"reward {initial:+.2f} -> {final:+.2f}  (delta {delta:+.2f})",
        color=TEXT,
        fontsize=13,
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.legend(facecolor=PANEL_BG, edgecolor=GREEN, labelcolor=TEXT, loc="lower right")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close()


def plot_loss_curve(rows: list[dict], out: Path) -> None:
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r.get("loss", 0.0) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(steps, loss, color=BLUE, linewidth=2.0, marker="o", markersize=3.5, label="loss")
    ax.set_title(
        f"GRPO training loss (real run, {len(loss)} steps)",
        color=TEXT,
        fontsize=13,
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend(facecolor=PANEL_BG, edgecolor=BLUE, labelcolor=TEXT, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close()


def plot_kl_curve(rows: list[dict], out: Path) -> None:
    steps = np.array([r["step"] for r in rows])
    kl = np.array([r.get("kl", 0.0) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(steps, kl, color=RED, linewidth=2.0, marker="o", markersize=3.5, label="KL divergence")
    ax.set_title(
        f"GRPO KL vs reference policy (real run, {len(kl)} steps)",
        color=TEXT,
        fontsize=13,
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("KL")
    ax.legend(facecolor=PANEL_BG, edgecolor=RED, labelcolor=TEXT, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close()


def write_summary(rows: list[dict], out: Path) -> dict:
    rewards = [r.get("reward", 0.0) for r in rows]
    losses = [r.get("loss", 0.0) for r in rows]
    stds = [r.get("reward_std", 0.0) for r in rows]
    times = [r.get("elapsed_minutes", 0.0) for r in rows]
    summary = {
        "total_steps": len(rows),
        "initial_reward": float(rewards[0]) if rewards else 0.0,
        "final_reward": float(rewards[-1]) if rewards else 0.0,
        "reward_delta": float(rewards[-1] - rewards[0]) if rewards else 0.0,
        "min_reward": float(min(rewards)) if rewards else 0.0,
        "max_reward": float(max(rewards)) if rewards else 0.0,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "max_reward_std": float(max(stds)) if stds else 0.0,
        "training_time_minutes": float(times[-1]) if times else 0.0,
        "source_log": "logs/grpo_progress.jsonl",
        "framework": "TRL GRPOTrainer + Unsloth",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/grpo_progress.jsonl")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    if not log_path.exists():
        raise SystemExit(f"GRPO progress log not found: {log_path}")

    rows = _clean(_load_progress(log_path))
    if not rows:
        raise SystemExit(f"No usable rows in {log_path} after filtering sentinel/duplicates")

    plot_reward_curve(rows, out_dir / "grpo_reward_curve.png")
    plot_loss_curve(rows, out_dir / "grpo_loss_curve.png")
    plot_kl_curve(rows, out_dir / "grpo_kl_curve.png")
    summary = write_summary(rows, out_dir / "grpo_training_summary.json")

    print(f"[ok] wrote {out_dir}/grpo_reward_curve.png")
    print(f"[ok] wrote {out_dir}/grpo_loss_curve.png")
    print(f"[ok] wrote {out_dir}/grpo_kl_curve.png")
    print(f"[ok] wrote {out_dir}/grpo_training_summary.json")
    print(f"     steps={summary['total_steps']}  "
          f"reward {summary['initial_reward']:+.2f} -> {summary['final_reward']:+.2f}  "
          f"(delta {summary['reward_delta']:+.2f}, max_std {summary['max_reward_std']:.2f})  "
          f"time={summary['training_time_minutes']:.1f}m")


if __name__ == "__main__":
    main()
