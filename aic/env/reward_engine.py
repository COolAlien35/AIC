# aic/env/reward_engine.py
"""
Reward engine: computes all reward components per step.

Phase 3 — Competitive Rubric Composition:
  R1 — Health Recovery (dense, every step)
  R2 — SLA Bonus (sparse, episode end only)
  R3 — Calibrated Trust (per adversarial interaction)
  R4 — Explanation Quality (prediction accuracy + causal consistency)
  R5 — Format validity
  R6 — Safety verifier gate
  R7 — Reasoning quality (process-aware)
  R8 — Progress signal (delta toward targets)
  R9 — Overconfidence penalty (penalizes high-confidence wrong predictions)

Hardening features:
  - Dynamic weight schedule (early: safety + format; later: strategic + long-horizon)
  - Reward scale normalization
  - Delayed-return proxy
  - Reward drift tracking
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

from aic.utils.constants import (
    METRIC_TARGETS, WEIGHT_DB, WEIGHT_INFRA, WEIGHT_APP,
    R2_SLA_BONUS_MAX, SLA_HEALTH_THRESHOLD, SLA_STEPS,
    R3_CORRECT_OVERRIDE, R3_CORRECT_TRUST, R3_WRONG_OVERRIDE, R3_WRONG_TRUST,
    R4_MAX_PER_STEP, R4_MIN_PER_STEP,
    R5_VALID_FORMAT, R5_INVALID_FORMAT, R5_INVALID_SELECTION,
    R6_VERIFIER_APPROVED, R6_VERIFIER_VETO,
    NOOP_ACTION_PENALTY, REPEATED_ACTION_PENALTY,
    OBS_DB, OBS_INFRA, OBS_APP,
    R7_REASONING_MAX, R7_REASONING_MIN,
    R8_PROGRESS_MAX, R8_PROGRESS_MIN,
)

DB_METRICS = set(OBS_DB)
INFRA_METRICS = set(OBS_INFRA)
APP_METRICS = set(OBS_APP)
LAYER_WEIGHTS = {"db": WEIGHT_DB, "infra": WEIGHT_INFRA, "app": WEIGHT_APP}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3.1 — Weight Schedule Constants
# ═══════════════════════════════════════════════════════════════════════════

# Step at which weight schedule transitions from "early" (safety-focused)
# to "late" (strategy-focused). Uses a smooth sigmoid transition.
WEIGHT_TRANSITION_MIDPOINT: int = 10
WEIGHT_TRANSITION_STEEPNESS: float = 0.5

# R9 overconfidence penalty constants
R9_OVERCONFIDENCE_MAX_PENALTY: float = -3.0
R9_LOW_INFO_PENALTY: float = -1.0


def compute_r1(metrics: dict[str, float]) -> float:
    """Health Recovery reward — dense, every step."""
    layer_scores: dict[str, float] = {"db": 0.0, "infra": 0.0, "app": 0.0}
    layer_counts: dict[str, int] = {"db": 0, "infra": 0, "app": 0}

    for metric_name, target in METRIC_TARGETS.items():
        if metric_name not in metrics:
            continue
        current = metrics[metric_name]
        if metric_name in DB_METRICS:
            layer = "db"
        elif metric_name in INFRA_METRICS:
            layer = "infra"
        elif metric_name in APP_METRICS:
            layer = "app"
        else:
            continue
        if target == 0.0:
            score = -current
        else:
            score = (target - current) / target
        layer_scores[layer] += score
        layer_counts[layer] += 1

    r1 = 0.0
    for layer, total_score in layer_scores.items():
        count = layer_counts[layer]
        if count > 0:
            r1 += (total_score / count) * LAYER_WEIGHTS[layer]
    return r1


def compute_r2(metrics: dict[str, float], steps_remaining: int, episode_success: bool) -> float:
    """SLA Bonus — sparse, episode end only."""
    if not episode_success:
        return 0.0
    return R2_SLA_BONUS_MAX * (steps_remaining / SLA_STEPS)


def compute_r3(override_applied: bool, adversary_was_correct: bool) -> float:
    """Calibrated Trust reward — 4-case matrix."""
    if override_applied and not adversary_was_correct:
        return R3_CORRECT_OVERRIDE
    elif not override_applied and adversary_was_correct:
        return R3_CORRECT_TRUST
    elif override_applied and adversary_was_correct:
        return R3_WRONG_OVERRIDE
    else:
        return R3_WRONG_TRUST


def compute_r5(format_valid: bool, selection_valid: bool) -> float:
    """Structured action / selection validity reward."""
    if not format_valid:
        return R5_INVALID_FORMAT
    if not selection_valid:
        return R5_INVALID_SELECTION
    return R5_VALID_FORMAT


def compute_r6(verifier_approved: bool) -> float:
    """Safety verifier reward."""
    return R6_VERIFIER_APPROVED if verifier_approved else R6_VERIFIER_VETO


def compute_behavior_penalty(action_is_noop: bool, action_repeated: bool) -> float:
    """Simple anti-stagnation penalties."""
    penalty = 0.0
    if action_is_noop:
        penalty += NOOP_ACTION_PENALTY
    if action_repeated:
        penalty += REPEATED_ACTION_PENALTY
    return penalty


def compute_r9_overconfidence(
    predicted_2step_impact: dict[str, float],
    actual_2step_delta: dict[str, float],
) -> float:
    """Overconfidence penalty — penalizes high-magnitude wrong predictions.

    If the model makes large confident predictions that turn out to be
    directionally wrong (predicted improvement but got degradation, or
    vice versa), it receives a scaled penalty.

    Also penalizes "low information" predictions (empty or near-zero)
    to discourage agents that hedge by predicting nothing.
    """
    if not predicted_2step_impact:
        return R9_LOW_INFO_PENALTY

    penalties = []
    for metric, predicted in predicted_2step_impact.items():
        actual = actual_2step_delta.get(metric, 0.0)
        pred_magnitude = abs(predicted)
        if pred_magnitude < 1e-6:
            continue

        # Directional correctness: did the sign match?
        directionally_wrong = (predicted > 0 and actual < 0) or (predicted < 0 and actual > 0)

        if directionally_wrong:
            # Penalty scales with confidence (magnitude of wrong prediction)
            confidence_scale = min(1.0, pred_magnitude / 100.0)
            penalties.append(R9_OVERCONFIDENCE_MAX_PENALTY * confidence_scale)

    if not penalties:
        return 0.0
    return sum(penalties) / len(penalties)


def get_weight_schedule(step: int) -> dict[str, float]:
    """Dynamic weight schedule for reward components.

    Early steps (0-9): Emphasize safety (R5, R6) and format compliance.
    Late steps (10-19): Emphasize strategic quality (R3, R4) and long-horizon (R8).

    Uses a smooth sigmoid transition to avoid discontinuities.
    """
    # Sigmoid transition: 0.0 (fully early) -> 1.0 (fully late)
    t = 1.0 / (1.0 + math.exp(-WEIGHT_TRANSITION_STEEPNESS * (step - WEIGHT_TRANSITION_MIDPOINT)))

    # Early weights (safety-focused)
    early = {
        "r1": 1.0,   # Health recovery always important
        "r3": 0.3,   # Trust calibration less important early
        "r4": 0.2,   # Explanation quality builds up
        "r5": 1.5,   # Format compliance critical early
        "r6": 1.5,   # Safety verifier critical early
        "r7": 0.5,   # Reasoning quality moderate
        "r8": 0.3,   # Progress signal lower early
        "r9": 0.2,   # Overconfidence penalty lower early
        "behavior": 1.0,
        "lock": 1.0,
    }

    # Late weights (strategic-focused)
    late = {
        "r1": 1.0,   # Health recovery always important
        "r3": 1.5,   # Trust calibration crucial later
        "r4": 1.2,   # Explanation quality matters more
        "r5": 0.5,   # Format compliance less weighted (should be learned)
        "r6": 0.8,   # Safety verifier still important
        "r7": 1.0,   # Reasoning quality important
        "r8": 1.5,   # Long-horizon progress critical
        "r9": 1.0,   # Overconfidence penalty fully active
        "behavior": 1.0,
        "lock": 1.0,
    }

    # Interpolate
    weights = {}
    for key in early:
        weights[key] = early[key] * (1 - t) + late[key] * t
    return weights


def normalize_reward_scale(raw_reward: float, step: int) -> float:
    """Normalize reward to prevent scale drift across training.

    Uses a soft clamp with tanh to keep rewards in a reasonable range
    while preserving gradient signal near zero.
    """
    # Target scale: rewards should typically be in [-20, +20]
    scale = 20.0
    return scale * math.tanh(raw_reward / scale)


def compute_r7_reasoning_quality(
    reasoning: str,
    prev_metrics: dict[str, float],
    current_metrics: dict[str, float],
) -> float:
    """Reasoning trace scorer — rewards intermediate steps that are logically coherent.

    Checks:
    1. Does the reasoning reference the previous state or specific metrics?
    2. Does the reasoning avoid contradictions (claiming improvement when metrics worsened)?
    3. Does the reasoning use causal language?
    """
    score = 0.0
    reasoning_lower = reasoning.lower()

    # Check 1: References specific metric names from the observation
    metric_names = list(METRIC_TARGETS.keys())
    mentioned_metrics = sum(
        1 for m in metric_names
        if m in reasoning_lower or m.replace("_", " ") in reasoning_lower
    )
    if mentioned_metrics > 0:
        score += 0.3 * min(1.0, mentioned_metrics / 3)

    # Check 2: Uses causal / analytical language
    causal_words = [
        "because", "therefore", "since", "due to", "causes", "results in",
        "leads to", "triggers", "in order to", "to reduce", "to improve",
        "correlat", "impact", "affect",
    ]
    causal_count = sum(1 for w in causal_words if w in reasoning_lower)
    if causal_count > 0:
        score += 0.2 * min(1.0, causal_count / 3)

    # Check 3: Consistency — if reasoning claims improvement, check metrics actually improved
    claims_improvement = any(
        w in reasoning_lower
        for w in ["improv", "recover", "reduc", "fix", "resolv", "mitigat"]
    )
    actually_improved = False
    improvements = 0
    total = 0
    for metric, target in METRIC_TARGETS.items():
        if metric in prev_metrics and metric in current_metrics:
            prev_dist = abs(prev_metrics[metric] - target)
            curr_dist = abs(current_metrics[metric] - target)
            total += 1
            if curr_dist < prev_dist:
                improvements += 1
    if total > 0:
        actually_improved = improvements > total / 2

    if claims_improvement and actually_improved:
        score += 0.3  # Consistent claim
    elif claims_improvement and not actually_improved:
        score -= 0.2  # Contradicted by reality
    elif not claims_improvement and len(reasoning) > 20:
        score += 0.1  # At least non-vacuous

    # Check 4: Minimum length — penalize vacuous one-word reasoning
    if len(reasoning.strip()) < 15:
        score -= 0.3

    # Scale to [R7_MIN, R7_MAX]
    normalized = max(0.0, min(1.0, (score + 0.3) / 1.1))  # map [-0.3, 0.8] to [0, 1]
    return R7_REASONING_MIN + normalized * (R7_REASONING_MAX - R7_REASONING_MIN)


def compute_r8_progress_signal(
    prev_metrics: dict[str, float],
    current_metrics: dict[str, float],
) -> float:
    """Progress signal — partial credit for getting closer to targets.

    Unlike R1 which measures absolute health, R8 measures the *delta*:
    did this step move us closer to or further from the targets?
    """
    if not prev_metrics or not current_metrics:
        return 0.0

    progress_scores = []
    for metric, target in METRIC_TARGETS.items():
        if metric not in prev_metrics or metric not in current_metrics:
            continue
        prev_dist = abs(prev_metrics[metric] - target)
        curr_dist = abs(current_metrics[metric] - target)
        if prev_dist < 1e-6:
            # Already at target, small bonus for maintaining
            progress_scores.append(0.5 if curr_dist < 1e-6 else -0.5)
        else:
            # Fractional improvement: 1.0 = closed the gap completely, -1.0 = doubled the gap
            improvement = (prev_dist - curr_dist) / prev_dist
            progress_scores.append(max(-1.0, min(1.0, improvement)))

    if not progress_scores:
        return 0.0

    mean_progress = sum(progress_scores) / len(progress_scores)
    # Scale to [R8_MIN, R8_MAX]
    normalized = (mean_progress + 1.0) / 2.0  # map [-1, 1] to [0, 1]
    return R8_PROGRESS_MIN + normalized * (R8_PROGRESS_MAX - R8_PROGRESS_MIN)


def compute_r4(
    predicted_2step_impact: dict[str, float],
    actual_2step_delta: dict[str, float],
    reasoning: str,
    actual_outcome_summary: str,
) -> tuple[float, float, float]:
    """Explanation Quality reward. Returns (r4_total, pred_accuracy, causal_consistency)."""
    # Component 1: Prediction Accuracy
    prediction_scores = []
    for metric, predicted_delta in predicted_2step_impact.items():
        actual_delta = actual_2step_delta.get(metric, 0.0)
        error = abs(predicted_delta - actual_delta)
        scale = abs(actual_delta) + 1e-6
        score = max(0.0, 1.0 - error / scale)
        prediction_scores.append(score)
    prediction_accuracy = (
        sum(prediction_scores) / len(prediction_scores) if prediction_scores else 0.5
    )

    # Component 2: Causal Consistency (heuristic)
    causal_score = 0.0
    reasoning_lower = reasoning.lower()
    outcome_lower = actual_outcome_summary.lower()

    causal_words = ["because", "therefore", "causes", "results in", "leads to", "due to", "triggers"]
    if any(w in reasoning_lower for w in causal_words):
        causal_score += 0.3

    changed_metrics = [m for m, d in actual_2step_delta.items() if abs(d) > 0.01]
    mentioned = sum(1 for m in changed_metrics if m.replace("_", " ") in reasoning_lower or m in reasoning_lower)
    if mentioned > 0:
        causal_score += 0.2 * min(1.0, mentioned / max(1, len(changed_metrics)))

    if "improv" in outcome_lower or "recover" in outcome_lower:
        if any(w in reasoning_lower for w in ("improv", "reduc", "fix", "recover")):
            causal_score += 0.5

    causal_consistency = min(1.0, causal_score)

    combined_0_1 = 0.5 * prediction_accuracy + 0.5 * causal_consistency
    r4 = R4_MIN_PER_STEP + combined_0_1 * (R4_MAX_PER_STEP - R4_MIN_PER_STEP)
    return r4, prediction_accuracy, causal_consistency


class RewardEngine:
    """Orchestrates all reward components per step and per episode.

    Phase 3 hardening:
      - Dynamic weight schedule (early=safety, late=strategic)
      - R9 overconfidence penalty
      - Reward normalization via soft clamp
      - Reward drift tracking (running mean/std)
      - Delayed-return proxy for R4
    """

    def __init__(self, use_weight_schedule: bool = True, use_normalization: bool = True):
        self._prediction_buffer: list[tuple[int, dict[str, float], dict[str, float]]] = []
        self._step_rewards: list[dict] = []
        self._use_weight_schedule = use_weight_schedule
        self._use_normalization = use_normalization

        # Reward drift tracker (exponential moving average)
        self._reward_ema: float = 0.0
        self._reward_ema_sq: float = 0.0
        self._reward_ema_alpha: float = 0.1
        self._reward_drift_log: list[dict[str, float]] = []

    def reset(self) -> None:
        self._prediction_buffer = []
        self._step_rewards = []
        self._reward_drift_log = []
        # Note: EMA is NOT reset between episodes to track cross-episode drift

    def _track_reward_drift(self, step: int, raw_total: float) -> dict[str, float]:
        """Update exponential moving average of rewards and log drift."""
        alpha = self._reward_ema_alpha
        self._reward_ema = alpha * raw_total + (1 - alpha) * self._reward_ema
        self._reward_ema_sq = alpha * (raw_total ** 2) + (1 - alpha) * self._reward_ema_sq
        variance = max(0.0, self._reward_ema_sq - self._reward_ema ** 2)
        std = math.sqrt(variance)

        drift_record = {
            "step": step,
            "raw_reward": raw_total,
            "ema_mean": self._reward_ema,
            "ema_std": std,
        }
        self._reward_drift_log.append(drift_record)
        return drift_record

    def compute_step_reward(
        self, step: int, metrics: dict[str, float], prev_metrics: dict[str, float],
        override_applied: bool, adversary_was_correct: bool,
        predicted_2step_impact: dict[str, float], reasoning: str,
        lock_penalty: float = 0.0,
        format_valid: bool = True,
        selection_valid: bool = True,
        verifier_approved: bool = True,
        action_is_noop: bool = False,
        action_repeated: bool = False,
        trust_signal_relevant: bool = True,
        enable_process_feedback: bool = True,
    ) -> dict[str, float]:
        """Compute all reward components for one step.

        Returns dict with r1..r9, weight schedule info, and total.
        """
        r1 = compute_r1(metrics)
        r3 = compute_r3(override_applied, adversary_was_correct) if trust_signal_relevant else 0.0

        r4 = 0.0
        r9 = 0.0
        pred_acc = 0.0
        causal_cons = 0.0
        actual_delta: dict[str, float] = {}
        if self._prediction_buffer:
            oldest_step, old_pred, old_metrics = self._prediction_buffer[0]
            if step - oldest_step >= 2:
                self._prediction_buffer.pop(0)
                actual_delta = {k: metrics.get(k, 0.0) - old_metrics.get(k, 0.0) for k in old_pred}
                improving = [m for m, d in actual_delta.items() if d < 0 and (m.endswith("_ms") or m.endswith("_pct"))]
                outcome = f"Metrics improving: {improving}" if improving else "No clear improvement"
                r4, pred_acc, causal_cons = compute_r4(old_pred, actual_delta, reasoning, outcome)
                # R9 overconfidence penalty (uses the same delayed delta)
                r9 = compute_r9_overconfidence(old_pred, actual_delta)

        r5 = compute_r5(format_valid, selection_valid)
        r6 = compute_r6(verifier_approved)
        behavior_penalty = compute_behavior_penalty(action_is_noop, action_repeated)

        # Process-aware feedback (R7 + R8) — togglable via config
        r7 = 0.0
        r8 = 0.0
        if enable_process_feedback:
            r7 = compute_r7_reasoning_quality(reasoning, prev_metrics, metrics)
            r8 = compute_r8_progress_signal(prev_metrics, metrics)

        self._prediction_buffer.append((step, predicted_2step_impact, prev_metrics.copy()))

        # --- Phase 3: Apply dynamic weight schedule ---
        if self._use_weight_schedule:
            weights = get_weight_schedule(step)
        else:
            weights = {k: 1.0 for k in ["r1","r3","r4","r5","r6","r7","r8","r9","behavior","lock"]}

        raw_total = (
            r1 * weights["r1"]
            + r3 * weights["r3"]
            + r4 * weights["r4"]
            + r5 * weights["r5"]
            + r6 * weights["r6"]
            + r7 * weights["r7"]
            + r8 * weights["r8"]
            + r9 * weights["r9"]
            + lock_penalty * weights["lock"]
            + behavior_penalty * weights["behavior"]
        )

        # --- Phase 3: Normalize reward scale ---
        if self._use_normalization:
            total = normalize_reward_scale(raw_total, step)
        else:
            total = raw_total

        # --- Phase 3: Track reward drift ---
        drift = self._track_reward_drift(step, raw_total)

        record = {
            "step": step, "r1": r1, "r2": 0.0, "r3": r3, "r4": r4,
            "r5": r5, "r6": r6, "r7": r7, "r8": r8, "r9": r9,
            "prediction_accuracy": pred_acc, "causal_consistency": causal_cons,
            "lock_adjustment": lock_penalty,
            "behavior_penalty": behavior_penalty,
            "format_valid": format_valid,
            "selection_valid": selection_valid,
            "verifier_approved": verifier_approved,
            "trust_signal_relevant": trust_signal_relevant,
            "raw_total": raw_total,
            "total": total,
            "reward_ema": drift["ema_mean"],
            "reward_std": drift["ema_std"],
        }
        self._step_rewards.append(record)
        return record

    def record_r2_bonus(self, r2: float) -> None:
        """Record R2 bonus (can be called mid-episode on early SLA success)."""
        self._r2_bonus = r2

    def compute_episode_end_reward(self, metrics: dict[str, float], steps_remaining: int) -> float:
        """Compute R2 at episode end."""
        episode_success = all(
            abs(metrics.get(m, 0) - METRIC_TARGETS[m]) / max(METRIC_TARGETS[m], 1e-6) <= SLA_HEALTH_THRESHOLD
            for m in METRIC_TARGETS
        )
        r2 = compute_r2(metrics, steps_remaining, episode_success)
        self._r2_bonus = r2
        return r2

    def get_total_episode_reward(self) -> float:
        return sum(r["total"] for r in self._step_rewards) + getattr(self, '_r2_bonus', 0.0)

    def get_reward_history(self) -> list[dict]:
        return self._step_rewards.copy()

    def get_reward_drift_log(self) -> list[dict[str, float]]:
        """Return the per-step reward drift tracking log."""
        return self._reward_drift_log.copy()

    def get_reward_drift_summary(self) -> dict[str, float]:
        """Summary statistics for reward drift detection."""
        if not self._reward_drift_log:
            return {"mean": 0.0, "std": 0.0, "max_deviation": 0.0}
        means = [d["ema_mean"] for d in self._reward_drift_log]
        stds = [d["ema_std"] for d in self._reward_drift_log]
        raw = [d["raw_reward"] for d in self._reward_drift_log]
        return {
            "mean": sum(means) / len(means),
            "std": sum(stds) / len(stds),
            "max_deviation": max(abs(r - means[-1]) for r in raw) if raw else 0.0,
            "final_ema": means[-1] if means else 0.0,
            "num_steps": len(self._reward_drift_log),
        }
