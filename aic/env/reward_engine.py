# aic/env/reward_engine.py
"""
Reward engine: computes all 4 reward components per step.

R1 — Health Recovery (dense, every step)
R2 — SLA Bonus (sparse, episode end only)
R3 — Calibrated Trust (per adversarial interaction)
R4 — Explanation Quality (prediction accuracy + causal consistency)
"""
from aic.utils.constants import (
    METRIC_TARGETS, WEIGHT_DB, WEIGHT_INFRA, WEIGHT_APP,
    R2_SLA_BONUS_MAX, SLA_HEALTH_THRESHOLD, SLA_STEPS,
    R3_CORRECT_OVERRIDE, R3_CORRECT_TRUST, R3_WRONG_OVERRIDE, R3_WRONG_TRUST,
    R4_MAX_PER_STEP, R4_MIN_PER_STEP,
    OBS_DB, OBS_INFRA, OBS_APP,
)

DB_METRICS = set(OBS_DB)
INFRA_METRICS = set(OBS_INFRA)
APP_METRICS = set(OBS_APP)
LAYER_WEIGHTS = {"db": WEIGHT_DB, "infra": WEIGHT_INFRA, "app": WEIGHT_APP}


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
    """Orchestrates all reward components per step and per episode."""

    def __init__(self):
        self._prediction_buffer: list[tuple[int, dict[str, float], dict[str, float]]] = []
        self._step_rewards: list[dict] = []

    def reset(self) -> None:
        self._prediction_buffer = []
        self._step_rewards = []

    def compute_step_reward(
        self, step: int, metrics: dict[str, float], prev_metrics: dict[str, float],
        override_applied: bool, adversary_was_correct: bool,
        predicted_2step_impact: dict[str, float], reasoning: str,
        lock_penalty: float = 0.0,
    ) -> dict[str, float]:
        """Compute all reward components for one step. Returns dict with r1..r4, total."""
        r1 = compute_r1(metrics)
        r3 = compute_r3(override_applied, adversary_was_correct)

        r4 = 0.0
        pred_acc = 0.0
        causal_cons = 0.0
        if self._prediction_buffer:
            oldest_step, old_pred, old_metrics = self._prediction_buffer[0]
            if step - oldest_step >= 2:
                self._prediction_buffer.pop(0)
                actual_delta = {k: metrics.get(k, 0.0) - old_metrics.get(k, 0.0) for k in old_pred}
                improving = [m for m, d in actual_delta.items() if d < 0 and (m.endswith("_ms") or m.endswith("_pct"))]
                outcome = f"Metrics improving: {improving}" if improving else "No clear improvement"
                r4, pred_acc, causal_cons = compute_r4(old_pred, actual_delta, reasoning, outcome)

        self._prediction_buffer.append((step, predicted_2step_impact, prev_metrics.copy()))
        total = r1 + r3 + r4 + lock_penalty

        record = {
            "step": step, "r1": r1, "r2": 0.0, "r3": r3, "r4": r4,
            "prediction_accuracy": pred_acc, "causal_consistency": causal_cons,
            "lock_adjustment": lock_penalty, "total": total,
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
