import copy

from aic.utils.constants import METRIC_FAULT_INIT


def test_invalid_format_is_penalized_more_than_valid_format():
    from aic.env.reward_engine import RewardEngine

    engine = RewardEngine()
    metrics = copy.deepcopy(METRIC_FAULT_INIT)

    valid = engine.compute_step_reward(
        step=0,
        metrics=metrics,
        prev_metrics=metrics,
        override_applied=False,
        adversary_was_correct=True,
        predicted_2step_impact={},
        reasoning="valid reasoning",
        format_valid=True,
        selection_valid=True,
    )
    invalid = engine.compute_step_reward(
        step=1,
        metrics=metrics,
        prev_metrics=metrics,
        override_applied=False,
        adversary_was_correct=True,
        predicted_2step_impact={},
        reasoning="invalid reasoning",
        format_valid=False,
        selection_valid=True,
    )

    assert invalid["r5"] < valid["r5"]


def test_verifier_veto_penalty_is_negative():
    from aic.env.reward_engine import compute_r6

    assert compute_r6(True) > 0.0
    assert compute_r6(False) < 0.0


def test_repeated_and_noop_behavior_penalties_stack():
    from aic.env.reward_engine import compute_behavior_penalty

    assert compute_behavior_penalty(False, False) == 0.0
    assert compute_behavior_penalty(True, False) < 0.0
    assert compute_behavior_penalty(False, True) < 0.0
    assert compute_behavior_penalty(True, True) < compute_behavior_penalty(True, False)


def test_irrelevant_trust_signal_zeroes_r3():
    from aic.env.reward_engine import RewardEngine

    engine = RewardEngine()
    metrics = copy.deepcopy(METRIC_FAULT_INIT)
    record = engine.compute_step_reward(
        step=0,
        metrics=metrics,
        prev_metrics=metrics,
        override_applied=True,
        adversary_was_correct=False,
        predicted_2step_impact={},
        reasoning="test",
        trust_signal_relevant=False,
    )
    assert record["r3"] == 0.0
