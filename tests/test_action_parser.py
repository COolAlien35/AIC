import pytest


def test_valid_structured_action_executes_candidate():
    from aic.env.aic_environment import AICEnvironment

    env = AICEnvironment(episode_id=0, log_dir="/tmp/aic_action_parser")
    obs = env.reset()
    first_candidate = obs["candidate_recommendations"][0]

    next_obs, reward, done, info = env.step(
        {
            "selected_recommendation_id": first_candidate["recommendation_id"],
            "override_adversary": False,
            "reasoning": "selecting first candidate for validation",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }
    )

    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["format_valid"] is True
    assert info["selection_valid"] is True
    assert info["selected_candidate_id"] == first_candidate["recommendation_id"]


def test_invalid_selection_falls_back_to_safe_action():
    from aic.env.aic_environment import AICEnvironment

    env = AICEnvironment(episode_id=0, log_dir="/tmp/aic_action_parser")
    env.reset()
    _, reward, _, info = env.step(
        {
            "selected_recommendation_id": 999,
            "override_adversary": False,
            "reasoning": "invalid selection",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }
    )

    assert info["selection_valid"] is False
    assert info["selected_agent"] == "recovery_verifier"
    assert reward < 0.0


def test_legacy_text_action_marks_format_invalid():
    from aic.env.aic_environment import AICEnvironment

    env = AICEnvironment(episode_id=0, log_dir="/tmp/aic_action_parser")
    env.reset()
    _, _, _, info = env.step("observe current state")

    assert info["format_valid"] is False
    assert info["used_legacy_fallback"] is True
