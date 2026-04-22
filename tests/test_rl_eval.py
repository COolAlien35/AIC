def test_evaluate_policy_fn_runs_on_heldout_envs():
    from aic.evals.rl_eval import evaluate_policy_fn
    from aic.training.config import TrainingConfig

    def safe_policy(obs: dict):
        safe = next(
            c for c in obs["candidate_recommendations"]
            if c["agent_name"] == "recovery_verifier"
        )
        return {
            "selected_recommendation_id": safe["recommendation_id"],
            "override_adversary": False,
            "reasoning": "safe baseline",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }

    results = evaluate_policy_fn(safe_policy, TrainingConfig(), num_episodes=2)
    assert len(results) == 2
    assert all(isinstance(r.total_reward, float) for r in results)