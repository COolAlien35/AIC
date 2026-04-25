"""Generate SFT examples from the heuristic orchestrator interacting with AICEnvironment.

Phase 2 upgrade:
  - Uses canonical 6-scenario registry (no ambiguous fault remaps)
  - Rich per-sample metadata tagging (scenario_name, fault_mode, difficulty_tier,
    adversarial_intensity)
  - Difficult negatives: adversarial bad recommendations, conflicting specialist
    outputs, ambiguous states
  - Scenario balancing enforcement
  - Dataset fingerprint at generation time
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.network_agent import NetworkAgent
from aic.agents.orchestrator_agent import OrchestratorAgent
from aic.agents.security_agent import SecurityAgent
from aic.env.aic_environment import AICEnvironment
from aic.env.scenario_registry import SCENARIO_REGISTRY
from aic.training.config import TrainingConfig
from aic.training.prompting import build_orchestrator_prompt, serialize_decision
from aic.training.rollout_env import make_structured_action, materialize_recommendations
from aic.training.scenario_contract import (
    CANONICAL_SCENARIO_IDS,
    SCENARIO_TRAINING_META,
    tag_sample_metadata,
)
from aic.utils.constants import DRIFT_TYPES
from aic.utils.seeding import get_adversary_cycle, make_episode_rng

# Drift types to cycle through for diversity (None = no drift)
ALL_DRIFT_TYPES = [None] + DRIFT_TYPES  # [None, field_rename, unit_shift, silent_null]


def _inject_difficult_negative(
    record: dict[str, Any],
    obs: dict,
    step: int,
    episode_id: int,
) -> dict[str, Any] | None:
    """Optionally create a difficult negative sample from a normal record.

    Difficult negatives include:
      1. Adversarial-selected decisions (bad recommendations the model should learn
         to override)
      2. Conflicting specialist outputs (ambiguous states where multiple agents
         disagree)
      3. Ambiguous states near SLA boundary

    Returns a new record dict if a negative should be generated, else None.
    """
    candidates = obs.get("candidate_recommendations", [])
    if not candidates:
        return None

    # --- Type 1: Adversarial bad recommendation ---
    # Every 5th step+episode, generate a "what if we followed adversary" negative
    if (episode_id + step) % 5 == 0:
        adv_candidates = [c for c in candidates if c.get("agent_name") == "adversarial_agent"]
        if adv_candidates:
            adv = adv_candidates[0]
            # Create a "wrong" completion that selects the adversary
            neg_completion = {
                "selected_recommendation_id": adv["recommendation_id"],
                "override_adversary": False,
                "reasoning": f"[NEGATIVE] Blindly trusting adversarial agent: {adv.get('reasoning', '')}",
                "predicted_2step_impact": {},
                "schema_drift_detected": False,
                "schema_drift_field": None,
            }
            neg_record = record.copy()
            neg_record["completion"] = json.dumps(neg_completion, sort_keys=True)
            neg_record["metadata"] = {
                **record.get("metadata", {}),
                "is_difficult_negative": True,
                "negative_type": "adversarial_bad_recommendation",
            }
            return neg_record

    # --- Type 2: Conflicting specialist (ambiguous state) ---
    # When 3+ agents have confidence > 0.5, it's a genuinely ambiguous state
    if (episode_id + step) % 7 == 0:
        high_conf = [c for c in candidates if c.get("confidence", 0) > 0.5]
        if len(high_conf) >= 3:
            # Pick the lowest-confidence "high confidence" candidate as the wrong choice
            worst = min(high_conf, key=lambda c: c.get("confidence", 0))
            neg_completion = {
                "selected_recommendation_id": worst["recommendation_id"],
                "override_adversary": False,
                "reasoning": f"[NEGATIVE] Choosing weakest of conflicting high-confidence agents",
                "predicted_2step_impact": {},
                "schema_drift_detected": False,
                "schema_drift_field": None,
            }
            neg_record = record.copy()
            neg_record["completion"] = json.dumps(neg_completion, sort_keys=True)
            neg_record["metadata"] = {
                **record.get("metadata", {}),
                "is_difficult_negative": True,
                "negative_type": "conflicting_specialists",
            }
            return neg_record

    # --- Type 3: Ambiguous SLA boundary ---
    sla_remaining = obs.get("sla_remaining_steps", 20)
    if sla_remaining <= 3 and (episode_id + step) % 3 == 0:
        # Near SLA boundary, generate a noop negative
        safe_id = max(c["recommendation_id"] for c in candidates)  # verifier is last
        neg_completion = {
            "selected_recommendation_id": safe_id,
            "override_adversary": False,
            "reasoning": "[NEGATIVE] Passively waiting at SLA boundary instead of acting decisively",
            "predicted_2step_impact": {},
            "schema_drift_detected": False,
            "schema_drift_field": None,
        }
        neg_record = record.copy()
        neg_record["completion"] = json.dumps(neg_completion, sort_keys=True)
        neg_record["metadata"] = {
            **record.get("metadata", {}),
            "is_difficult_negative": True,
            "negative_type": "ambiguous_sla_boundary",
        }
        return neg_record

    return None


def generate_sft_dataset(config: TrainingConfig | None = None) -> Path:
    """Generate JSONL SFT records from heuristic orchestrator rollouts.

    Uses the canonical 6-scenario registry (no fault-mode aliasing).
    Iterates over all scenarios with varied drift types to produce diverse,
    high-quality training data covering adversarial detection, schema drift
    handling, and multi-scenario recovery strategies.

    Each record is tagged with rich metadata for downstream analysis.
    """
    if config is None:
        config = TrainingConfig()

    output_path = Path(config.sft_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBAgent(use_llm=False)
    infra = InfraAgent(use_llm=False)
    app = AppAgent(use_llm=False)
    net = NetworkAgent(use_llm=False)
    sec = SecurityAgent(use_llm=False)

    # Balanced episodes per scenario
    num_scenarios = len(CANONICAL_SCENARIO_IDS)
    episodes_per_scenario = config.sft_num_episodes // num_scenarios

    records: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    episode_id = 0

    for scenario_id in CANONICAL_SCENARIO_IDS:
        scenario = SCENARIO_REGISTRY[scenario_id]
        meta = SCENARIO_TRAINING_META[scenario_id]

        print(f"\n📋 Scenario {scenario_id}: {meta.scenario_name} "
              f"[{meta.difficulty_tier}/{meta.adversarial_intensity}]")

        for ep_local in range(episodes_per_scenario):
            drift_type = ALL_DRIFT_TYPES[ep_local % len(ALL_DRIFT_TYPES)]

            try:
                # Use the mapped FaultInjector mode (not the topology node name)
                env = AICEnvironment(
                    episode_id=episode_id,
                    base_seed=config.base_seed,
                    fault_mode=meta.fault_injector_mode,  # Valid FaultInjector mode
                    drift_type=drift_type,
                    log_dir=config.log_dir,
                    db_agent=db,
                    infra_agent=infra,
                    app_agent=app,
                    net_agent=net,
                    sec_agent=sec,
                    include_network=True,
                    include_security=True,
                    manage_trust_scores=False,
                )
                obs = env.reset()

                cycle = get_adversary_cycle(make_episode_rng(episode_id, config.base_seed))
                adv = AdversarialAgent(cycle, db)
                orch = OrchestratorAgent(adv, use_llm=False)
                orch.mode = "trained"
                orch.reset()

                prev_metrics = obs["current_metrics"]
                done = False
                step = 0

                while not done:
                    recommendations = materialize_recommendations(obs)
                    action, override_applied = orch.decide(
                        step=obs["step"],
                        sla_remaining=obs["sla_remaining_steps"],
                        sub_agent_recommendations=recommendations,
                        alert_summary=obs["alert_summary_text"],
                        prev_metrics=prev_metrics,
                        current_metrics=obs["current_metrics"],
                    )

                    structured_action = make_structured_action(
                        obs,
                        action,
                        getattr(orch, "_followed_agent", None),
                        override_applied,
                    )
                    has_adversarial_candidate = any(
                        rec.get("agent_name") == "adversarial_agent"
                        for rec in obs.get("candidate_recommendations", [])
                    )
                    schema_drift_active = obs.get("schema_drift_active", False)

                    # Build rich metadata tags (Phase 2.2)
                    sample_metadata = tag_sample_metadata(
                        scenario_id=scenario_id,
                        episode_id=episode_id,
                        step=obs["step"],
                        override_applied=override_applied,
                        has_adversarial_candidate=has_adversarial_candidate,
                        schema_drift_active=schema_drift_active,
                        drift_type=drift_type,
                    )

                    record = {
                        "prompt": build_orchestrator_prompt(obs),
                        "completion": serialize_decision(structured_action),
                        "episode_id": episode_id,
                        "step": obs["step"],
                        "scenario": meta.scenario_name,
                        "scenario_id": scenario_id,
                        "drift_type": drift_type,
                        "metadata": sample_metadata,
                    }
                    records.append(record)

                    # Difficult negatives (Phase 2.1 requirement)
                    neg = _inject_difficult_negative(record, obs, step, episode_id)
                    if neg is not None:
                        negatives.append(neg)

                    env.trust_scores = orch.trust_scores.copy()
                    obs, _reward, done, _info = env.step(structured_action)
                    prev_metrics = obs["current_metrics"]
                    step += 1

            except Exception as e:
                print(f"  ⚠️  Episode {episode_id} (scenario {scenario_id}) failed: {e}")

            episode_id += 1

    # Merge negatives into the dataset
    all_records = records + negatives

    # Write to JSONL
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # --- Summary statistics ---
    scenarios_seen = set(r["scenario"] for r in all_records)
    adversarial_count = sum(1 for r in all_records if r["metadata"].get("has_adversarial"))
    drift_count = sum(1 for r in all_records if r["metadata"].get("schema_drift"))
    negative_count = sum(1 for r in all_records if r["metadata"].get("is_difficult_negative"))

    difficulty_tiers = set(r["metadata"].get("difficulty_tier", "?") for r in all_records)

    # Per-scenario counts
    from collections import Counter
    scenario_counts = Counter(r["scenario"] for r in all_records)

    print(f"\n✅ SFT Dataset Generated:")
    print(f"   Total examples:      {len(all_records)} ({len(records)} positive + {len(negatives)} negatives)")
    print(f"   Scenarios covered:   {len(scenarios_seen)}/{num_scenarios} → {scenarios_seen}")
    print(f"   Per-scenario counts: {dict(scenario_counts)}")
    print(f"   Adversarial:         {adversarial_count} ({adversarial_count/max(len(all_records),1)*100:.1f}%)")
    print(f"   Schema drift:        {drift_count} ({drift_count/max(len(all_records),1)*100:.1f}%)")
    print(f"   Difficult negatives: {negative_count}")
    print(f"   Difficulty tiers:    {difficulty_tiers}")

    # --- Dataset fingerprint ---
    hasher = hashlib.sha256()
    for r in all_records:
        hasher.update(json.dumps(r, sort_keys=True).encode())
    fingerprint = hasher.hexdigest()

    fp_record = {
        "fingerprint": fingerprint,
        "total_records": len(all_records),
        "positive_records": len(records),
        "negative_records": len(negatives),
        "scenarios_covered": len(scenarios_seen),
        "episodes": episode_id,
        "config": {
            "base_seed": config.base_seed,
            "sft_num_episodes": config.sft_num_episodes,
            "model_name": config.model_name,
        },
    }
    fp_path = output_path.parent / "generation_fingerprint.json"
    with open(fp_path, "w") as f:
        json.dump(fp_record, f, indent=2)
    print(f"   Fingerprint:         {fingerprint[:16]}... → {fp_path}")

    # --- Hard assertions ---
    assert len(all_records) >= 400, f"FAILED: Only {len(all_records)} examples. Need 400+."
    assert len(all_records) >= 500, f"FAILED: Only {len(all_records)} examples. Need 500+."
    assert len(scenarios_seen) == num_scenarios, f"FAILED: Only {len(scenarios_seen)} scenarios covered."

    return output_path


if __name__ == "__main__":
    path = generate_sft_dataset()
    print(path)
