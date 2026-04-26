# aic/env/aic_environment.py
"""
AIC OpenEnv-compliant environment.

This environment is the single source of truth for episode rollouts:
- generates candidate recommendations,
- applies structured orchestrator actions,
- evolves world state,
- computes verifier-backed rewards,
- logs traces and episode summaries.

The trainable action space is a structured JSON object selecting one of the
candidate recommendations shown in the observation.
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Optional

from openenv.env import Env as OpenEnvBase

from aic.agents.adversarial_agent import AdversarialAgent
from aic.agents.app_agent import AppAgent
from aic.agents.db_agent import DBAgent
from aic.agents.infra_agent import InfraAgent
from aic.agents.network_agent import NetworkAgent
from aic.agents.recovery_verifier_agent import RecoveryVerifierAgent
from aic.env.counterfactual_simulator import simulate_action
from aic.agents.security_agent import SecurityAgent
from aic.env.fault_injector import FaultInjector
from aic.env.lock_manager import ResourceLockManager
from aic.env.scenario_registry import ScenarioEngine
from aic.env.reward_engine import RewardEngine
from aic.env.schema_drift import SchemaDriftInjector
from aic.env.world_state import WorldState
from aic.schemas.actions import (
    CandidateRecommendation,
    OrchestratorDecision,
    ParsedActionResult,
)
from aic.schemas.observations import OrchestratorObservation
from aic.schemas.traces import ExplanationTrace, SubAgentRecommendation
from aic.utils.constants import (
    AGENT_ADV,
    AGENT_APP,
    AGENT_DB,
    AGENT_INFRA,
    AGENT_NET,
    AGENT_SEC,
    AGENT_VERIFIER,
    ALL_AGENTS,
    DRIFT_TYPES,
    INITIAL_TRUST,
    SLA_STEPS,
    TRACE_HISTORY_WINDOW,
    OBS_DB,
    OBS_INFRA,
    OBS_APP,
    OBS_NET,
    OBS_SEC,
)
from aic.utils.logging_utils import EpisodeLogger, StepRecord
from aic.utils.seeding import get_adversary_cycle, get_t_drift, make_episode_rng
from aic.utils.war_room_utils import (
    build_action_deltas,
    build_network_observation,
    build_security_observation,
)


class AICEnvironment(OpenEnvBase):
    """Adaptive Incident Choreographer OpenEnv environment."""

    def __init__(
        self,
        episode_id: int = 0,
        base_seed: int = 42,
        fault_mode: str = "cascading_failure",
        render_mode: Optional[str] = None,
        log_dir: str = "logs",
        drift_type: Optional[str] = None,
        use_llm_agents: bool = False,
        db_agent: Optional[DBAgent] = None,
        infra_agent: Optional[InfraAgent] = None,
        app_agent: Optional[AppAgent] = None,
        net_agent: Optional[NetworkAgent] = None,
        sec_agent: Optional[SecurityAgent] = None,
        include_network: Optional[bool] = None,
        include_security: Optional[bool] = None,
        manage_trust_scores: bool = True,
        scenario_id: Optional[int] = None,
    ):
        state_space = {
            "alert_summary_text": "str (max 5000 chars)",
            "sla_remaining_steps": f"int [0, {SLA_STEPS}]",
            "step": f"int [0, {SLA_STEPS}]",
            "current_metrics": "dict[str, float]",
            "candidate_recommendations": "list[CandidateRecommendation]",
            "sub_agent_recommendations": "list[dict]",
            "current_recommendation_ids": "list[int]",
            "trace_history": "list[dict]",
            "current_trust_scores": "dict[str, float]",
            "schema_drift_active": "bool",
            "schema_drift_type": "str | None",
        }
        action_space = {
            "type": "json_or_text",
            "schema": {
                "selected_recommendation_id": "int",
                "override_adversary": "bool",
                "reasoning": "str",
                "predicted_2step_impact": "dict[str, float]",
                "schema_drift_detected": "bool",
                "schema_drift_field": "str | null",
            },
            "description": "Structured orchestrator decision JSON or legacy text action.",
        }
        super().__init__(
            name="AICEnvironment",
            state_space=state_space,
            action_space=action_space,
            episode_max_length=SLA_STEPS,
        )

        self.episode_id = episode_id
        self.base_seed = base_seed
        self.fault_mode = fault_mode
        self.render_mode = render_mode
        self.log_dir = log_dir
        self.use_llm_agents = use_llm_agents
        self.manage_trust_scores = manage_trust_scores
        self._configured_drift_type = drift_type
        self.scenario_id = scenario_id

        self.db_agent = db_agent or DBAgent(use_llm=use_llm_agents)
        self.infra_agent = infra_agent or InfraAgent(use_llm=use_llm_agents)
        self.app_agent = app_agent or AppAgent(use_llm=use_llm_agents)
        self.include_network = net_agent is not None if include_network is None else include_network
        self.include_security = sec_agent is not None if include_security is None else include_security
        self.net_agent = net_agent or (NetworkAgent(use_llm=use_llm_agents) if self.include_network else None)
        self.sec_agent = sec_agent or (SecurityAgent(use_llm=use_llm_agents) if self.include_security else None)

        self._episode_rng = make_episode_rng(episode_id, base_seed)
        self.world_state = WorldState(self._episode_rng)

        # B1: Use ScenarioEngine when scenario_id is provided, else FaultInjector
        self._scenario_engine: Optional[ScenarioEngine] = None
        if scenario_id is not None:
            from aic.env.scenario_registry import SCENARIO_REGISTRY
            if scenario_id in SCENARIO_REGISTRY:
                self._scenario_engine = ScenarioEngine(scenario_id)
        self.fault_injector = FaultInjector(fault_mode)

        self.reward_engine = RewardEngine()
        self.locks = ResourceLockManager()
        self.logger = EpisodeLogger(log_dir=log_dir, episode_id=episode_id)
        self._verifier = RecoveryVerifierAgent()

        # Competitive scarcity: limited intervention credits per episode.
        self.episode_budget_max: float = 10.0
        self.episode_budget_remaining: float = self.episode_budget_max

        self.step_count = 0
        self.done = False
        self.trace_history: deque[dict[str, Any]] = deque(maxlen=TRACE_HISTORY_WINDOW)
        self.trust_scores: dict[str, float] = {agent: INITIAL_TRUST for agent in ALL_AGENTS}
        self._action_history: list[str] = []
        self._current_recommendations: list[SubAgentRecommendation] = []
        self._candidate_recommendations: list[CandidateRecommendation] = []
        self._candidate_index: dict[int, SubAgentRecommendation] = {}
        self._safe_candidate_id: int = 0
        self._latest_info: dict[str, Any] = {}
        self._drift_type = self._configured_drift_type or DRIFT_TYPES[self.episode_id % len(DRIFT_TYPES)]
        self._schema_drift = SchemaDriftInjector(get_t_drift(self._episode_rng), self._drift_type)
        self._adversary_cycle = get_adversary_cycle(make_episode_rng(self.episode_id, self.base_seed))
        self.adv_agent = AdversarialAgent(self._adversary_cycle, self.db_agent)
        self._active_agents = self._compute_active_agents()

    def _compute_active_agents(self) -> list[str]:
        agents = [
            self.db_agent.agent_name,
            self.infra_agent.agent_name,
            self.app_agent.agent_name,
            self.adv_agent.agent_name,
        ]
        if self.include_network and self.net_agent is not None:
            agents.append(self.net_agent.agent_name)
        if self.include_security and self.sec_agent is not None:
            agents.append(self.sec_agent.agent_name)
        agents.append(AGENT_VERIFIER)
        return agents

    def get_active_agents(self) -> list[str]:
        return list(self._active_agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> dict:
        """Reset the environment for a new episode and return the first observation."""
        options = options or {}
        effective_seed = seed if seed is not None else self.base_seed
        self._episode_rng = make_episode_rng(self.episode_id, effective_seed)

        self._drift_type = (
            options.get("drift_type")
            or self._configured_drift_type
            or DRIFT_TYPES[self.episode_id % len(DRIFT_TYPES)]
        )
        self._schema_drift = SchemaDriftInjector(
            t_drift=get_t_drift(make_episode_rng(self.episode_id, effective_seed)),
            drift_type=self._drift_type,
        )
        self._adversary_cycle = get_adversary_cycle(make_episode_rng(self.episode_id, effective_seed))
        self.adv_agent = AdversarialAgent(self._adversary_cycle, self.db_agent)
        self._active_agents = self._compute_active_agents()

        self.world_state.reset(self._episode_rng)
        # Re-init scenario engine on reset if scenario_id changed via options
        reset_scenario_id = options.get("scenario_id", self.scenario_id)
        if reset_scenario_id is not None:
            from aic.env.scenario_registry import SCENARIO_REGISTRY
            if reset_scenario_id in SCENARIO_REGISTRY:
                self._scenario_engine = ScenarioEngine(reset_scenario_id)
                self.scenario_id = reset_scenario_id
        self.fault_injector = FaultInjector(options.get("fault_mode", self.fault_mode))
        self.reward_engine.reset()
        self.locks.reset()
        self._verifier.reset()
        self.logger = EpisodeLogger(log_dir=self.log_dir, episode_id=self.episode_id)

        self.step_count = 0
        self.done = False
        self.episode_budget_remaining = self.episode_budget_max
        self.trace_history = deque(maxlen=TRACE_HISTORY_WINDOW)
        self.trust_scores = {agent: INITIAL_TRUST for agent in ALL_AGENTS}
        self._action_history = []
        self._latest_info = {}

        self._refresh_candidates()
        return self._get_orchestrator_obs()

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        """Execute one structured orchestrator decision step."""
        if self.done:
            raise RuntimeError("Episode already done. Call reset() before stepping.")

        current_step = self.step_count
        prev_metrics = self.world_state.snapshot()
        parsed = self._parse_action(action)
        decision = parsed.decision

        selected_rec, selection_valid = self._resolve_selected_recommendation(parsed)
        selected_candidate_id = (
            decision.selected_recommendation_id
            if decision is not None else self._safe_candidate_id
        )

        verifier_report = self._verifier.verify(selected_rec, prev_metrics)
        executed_rec = selected_rec if verifier_report.approved else self._verifier.get_safe_minimal_action()

        # Budget gate: even an approved action may be unaffordable.
        budget_blocked = False
        selected_cost = float(getattr(executed_rec, "action_cost", 0.3) or 0.3)
        if selected_cost > self.episode_budget_remaining:
            budget_blocked = True
            executed_rec = self._verifier.get_safe_minimal_action()
            selected_cost = float(getattr(executed_rec, "action_cost", 0.1) or 0.1)
        self.episode_budget_remaining = max(0.0, self.episode_budget_remaining - selected_cost)

        coordination = self._compute_coordination_diagnostics(
            selected_rec=selected_rec,
            executed_rec=executed_rec,
            verifier_approved=verifier_report.approved,
        )
        coordination.update(
            {
                "episode_budget_remaining": float(round(self.episode_budget_remaining, 4)),
                "selected_action_cost": float(round(selected_cost, 4)),
                "budget_blocked": bool(budget_blocked),
            }
        )

        action_deltas = build_action_deltas(executed_rec)
        action_is_noop = len(action_deltas) == 0
        action_repeated = (
            len(self._action_history) > 0
            and self._action_history[-1] == executed_rec.action
        )

        # B1: Use ScenarioEngine contributions when available, else FaultInjector
        if self._scenario_engine is not None:
            faults = self._scenario_engine.get_contributions(current_step)
        else:
            faults = self.fault_injector.get_contributions(current_step)
        self.world_state.step(action_deltas, faults)
        lock_penalty = self.locks.detect_and_resolve_deadlocks()
        current_metrics = self.world_state.snapshot()

        if self.manage_trust_scores:
            self._update_trust_scores(executed_rec.agent_name, prev_metrics, current_metrics)

        override_applied = bool(decision and decision.override_adversary)
        adversary_relevant = (
            selected_rec.agent_name == AGENT_ADV
            or override_applied
            or self._current_adversary_is_top_confidence()
        )
        reward_record = self.reward_engine.compute_step_reward(
            step=current_step,
            metrics=current_metrics,
            prev_metrics=prev_metrics,
            override_applied=override_applied,
            adversary_was_correct=self.adv_agent.was_correct_at_step(current_step),
            predicted_2step_impact=(decision.predicted_2step_impact if decision else {}),
            reasoning=(decision.reasoning if decision else f"Fallback action: {executed_rec.reasoning}"),
            lock_penalty=lock_penalty,
            format_valid=parsed.format_valid,
            selection_valid=selection_valid,
            verifier_approved=verifier_report.approved,
            action_is_noop=action_is_noop,
            action_repeated=action_repeated,
            trust_signal_relevant=adversary_relevant,
        )

        trace = self._build_trace(
            step=current_step,
            decision=decision,
            executed_rec=executed_rec,
            verifier_report=verifier_report.to_dict(),
            override_applied=override_applied,
        )
        self.trace_history.append(trace.model_dump())

        self.step_count += 1
        done = self.step_count >= SLA_STEPS or self.world_state.is_within_sla()
        self.done = done

        r2_bonus = 0.0
        if done:
            steps_remaining = max(0, SLA_STEPS - self.step_count)
            r2_bonus = self.reward_engine.compute_episode_end_reward(current_metrics, steps_remaining)

        returned_reward = reward_record["total"] + r2_bonus
        reward_components = {
            k: v for k, v in reward_record.items() if k not in {"step", "total"}
        }
        reward_components["r2"] = r2_bonus

        record = StepRecord(
            episode_id=self.episode_id,
            step=current_step,
            timestamp=time.time(),
            world_state=current_metrics,
            agent_recommendations={rec.agent_name: rec.action for rec in self._current_recommendations},
            orchestrator_action=executed_rec.action,
            reward_components=reward_components,
            reward_total=returned_reward,
            trust_scores=self.trust_scores.copy(),
            schema_drift_active=self._schema_drift.was_active_at(current_step),
            schema_drift_type=self._drift_type if self._schema_drift.was_active_at(current_step) else None,
            deadlock_detected=lock_penalty < 0,
            extra={
                "selected_candidate_id": selected_candidate_id,
                "active_agents": self.get_active_agents(),
                **coordination,
                "selection_valid": selection_valid,
                "format_valid": parsed.format_valid,
                "parse_error": parsed.parse_error,
                "used_legacy_fallback": parsed.used_legacy_fallback,
                "selected_agent": selected_rec.agent_name,
                "executed_agent": executed_rec.agent_name,
                "verifier_report": verifier_report.to_dict(),
                "trace": trace.model_dump(),
            },
        )
        self.logger.log_step(record)

        info = {
            "step": self.step_count,
            "health": self.world_state.get_health_score(),
            "is_within_sla": self.world_state.is_within_sla(),
            "reward_record": reward_record,
            "r2_bonus": r2_bonus,
            "selected_candidate_id": selected_candidate_id,
            "active_agents": self.get_active_agents(),
            **coordination,
            "selection_valid": selection_valid,
            "format_valid": parsed.format_valid,
            "parse_error": parsed.parse_error,
            "used_legacy_fallback": parsed.used_legacy_fallback,
            "selected_agent": selected_rec.agent_name,
            "executed_agent": executed_rec.agent_name,
            "verifier_report": verifier_report.to_dict(),
            "trace": trace.model_dump(),
            "trust_scores": self.trust_scores.copy(),
            "schema_drift_active": self._schema_drift.was_active_at(current_step),
            "schema_drift_type": self._drift_type if self._schema_drift.was_active_at(current_step) else None,
            "candidate_recommendations": [c.model_dump() for c in self._candidate_recommendations],
            "current_metrics": current_metrics,
        }
        self._latest_info = info

        self._action_history.append(executed_rec.action)
        if not done:
            self._refresh_candidates()
        else:
            self.logger.finalize(
                total_reward=self.reward_engine.get_total_episode_reward(),
                success=self.world_state.is_within_sla(),
            )

        return self._get_orchestrator_obs(), returned_reward, done, info

    def _compute_coordination_diagnostics(
        self,
        selected_rec: SubAgentRecommendation,
        executed_rec: SubAgentRecommendation,
        verifier_approved: bool,
    ) -> dict[str, Any]:
        """
        Lightweight coordination diagnostics for judge-facing evidence.

        These are deliberately simple and stable so they remain meaningful across
        different policy implementations.
        """
        recs = list(self._current_recommendations)
        non_verifier = [r for r in recs if r.agent_name != AGENT_VERIFIER]
        agent_names = [r.agent_name for r in non_verifier]
        target_sets = [tuple(sorted(r.target_metrics)) for r in non_verifier]
        unique_targets = len(set(target_sets)) if target_sets else 0
        conflict_rate = (unique_targets - 1) / max(1, len(target_sets) - 1) if target_sets else 0.0

        adversary_present = any(r.agent_name == AGENT_ADV for r in non_verifier)
        adversary_selected = selected_rec.agent_name == AGENT_ADV
        adversary_overridden = adversary_present and (executed_rec.agent_name != selected_rec.agent_name)

        return {
            "num_candidate_agents": len(non_verifier),
            "candidate_agent_names": agent_names,
            "unique_target_sets": unique_targets,
            "conflict_rate": float(round(conflict_rate, 4)),
            "verifier_approved": bool(verifier_approved),
            "selected_agent_name": selected_rec.agent_name,
            "executed_agent_name": executed_rec.agent_name,
            "adversary_present": bool(adversary_present),
            "adversary_selected": bool(adversary_selected),
            "adversary_overridden": bool(adversary_overridden),
        }

    def _refresh_candidates(self) -> None:
        """Build the recommendation slate for the current state."""
        db_obs_raw = self.world_state.get_db_observation()
        infra_obs_raw = self.world_state.get_infra_observation()
        app_obs_raw = self.world_state.get_app_observation()

        db_obs = self._schema_drift.inject(self.step_count, "db", db_obs_raw)
        app_obs = self._schema_drift.inject(self.step_count, "app", app_obs_raw)

        recommendations = [
            self.db_agent.recommend(db_obs, self.step_count),
            self.infra_agent.recommend(infra_obs_raw, self.step_count),
            self.app_agent.recommend(app_obs, self.step_count),
            self.adv_agent.recommend({**db_obs, **infra_obs_raw, **app_obs}, self.step_count),
        ]

        current_metrics = self.world_state.snapshot()
        if self.include_network and self.net_agent is not None:
            recommendations.append(
                self.net_agent.recommend(build_network_observation(current_metrics), self.step_count)
            )
        if self.include_security and self.sec_agent is not None:
            recommendations.append(
                self.sec_agent.recommend(build_security_observation(current_metrics), self.step_count)
            )

        recommendations.append(self._verifier.get_safe_minimal_action())

        self._current_recommendations = recommendations
        self._candidate_recommendations = []
        self._candidate_index = {}
        for idx, rec in enumerate(recommendations):
            candidate = CandidateRecommendation(
                recommendation_id=idx,
                agent_name=rec.agent_name,
                action=rec.action,
                reasoning=rec.reasoning,
                confidence=rec.confidence,
                target_metrics=rec.target_metrics,
                expected_impact=rec.expected_impact,
                bid=getattr(rec, "bid", 0.0),
                action_cost=getattr(rec, "action_cost", 0.3),
                risk_score=rec.risk_score,
                blast_radius=rec.blast_radius,
                rollback_plan=rec.rollback_plan,
            )
            self._candidate_recommendations.append(candidate)
            self._candidate_index[idx] = rec
            if rec.agent_name == AGENT_VERIFIER:
                self._safe_candidate_id = idx

    def _parse_action(self, action: Any) -> ParsedActionResult:
        """Parse a structured action or legacy text command into a decision."""
        if isinstance(action, OrchestratorDecision):
            return ParsedActionResult(decision=action)

        if isinstance(action, dict):
            try:
                return ParsedActionResult(decision=OrchestratorDecision.model_validate(action))
            except Exception as exc:  # pragma: no cover
                return ParsedActionResult(
                    decision=self._fallback_decision(""),
                    format_valid=False,
                    parse_error=str(exc),
                )

        if isinstance(action, str):
            stripped = action.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    return ParsedActionResult(decision=OrchestratorDecision.model_validate(parsed))
                except Exception as exc:
                    fallback = self._fallback_decision(stripped)
                    return ParsedActionResult(
                        decision=fallback,
                        format_valid=False,
                        parse_error=str(exc),
                        used_legacy_fallback=True,
                    )

            fallback = self._fallback_decision(stripped)
            return ParsedActionResult(
                decision=fallback,
                format_valid=False,
                parse_error="legacy_text_action",
                used_legacy_fallback=True,
            )

        fallback = self._fallback_decision("")
        return ParsedActionResult(
            decision=fallback,
            format_valid=False,
            parse_error=f"unsupported action type: {type(action).__name__}",
        )

    def _fallback_decision(self, text: str) -> OrchestratorDecision:
        """Map legacy natural language actions onto the current recommendation slate."""
        lowered = text.lower().strip()
        if not lowered or any(token in lowered for token in ("noop", "observe", "wait")):
            return OrchestratorDecision(
                selected_recommendation_id=self._safe_candidate_id,
                override_adversary=False,
                reasoning="Fallback to safe minimal action for legacy/no-op command.",
                predicted_2step_impact={},
            )

        def overlap_score(candidate: CandidateRecommendation) -> int:
            words = set(lowered.split())
            candidate_words = set(candidate.action.lower().split()) | {candidate.agent_name.lower()}
            return len(words & candidate_words)

        best = max(self._candidate_recommendations, key=overlap_score, default=None)
        if best is None or overlap_score(best) == 0:
            best_id = self._safe_candidate_id
            reason = "Legacy action could not be matched; using safe minimal action."
        else:
            best_id = best.recommendation_id
            reason = f"Legacy action matched candidate {best_id} via keyword overlap."

        return OrchestratorDecision(
            selected_recommendation_id=best_id,
            override_adversary=False,
            reasoning=reason,
            predicted_2step_impact={},
        )

    def _resolve_selected_recommendation(
        self,
        parsed: ParsedActionResult,
    ) -> tuple[SubAgentRecommendation, bool]:
        decision = parsed.decision
        if decision is None:
            return self._candidate_index[self._safe_candidate_id], False
        rec = self._candidate_index.get(decision.selected_recommendation_id)
        if rec is None:
            return self._candidate_index[self._safe_candidate_id], False
        return rec, True

    def _update_trust_scores(
        self,
        followed_agent: str,
        prev_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> None:
        if followed_agent not in self.trust_scores:
            return
        improvements = 0
        total = 0
        for metric_name, target in self.world_state.targets.items():
            prev_dist = abs(prev_metrics[metric_name] - target)
            curr_dist = abs(current_metrics[metric_name] - target)
            total += 1
            if curr_dist < prev_dist:
                improvements += 1
        outcome_score = 1.0 if total and improvements > total / 2 else 0.0
        old = self.trust_scores[followed_agent]
        self.trust_scores[followed_agent] = max(0.0, min(1.0, old * 0.9 + outcome_score * 0.1))

    def _current_adversary_is_top_confidence(self) -> bool:
        if not self._candidate_recommendations:
            return False
        top = max(self._candidate_recommendations, key=lambda c: c.confidence)
        return top.agent_name == AGENT_ADV

    def _build_trace(
        self,
        step: int,
        decision: Optional[OrchestratorDecision],
        executed_rec: SubAgentRecommendation,
        verifier_report: dict[str, Any],
        override_applied: bool,
    ) -> ExplanationTrace:
        drift_detected = bool(decision.schema_drift_detected) if decision else False
        drift_field = decision.schema_drift_field if decision else None
        return ExplanationTrace(
            step=step,
            followed_agent=executed_rec.agent_name,
            action_taken=executed_rec.action,
            reasoning=(decision.reasoning if decision else executed_rec.reasoning),
            sub_agent_trust_scores=self.trust_scores.copy(),
            override_applied=override_applied,
            override_reason=(
                "Policy explicitly overrode the adversarial recommendation."
                if override_applied else None
            ),
            predicted_2step_impact=(decision.predicted_2step_impact if decision else {}),
            schema_drift_detected=drift_detected,
            schema_drift_field=(drift_field if drift_detected else None),
            verifier_report=verifier_report,
        )

    def _build_alert_summary(self, metrics: dict[str, float]) -> str:
        alerts = []
        for name, value in sorted(metrics.items()):
            target = self.world_state.targets.get(name, 0.0)
            if target == 0.0:
                if value > 0.5:
                    alerts.append(f"ALERT: {name}={value:.1f} (target={target:.1f})")
            else:
                pct_off = abs(value - target) / target * 100
                if pct_off > 10:
                    alerts.append(
                        f"ALERT: {name}={value:.1f} ({pct_off:.0f}% from target {target:.1f})"
                    )
        return "\n".join(alerts) if alerts else "All metrics nominal."

    def _get_orchestrator_obs(self) -> dict:
        metrics = self.world_state.snapshot()
        noisy_health = float(self.world_state.get_health_score() + float(self._episode_rng.normal(0.0, 0.03)))
        shared_noisy_signal = {
            "noisy_health_estimate": round(max(0.0, min(1.0, noisy_health)), 4),
            "rumor": "Network congestion suspected" if metrics.get("net_io_mbps", 0) > 200 else "DB saturation suspected",
        }
        observation_masks = {
            "db_agent": list(OBS_DB),
            "infra_agent": list(OBS_INFRA),
            "app_agent": list(OBS_APP),
            "network_agent": list(OBS_NET),
            "security_agent": list(OBS_SEC),
        }
        visible_recommendations = [
            rec for rec in self._current_recommendations if rec.agent_name != AGENT_VERIFIER
        ]

        # B1: Derive scenario metadata and telemetry corruption from ScenarioEngine
        scenario_name = None
        root_cause_node = None
        telemetry_corruption_active = False
        telemetry_corruption_types: list[str] = []
        telemetry_corruption_fields: list[str] = []
        if self._scenario_engine is not None:
            scenario_name = self._scenario_engine.get_scenario_name()
            root_cause_node = self._scenario_engine.get_root_cause_node()
            mask = self._scenario_engine.get_telemetry_mask(self.step_count)
            renames = self._scenario_engine.get_telemetry_renames(self.step_count)
            shifts = self._scenario_engine.get_telemetry_unit_shifts(self.step_count)
            if mask or renames or shifts:
                telemetry_corruption_active = True
                if mask:
                    telemetry_corruption_types.append("nan_blackout")
                    telemetry_corruption_fields.extend(mask)
                if renames:
                    telemetry_corruption_types.append("field_rename")
                    telemetry_corruption_fields.extend(renames.keys())
                if shifts:
                    telemetry_corruption_types.append("unit_shift")
                    telemetry_corruption_fields.extend(shifts.keys())

        # C3: Derive schema_drift_field from actual drift spec, not hardcoded
        drift_active = self._schema_drift.was_active_at(self.step_count)
        drift_field = self._schema_drift.get_affected_field() if drift_active else None

        obs = OrchestratorObservation(
            alert_summary_text=self._build_alert_summary(metrics),
            sla_remaining_steps=max(0, SLA_STEPS - self.step_count),
            scenario_id=self.scenario_id,
            scenario_name=scenario_name,
            root_cause_node=root_cause_node,
            episode_budget_remaining=float(round(self.episode_budget_remaining, 4)),
            shared_noisy_signal=shared_noisy_signal,
            observation_masks=observation_masks,
            current_metrics=metrics,
            candidate_recommendations=self._candidate_recommendations,
            sub_agent_recommendations=[rec.model_dump() for rec in visible_recommendations],
            current_recommendation_ids=[c.recommendation_id for c in self._candidate_recommendations],
            trace_history=list(self.trace_history),
            current_trust_scores=self.trust_scores.copy(),
            telemetry_corruption_active=telemetry_corruption_active,
            telemetry_corruption_types=telemetry_corruption_types,
            telemetry_corruption_fields=telemetry_corruption_fields,
            schema_drift_active=drift_active,
            schema_drift_type=self._drift_type if drift_active else None,
            schema_drift_field=drift_field,
            step=self.step_count,
        )
        return obs.model_dump()

    def state(self) -> dict:
        """OpenEnv ``state`` accessor.

        Returns the full structured state of the environment as a JSON-serialisable
        dictionary. This is required by the OpenEnv interface (``state_method`` in
        ``openenv.yaml``) and is what the ``GET /state/{env_id}`` HTTP endpoint
        exposes for remote evaluation.
        """
        try:
            metrics = self.world_state.snapshot()
        except Exception:  # pragma: no cover - defensive
            metrics = {}

        try:
            health = float(self.world_state.get_health_score())
        except Exception:  # pragma: no cover
            health = 0.0

        try:
            within_sla = bool(self.world_state.is_within_sla())
        except Exception:  # pragma: no cover
            within_sla = False

        return {
            "episode_id": int(self.episode_id),
            "step": int(self.step_count),
            "sla_remaining_steps": max(0, SLA_STEPS - self.step_count),
            "scenario_id": self.scenario_id,
            "scenario_name": (
                self._scenario_engine.get_scenario_name()
                if self._scenario_engine is not None
                else None
            ),
            "is_done": bool(self.done),
            "is_within_sla": within_sla,
            "health_score": round(health, 4),
            "current_metrics": metrics,
            "trust_scores": dict(self.trust_scores),
            "active_agents": self.get_active_agents(),
            "schema_drift_active": self._schema_drift.was_active_at(self.step_count),
            "schema_drift_type": (
                self._drift_type
                if self._schema_drift.was_active_at(self.step_count)
                else None
            ),
            "episode_budget_remaining": float(round(self.episode_budget_remaining, 4)),
            "candidate_recommendation_ids": [
                c.recommendation_id for c in self._candidate_recommendations
            ],
            "trace_history": list(self.trace_history),
            "drift_type": self._drift_type,
            "fault_mode": self.fault_mode,
        }

    def render(self) -> Optional[str]:
        """Render the current environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        lines = [
            f"=== AIC Environment | Episode {self.episode_id} | Step {self.step_count}/{SLA_STEPS} ===",
            f"Health: {self.world_state.get_health_score():.3f}  SLA: {'OK' if self.world_state.is_within_sla() else 'BREACH'}",
            "",
            "Metrics:",
        ]
        for name in sorted(self.world_state.metrics.keys()):
            current = self.world_state.metrics[name]
            target = self.world_state.targets[name]
            lines.append(f"  {name:25s}  {current:10.2f}  (target: {target:.1f})")

        lines.append("")
        lines.append("Trust Scores:")
        for agent, score in self.trust_scores.items():
            lines.append(f"  {agent:25s}  {score:.3f}")

        lines.append("")
        lines.append("Candidates:")
        for candidate in self._candidate_recommendations:
            lines.append(
                f"  [{candidate.recommendation_id}] {candidate.agent_name}: {candidate.action} "
                f"(conf={candidate.confidence:.2f}, risk={candidate.risk_score:.2f})"
            )
        return "\n".join(lines)
