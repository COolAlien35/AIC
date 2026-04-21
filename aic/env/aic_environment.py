# aic/env/aic_environment.py
"""
AIC Gymnasium environment.

Wraps WorldState, FaultInjector/ScenarioEngine, ServiceTopology,
BusinessImpactCalculator, and EpisodeLogger into a standard
gymnasium.Env interface.

Phase 8: Supports optional scenario_id for scenario-driven fault injection,
service topology pressure propagation, and business impact calculation.
"""
import math
import time
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aic.utils.constants import (
    SLA_STEPS, INITIAL_TRUST, ALL_AGENTS,
    TRACE_HISTORY_WINDOW,
)
from aic.utils.seeding import make_episode_rng
from aic.utils.logging_utils import EpisodeLogger, StepRecord
from aic.env.world_state import WorldState
from aic.env.fault_injector import FaultInjector
from aic.env.service_topology import ServiceTopology
from aic.env.business_impact import compute_business_impact
from aic.env.scenario_registry import ScenarioEngine


class AICEnvironment(gym.Env):
    """
    Adaptive Incident Choreographer environment.

    Phase 8: topology propagation, business impact, scenario engine.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        episode_id: int = 0,
        base_seed: int = 42,
        fault_mode: str = "cascading_failure",
        scenario_id: Optional[int] = None,
        render_mode: Optional[str] = None,
        log_dir: str = "logs",
    ):
        super().__init__()
        self.episode_id = episode_id
        self.base_seed = base_seed
        self.fault_mode = fault_mode
        self.scenario_id = scenario_id
        self.render_mode = render_mode
        self.log_dir = log_dir

        self._episode_rng = make_episode_rng(episode_id, base_seed)
        self.world_state = WorldState(self._episode_rng)
        self.logger = EpisodeLogger(log_dir=log_dir, episode_id=episode_id)

        # Fault source: ScenarioEngine or FaultInjector
        self._scenario_engine: Optional[ScenarioEngine] = None
        if scenario_id is not None:
            self._scenario_engine = ScenarioEngine(scenario_id)
        self.fault_injector = FaultInjector(fault_mode)

        # Service topology
        self.topology = ServiceTopology()

        self.trust_scores: dict[str, float] = {
            agent: INITIAL_TRUST for agent in ALL_AGENTS
        }
        self.step_count: int = 0
        self.done: bool = False
        self.trace_history: deque = deque(maxlen=TRACE_HISTORY_WINDOW)

        self.action_space = spaces.Text(max_length=2000)
        self.observation_space = spaces.Dict({
            "alert_summary_text": spaces.Text(max_length=5000),
            "sla_remaining_steps": spaces.Discrete(SLA_STEPS + 1),
            "step": spaces.Discrete(SLA_STEPS + 1),
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset for a new episode. options may contain 'scenario_id'."""
        super().reset(seed=seed)

        if seed is not None:
            self._episode_rng = make_episode_rng(self.episode_id, seed)
        else:
            self._episode_rng = make_episode_rng(self.episode_id, self.base_seed)

        self.world_state.reset(self._episode_rng)
        self.logger = EpisodeLogger(log_dir=self.log_dir, episode_id=self.episode_id)

        if options and "scenario_id" in options:
            self.scenario_id = options["scenario_id"]

        if self.scenario_id is not None:
            self._scenario_engine = ScenarioEngine(self.scenario_id)
        else:
            self._scenario_engine = None
        self.fault_injector = FaultInjector(self.fault_mode)

        self.topology.reset()
        self.trust_scores = {agent: INITIAL_TRUST for agent in ALL_AGENTS}
        self.step_count = 0
        self.done = False
        self.trace_history = deque(maxlen=TRACE_HISTORY_WINDOW)

        return self._get_orchestrator_obs(), {}

    def step(
        self, action: str
    ) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Execute one step with topology propagation and business impact."""
        if self.done:
            raise RuntimeError(
                "Episode already done. Call reset() before stepping."
            )

        # 1. Flush topology buffer (apply previous step's ripple effects)
        self.topology.flush_propagation_buffer()

        # 2. Parse action
        action_deltas = self._parse_action(action)

        # 3. Get fault contributions
        if self._scenario_engine is not None:
            fault_contributions = self._scenario_engine.get_contributions(self.step_count)
        else:
            fault_contributions = self.fault_injector.get_contributions(self.step_count)

        # 4. Evolve world state
        prev_metrics = self.world_state.snapshot()
        self.world_state.step(action_deltas, fault_contributions)
        current_metrics = self.world_state.snapshot()

        # 5. Propagate pressure through topology
        self._propagate_metric_changes(prev_metrics, current_metrics)

        # 6. Natural cooldown
        self.topology.cool_down()

        # 7. Business impact
        biz = compute_business_impact(current_metrics)
        biz_dict = {
            "revenue_loss_per_minute": biz.revenue_loss_per_minute,
            "users_impacted": biz.users_impacted,
            "compliance_risk_score": biz.compliance_risk_score,
            "severity_level": biz.severity_level,
        }

        # 8. Advance step
        self.step_count += 1
        terminated = self.step_count >= SLA_STEPS
        self.done = terminated

        # 9. Build observation (with telemetry corruption)
        obs = self._get_orchestrator_obs()

        reward = 0.0
        health = self.world_state.get_health_score()

        # 10. Log step
        record = StepRecord(
            episode_id=self.episode_id,
            step=self.step_count,
            timestamp=time.time(),
            world_state=self.world_state.snapshot(),
            agent_recommendations={},
            orchestrator_action=action,
            reward_components={},
            reward_total=reward,
            trust_scores=self.trust_scores.copy(),
            schema_drift_active=False,
            schema_drift_type=None,
            deadlock_detected=False,
            extra={
                "business_impact_metrics": biz_dict,
                "topology_state": self.topology.get_topology_state(),
            },
        )
        self.logger.log_step(record)

        info = {
            "step": self.step_count,
            "health": health,
            "is_within_sla": self.world_state.is_within_sla(),
            "business_impact": biz_dict,
            "topology_state": self.topology.get_topology_state(),
        }

        if terminated:
            self.logger.finalize(
                total_reward=reward,
                success=self.world_state.is_within_sla(),
            )

        return obs, reward, terminated, False, info

    def _propagate_metric_changes(self, prev: dict, curr: dict) -> None:
        """Map metric deltas to topology pressure and propagate."""
        db_lat_d = curr.get("db_latency_ms", 0) - prev.get("db_latency_ms", 0)
        db_conn_d = curr.get("conn_pool_pct", 0) - prev.get("conn_pool_pct", 0)
        if abs(db_lat_d) > 1.0 or abs(db_conn_d) > 0.5:
            self.topology.propagate_pressure("db", {
                "latency": max(0, db_lat_d),
                "load": max(0, db_conn_d),
            })

        q_d = curr.get("queue_depth", 0) - prev.get("queue_depth", 0)
        if abs(q_d) > 5.0:
            self.topology.propagate_pressure("queue", {
                "load": max(0, q_d * 0.1),
                "latency": max(0, q_d * 0.5),
            })

        err_d = curr.get("error_rate_pct", 0) - prev.get("error_rate_pct", 0)
        app_lat_d = curr.get("p95_latency_ms", 0) - prev.get("p95_latency_ms", 0)
        if abs(err_d) > 0.1 or abs(app_lat_d) > 10.0:
            self.topology.propagate_pressure("app", {
                "error_rate": max(0, err_d),
                "latency": max(0, app_lat_d),
            })

    def _parse_action(self, action: str) -> dict[str, float]:
        """Parse action string into metric deltas. Stub."""
        return {}

    def _get_orchestrator_obs(self) -> dict:
        """Build observation dict with telemetry corruption if scenario active."""
        metrics = self.world_state.snapshot()

        if self._scenario_engine is not None:
            metrics = self._scenario_engine.apply_telemetry_corruption(
                metrics, self.step_count
            )

        alerts = []
        for name, value in sorted(metrics.items()):
            if isinstance(value, float) and math.isnan(value):
                alerts.append(f"ALERT: {name}=UNAVAILABLE (telemetry blackout)")
                continue
            target = self.world_state.targets.get(name, 0.0)
            if target == 0.0:
                if value is not None and value > 0.5:
                    alerts.append(f"ALERT: {name}={value:.1f} (target={target:.1f})")
            else:
                pct_off = abs(value - target) / target * 100
                if pct_off > 10:
                    alerts.append(
                        f"ALERT: {name}={value:.1f} "
                        f"({pct_off:.0f}% from target {target:.1f})"
                    )

        alert_text = "\n".join(alerts) if alerts else "All metrics nominal."

        return {
            "alert_summary_text": alert_text,
            "sla_remaining_steps": SLA_STEPS - self.step_count,
            "sub_agent_recommendations": [],
            "trace_history": list(self.trace_history),
            "current_trust_scores": self.trust_scores.copy(),
            "step": self.step_count,
        }

    def render(self) -> Optional[str]:
        """Render the current environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Build an ANSI text representation of the current state."""
        scn = ""
        if self._scenario_engine:
            scn = f" | Scenario: {self._scenario_engine.get_scenario_name()}"

        lines = [
            f"=== AIC Environment | Episode {self.episode_id} | "
            f"Step {self.step_count}/{SLA_STEPS}{scn} ===",
            f"Health: {self.world_state.get_health_score():.3f}  "
            f"SLA: {'OK' if self.world_state.is_within_sla() else 'BREACH'}",
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
        lines.append("Topology:")
        for node_name, node_state in self.topology.get_topology_state().items():
            lines.append(
                f"  {node_name:10s}  health={node_state['health']:.3f}  "
                f"load={node_state['load']:.1f}  "
                f"latency={node_state['latency']:.1f}"
            )

        return "\n".join(lines)
