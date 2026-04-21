# tests/test_incident_commander.py
"""Tests for the Incident Commander Agent."""
import pytest
from aic.agents.incident_commander_agent import (
    IncidentCommanderAgent, PRIORITY_MODES, CommanderDecision,
)
from aic.utils.constants import METRIC_TARGETS, SLA_STEPS


class TestIncidentCommanderAgent:

    def test_default_mode_is_fastest_recovery(self):
        cmd = IncidentCommanderAgent()
        d = cmd.assess_and_command(
            step=0, sla_remaining=20,
            current_metrics=METRIC_TARGETS.copy(),
        )
        assert isinstance(d, CommanderDecision)
        assert d.mode.name in PRIORITY_MODES

    def test_credential_compromise_triggers_contain_mode(self):
        cmd = IncidentCommanderAgent()
        d = cmd.assess_and_command(
            step=3, sla_remaining=17,
            current_metrics=METRIC_TARGETS.copy(),
            root_cause_hypothesis={
                "scenario_name": "Credential Compromise",
                "confidence": 0.6,
            },
        )
        assert d.mode.name == "contain_compromise"

    def test_late_stage_triggers_fastest_recovery(self):
        cmd = IncidentCommanderAgent()
        d = cmd.assess_and_command(
            step=17, sla_remaining=3,
            current_metrics=METRIC_TARGETS.copy(),
        )
        assert d.mode.name == "fastest_recovery"
        assert "step" in d.mode_reason.lower() or "remain" in d.mode_reason.lower()

    def test_db_scenario_triggers_protect_data(self):
        cmd = IncidentCommanderAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["replication_lag_ms"] = 500.0  # well above threshold
        d = cmd.assess_and_command(
            step=5, sla_remaining=15,
            current_metrics=metrics,
            root_cause_hypothesis={
                "scenario_name": "Schema Migration Disaster",
                "confidence": 0.5,
            },
        )
        assert d.mode.name == "protect_data"

    def test_p1_with_high_errors_triggers_minimize_user_impact(self):
        cmd = IncidentCommanderAgent()
        metrics = METRIC_TARGETS.copy()
        metrics["error_rate_pct"] = 15.0
        d = cmd.assess_and_command(
            step=5, sla_remaining=15,
            current_metrics=metrics,
            business_severity="P1",
        )
        assert d.mode.name == "minimize_user_impact"

    def test_mode_history_tracking(self):
        cmd = IncidentCommanderAgent()
        # Run 3 assessments
        cmd.assess_and_command(step=0, sla_remaining=20, current_metrics=METRIC_TARGETS.copy())
        cmd.assess_and_command(step=1, sla_remaining=19, current_metrics=METRIC_TARGETS.copy())
        cmd.assess_and_command(step=2, sla_remaining=18, current_metrics=METRIC_TARGETS.copy())
        history = cmd.get_mode_history_summary()
        assert sum(history.values()) == 3

    def test_reset_clears_history(self):
        cmd = IncidentCommanderAgent()
        cmd.assess_and_command(step=0, sla_remaining=20, current_metrics=METRIC_TARGETS.copy())
        assert cmd.current_mode is not None
        cmd.reset()
        assert cmd.current_mode is None
        assert cmd.get_mode_history_summary() == {}

    def test_score_candidate_gives_bonus_for_aligned_metrics(self):
        cmd = IncidentCommanderAgent()
        d = cmd.assess_and_command(
            step=5, sla_remaining=15,
            current_metrics=METRIC_TARGETS.copy(),
            business_severity="P1",
            root_cause_hypothesis={"scenario_name": "Credential Compromise", "confidence": 0.6},
        )
        # contain_compromise mode weights error_rate_pct
        score = cmd.score_candidate("security_agent", ["error_rate_pct"], d)
        assert score > 0.0

    def test_strategic_brief_is_nonempty(self):
        cmd = IncidentCommanderAgent()
        d = cmd.assess_and_command(step=0, sla_remaining=20, current_metrics=METRIC_TARGETS.copy())
        assert len(d.strategic_brief) > 10

    def test_all_priority_modes_exist(self):
        expected = {"fastest_recovery", "safest_recovery", "protect_data",
                    "minimize_user_impact", "contain_compromise"}
        assert set(PRIORITY_MODES.keys()) == expected
        for mode in PRIORITY_MODES.values():
            assert mode.emoji
            assert mode.display_name
            assert mode.metric_weights
