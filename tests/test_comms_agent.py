# tests/test_comms_agent.py
"""Tests for the Comms/Postmortem Agent."""
import pytest
from aic.agents.comms_agent import CommsAgent, PostmortemReport


class TestCommsAgent:

    def _make_traces(self, n: int = 5) -> list[dict]:
        return [
            {
                "step": i,
                "health": 0.3 + i * 0.1,
                "trace": {
                    "step": i,
                    "action_taken": f"Action at step {i}",
                    "followed_agent": "db_agent",
                    "reasoning": f"Reasoning for step {i}",
                    "override_applied": i == 2,
                    "override_reason": "trust threshold breach" if i == 2 else None,
                    "root_cause_hypothesis": {
                        "scenario_name": "Cache Stampede",
                        "confidence": 0.5 + i * 0.05,
                    } if i >= 2 else None,
                    "verifier_report": {"approved": True, "risk_score": 0.3},
                },
            }
            for i in range(n)
        ]

    def test_generate_postmortem_returns_report(self):
        comms = CommsAgent()
        traces = self._make_traces()
        report = comms.generate_postmortem(
            scenario_name="Cache Stampede",
            episode_traces=traces,
            final_health=0.75,
            mttr_steps=8,
        )
        assert isinstance(report, PostmortemReport)
        assert report.scenario_name == "Cache Stampede"
        assert report.final_health == 0.75
        assert report.mttr_steps == 8
        assert report.sla_met is True

    def test_executive_summary_nonempty(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Canary Failure",
            episode_traces=self._make_traces(),
            final_health=0.6,
            mttr_steps=12,
        )
        assert len(report.executive_summary) > 50
        assert "Canary Failure" in report.executive_summary

    def test_customer_safe_no_technical_jargon(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Regional Outage",
            episode_traces=self._make_traces(),
            final_health=0.4,
            mttr_steps=20,
        )
        text = report.customer_safe_summary.lower()
        # Should NOT contain internal terms
        assert "orchestrator" not in text
        assert "bayesian" not in text
        assert "trust score" not in text

    def test_remediation_checklist_is_scenario_specific(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Schema Migration Disaster",
            episode_traces=self._make_traces(),
            final_health=0.55,
            mttr_steps=15,
        )
        assert len(report.remediation_checklist) >= 3
        # Should mention blue-green or migration for this scenario
        all_items = " ".join(report.remediation_checklist).lower()
        assert "migration" in all_items or "schema" in all_items

    def test_to_markdown_produces_valid_markdown(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Credential Compromise",
            episode_traces=self._make_traces(),
            final_health=0.7,
            mttr_steps=10,
        )
        md = report.to_markdown()
        assert md.startswith("# 🚨 Incident Postmortem:")
        assert "## 📋 Executive Summary" in md
        assert "## ✅ Remediation Checklist" in md

    def test_to_dict_all_keys_present(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Queue Cascade",
            episode_traces=self._make_traces(),
            final_health=0.8,
            mttr_steps=6,
        )
        d = report.to_dict()
        expected_keys = {
            "generated_at", "scenario_name", "root_cause", "severity",
            "mttr_steps", "final_health", "sla_met", "executive_summary",
            "engineering_summary", "customer_safe_summary", "timeline",
            "contributing_factors", "key_decisions", "debate_highlights",
            "remediation_checklist", "prevention_recommendations",
            "revenue_impact_usd", "users_protected", "slo_impact",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_sla_breach_reflected_in_report(self):
        comms = CommsAgent()
        report = comms.generate_postmortem(
            scenario_name="Cache Stampede",
            episode_traces=self._make_traces(),
            final_health=0.3,  # below 0.5 threshold
            mttr_steps=20,
        )
        assert report.sla_met is False
        assert "breach" in report.slo_impact.lower() or "exhausted" in report.slo_impact.lower()

    def test_root_cause_hypothesis_integrated(self):
        comms = CommsAgent()
        rca = {"scenario_name": "Cache Stampede", "confidence": 0.85}
        report = comms.generate_postmortem(
            scenario_name="Cache Stampede",
            episode_traces=self._make_traces(),
            final_health=0.7,
            mttr_steps=8,
            root_cause_hypothesis=rca,
        )
        assert "85%" in report.root_cause or "0.85" in report.root_cause
