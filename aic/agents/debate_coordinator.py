# aic/agents/debate_coordinator.py
"""
Debate Coordinator — structured multi-agent debate before orchestrator selection.

Round structure (one round per orchestrator step):
  1. Top-2 candidate recommendations are identified by confidence.
  2. Each specialist agent gets to *criticise* the competing top recommendation.
  3. SecurityAgent gets a special veto if an action increases attack surface.
  4. DBAgent critiques network/infra actions that ignore DB coupling.
  5. The orchestrator receives the full debate transcript alongside recommendations.

All criticism is rule-based (no LLM required) — deterministic and fast.
The debate mutates SubAgentRecommendation in place by appending to
`criticisms`, `supports`, and `rebuttals` fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aic.schemas.traces import SubAgentRecommendation
from aic.utils.constants import (
    AGENT_DB, AGENT_INFRA, AGENT_APP, AGENT_ADV,
    AGENT_NET, AGENT_SEC, METRIC_TARGETS,
)


# ── Thresholds that trigger criticisms ────────────────────────────────────
_DB_COUPLING_THRESHOLD = 0.5          # conn_pool above 50% above target triggers DB critique
_SEC_RISK_THRESHOLD = 0.45            # risk_score above this → security flag
_BLAST_HIGH_THRESHOLD = "high"        # blast radius flag
_INFRA_CPU_CRITICAL = 85.0            # cpu above this → infra critique of non-infra actions


@dataclass
class DebateRound:
    """A single round of structured agent debate."""
    step: int
    criticisms: list[dict]       # {critic, target_agent, text}
    supports: list[dict]         # {supporter, target_agent, text}
    security_vetoes: list[dict]  # {reason, action_blocked}
    consensus_agent: Optional[str]  # agent the debate pointed toward (or None)
    debate_changed_selection: bool  # True if debate overrode the pre-debate top pick


class DebateCoordinator:
    """
    Runs one structured debate round and annotates recommendations
    with criticisms, supports, and rebuttals.

    All logic is deterministic — no LLM calls.
    """

    def __init__(self) -> None:
        self._rounds: list[DebateRound] = []

    def reset(self) -> None:
        self._rounds.clear()

    def run_debate(
        self,
        recommendations: list[SubAgentRecommendation],
        current_metrics: dict[str, float],
        root_cause_hypothesis: Optional[dict] = None,
    ) -> tuple[list[SubAgentRecommendation], DebateRound]:
        """
        Conduct one debate round.

        Args:
            recommendations: Current step's recommendations (modified in place).
            current_metrics: Live metric snapshot.
            root_cause_hypothesis: Optional Bayesian root cause dict.

        Returns:
            (annotated_recommendations, DebateRound)
        """
        step = len(self._rounds)
        criticisms: list[dict] = []
        supports: list[dict] = []
        security_vetoes: list[dict] = []

        # Exclude adversarial agent from debate participation
        valid = [r for r in recommendations if r.agent_name != AGENT_ADV]
        if not valid:
            round_ = DebateRound(
                step=step, criticisms=[], supports=[],
                security_vetoes=[], consensus_agent=None,
                debate_changed_selection=False,
            )
            self._rounds.append(round_)
            return recommendations, round_

        # Sort by confidence to get top-2
        sorted_recs = sorted(valid, key=lambda r: r.confidence, reverse=True)
        top1 = sorted_recs[0] if len(sorted_recs) > 0 else None
        top2 = sorted_recs[1] if len(sorted_recs) > 1 else None

        pre_debate_winner = top1.agent_name if top1 else None

        # ── DB Agent critiques non-DB actions when DB coupling is high ────
        if top1 and top1.agent_name not in (AGENT_DB,):
            conn_pool = current_metrics.get("conn_pool_pct", 0)
            db_lat = current_metrics.get("db_latency_ms", 0)
            target_conn = METRIC_TARGETS.get("conn_pool_pct", 60.0)
            target_db = METRIC_TARGETS.get("db_latency_ms", 50.0)

            conn_severity = (conn_pool - target_conn) / max(target_conn, 1)
            db_severity = (db_lat - target_db) / max(target_db, 1)

            if conn_severity > _DB_COUPLING_THRESHOLD or db_severity > 1.5:
                text = (
                    f"DBAgent challenges {top1.agent_name}: "
                    f"conn_pool={conn_pool:.0f}% (target={target_conn:.0f}%), "
                    f"db_latency={db_lat:.0f}ms (target={target_db:.0f}ms). "
                    f"DB coupling is critical — acting on {top1.target_metrics} "
                    f"without addressing the DB bottleneck may fail."
                )
                criticisms.append({
                    "critic": AGENT_DB,
                    "target_agent": top1.agent_name,
                    "text": text,
                    "severity": "high" if db_severity > 3.0 else "medium",
                })
                # Annotate the recommendation
                top1.criticisms.append(text)

                # DB Agent supports top2 if it targets DB metrics
                if top2 and any("db" in m or "conn" in m or "repl" in m for m in top2.target_metrics):
                    sup_text = f"DBAgent supports {top2.agent_name} — targets DB-layer metrics directly."
                    supports.append({"supporter": AGENT_DB, "target_agent": top2.agent_name, "text": sup_text})
                    top2.supports.append(sup_text)

        # ── Security Agent vetoes actions that increase attack surface ────
        sec_agent_rec = next((r for r in valid if r.agent_name == AGENT_SEC), None)
        if top1 and top1.risk_score > _SEC_RISK_THRESHOLD:
            # Check if this is a security-sensitive scenario
            is_credential_scenario = (
                root_cause_hypothesis and
                "credential" in root_cause_hypothesis.get("scenario_name", "").lower()
            )
            error_rate = current_metrics.get("error_rate_pct", 0)

            if is_credential_scenario or error_rate > 15.0:
                veto_text = (
                    f"SecurityAgent flags {top1.agent_name}: "
                    f"risk_score={top1.risk_score:.2f} (threshold={_SEC_RISK_THRESHOLD}). "
                    f"Under active credential compromise, high-risk actions increase blast radius. "
                    f"Recommend: quarantine > restore > verify."
                )
                security_vetoes.append({
                    "reason": veto_text,
                    "action_blocked": top1.action[:100],
                    "risk_score": top1.risk_score,
                })
                top1.criticisms.append(veto_text)
                criticisms.append({
                    "critic": AGENT_SEC,
                    "target_agent": top1.agent_name,
                    "text": veto_text,
                    "severity": "critical",
                })

        # ── InfraAgent critiques when CPU is critical but action ignores it ─
        if top1 and top1.agent_name not in (AGENT_INFRA,):
            cpu = current_metrics.get("cpu_pct", 0)
            if cpu > _INFRA_CPU_CRITICAL:
                text = (
                    f"InfraAgent warns {top1.agent_name}: "
                    f"cpu_pct={cpu:.0f}% (critical). "
                    f"Without CPU relief, action outcome will degrade rapidly. "
                    f"Suggest pairing with pod scaling."
                )
                criticisms.append({
                    "critic": AGENT_INFRA,
                    "target_agent": top1.agent_name,
                    "text": text,
                    "severity": "medium",
                })
                top1.criticisms.append(text)

        # ── High blast radius without rollback — cross-agent challenge ────
        if top1 and top1.blast_radius == _BLAST_HIGH_THRESHOLD and not top1.rollback_plan.strip():
            challenger = next(
                (r.agent_name for r in valid if r.agent_name != top1.agent_name),
                None,
            )
            if challenger:
                text = (
                    f"{challenger} challenges {top1.agent_name}: "
                    f"blast_radius='high' with no rollback plan. "
                    f"This action is irreversible — request rollback documentation."
                )
                criticisms.append({
                    "critic": challenger,
                    "target_agent": top1.agent_name,
                    "text": text,
                    "severity": "high",
                })
                top1.criticisms.append(text)

        # ── Generate rebuttals (top1 defends itself) ──────────────────────
        if top1 and top1.criticisms:
            rebuttal = (
                f"Rebuttal from {top1.agent_name}: Despite concerns, "
                f"confidence={top1.confidence:.2f} is highest among validated candidates. "
                f"Predicted improvement on {top1.target_metrics}. "
                f"Risk={top1.risk_score:.2f} is within acceptable parameters."
            )
            top1.rebuttals.append(rebuttal)

        # ── Determine if debate changed the selection ──────────────────────
        # If top1 accumulated critical security vetoes, elevate top2
        debate_changed = False
        final_winner = pre_debate_winner

        if security_vetoes and top2 and top1:
            # Security override: swap top1 and top2
            idx1 = recommendations.index(top1)
            idx2 = recommendations.index(top2)
            recommendations[idx1], recommendations[idx2] = recommendations[idx2], recommendations[idx1]
            debate_changed = True
            final_winner = top2.agent_name

        # ── Determine consensus ───────────────────────────────────────────
        consensus_agent = None
        if len(supports) > 0:
            support_counts: dict[str, int] = {}
            for s in supports:
                support_counts[s["target_agent"]] = support_counts.get(s["target_agent"], 0) + 1
            consensus_agent = max(support_counts, key=support_counts.get)

        round_ = DebateRound(
            step=step,
            criticisms=criticisms,
            supports=supports,
            security_vetoes=security_vetoes,
            consensus_agent=consensus_agent,
            debate_changed_selection=debate_changed,
        )
        self._rounds.append(round_)
        return recommendations, round_

    def get_rounds(self) -> list[DebateRound]:
        return list(self._rounds)

    def get_last_round(self) -> Optional[DebateRound]:
        return self._rounds[-1] if self._rounds else None

    def summarise(self) -> dict:
        """Aggregate debate stats across the episode."""
        return {
            "total_rounds": len(self._rounds),
            "total_criticisms": sum(len(r.criticisms) for r in self._rounds),
            "total_security_vetoes": sum(len(r.security_vetoes) for r in self._rounds),
            "debate_changed_selection": sum(1 for r in self._rounds if r.debate_changed_selection),
        }
