# aic/agents/knowledge_agent.py
"""
Knowledge Agent — keyword-based RAG over incident runbooks.

Retrieves the most relevant runbook for the current incident based on
metric signatures and keyword matching. Returns KnowledgeEvidence or
None if confidence is too low (prevents hallucinated matches).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KnowledgeEvidence:
    """Evidence retrieved from a runbook."""
    related_incident_id: str
    incident_name: str
    suggested_remediation: str
    confidence_score: float
    source_file: str
    keywords_matched: list[str]


# Default runbook directory
_DEFAULT_RUNBOOK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "knowledge", "runbooks"
)

# Metric-to-keyword mapping for automatic keyword extraction from observations
_METRIC_KEYWORD_MAP: dict[str, list[str]] = {
    "db_latency_ms": ["db_latency", "database", "latency"],
    "conn_pool_pct": ["connection pool", "conn_pool"],
    "replication_lag_ms": ["replication", "lag", "replica"],
    "queue_depth": ["queue", "consumer", "message backlog"],
    "error_rate_pct": ["error rate", "error"],
    "p95_latency_ms": ["latency", "p95"],
    "packet_loss_pct": ["packet loss", "network"],
    "dns_latency_ms": ["DNS", "dns"],
    "net_io_mbps": ["network", "net_io"],
    "auth_failure_rate": ["auth failure", "credential", "authentication"],
    "suspicious_token_count": ["token", "suspicious", "credential"],
    "compromised_ip_count": ["compromised", "IP", "security"],
    "cpu_pct": ["cpu", "compute"],
    "mem_pct": ["memory", "mem"],
    "throughput_rps": ["throughput"],
}


class RunbookEntry:
    """Parsed runbook with searchable content."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.incident_id = self.filename.replace(".md", "")
        self.content = ""
        self.title = ""
        self.keywords: list[str] = []
        self.remediation: str = ""
        self._parse()

    def _parse(self) -> None:
        """Parse markdown runbook for title, keywords, and remediation."""
        with open(self.filepath, "r") as f:
            self.content = f.read().lower()

        # Extract title (first H1)
        lines = self.content.strip().split("\n")
        for line in lines:
            if line.startswith("# "):
                self.title = line[2:].strip()
                break

        # Extract keywords section
        kw_match = re.search(
            r"##\s*keywords\s*\n(.*?)(?:\n##|\Z)",
            self.content,
            re.DOTALL,
        )
        if kw_match:
            kw_text = kw_match.group(1).strip()
            self.keywords = [
                k.strip() for k in kw_text.split(",") if k.strip()
            ]

        # Extract remediation section
        rem_match = re.search(
            r"##\s*remediation\s*steps?\s*\n(.*?)(?:\n##|\Z)",
            self.content,
            re.DOTALL,
        )
        if rem_match:
            self.remediation = rem_match.group(1).strip()
        else:
            self.remediation = "See runbook for detailed remediation steps."


class KnowledgeAgent:
    """
    Retrieves relevant incident runbooks based on metric observations.

    Uses keyword matching against runbook content and keyword sections.
    Returns None if confidence is below threshold (prevents hallucination).
    """

    MIN_CONFIDENCE: float = 0.3

    def __init__(self, runbook_dir: Optional[str] = None):
        self._runbook_dir = runbook_dir or _DEFAULT_RUNBOOK_DIR
        self._runbooks: list[RunbookEntry] = []
        self._load_runbooks()

    def _load_runbooks(self) -> None:
        """Load all markdown runbooks from the runbook directory."""
        runbook_path = Path(self._runbook_dir)
        if not runbook_path.exists():
            return
        for md_file in sorted(runbook_path.glob("*.md")):
            self._runbooks.append(RunbookEntry(str(md_file)))

    def retrieve(
        self,
        metrics: dict[str, float],
        hypothesis: Optional[str] = None,
    ) -> Optional[KnowledgeEvidence]:
        """
        Retrieve the most relevant runbook for the current situation.

        Args:
            metrics: Current metric values (used to extract search keywords).
            hypothesis: Optional root cause hypothesis name to boost matching.

        Returns:
            KnowledgeEvidence if confidence >= MIN_CONFIDENCE, else None.
        """
        if not self._runbooks:
            return None

        # Extract search keywords from elevated metrics
        search_keywords = self._extract_keywords(metrics)

        # Add hypothesis keywords if provided
        if hypothesis:
            hyp_words = hypothesis.lower().replace("_", " ").split()
            search_keywords.extend(hyp_words)

        # Score each runbook
        best_score = 0.0
        best_runbook: Optional[RunbookEntry] = None
        best_matched: list[str] = []

        for runbook in self._runbooks:
            score, matched = self._score_runbook(runbook, search_keywords)
            if score > best_score:
                best_score = score
                best_runbook = runbook
                best_matched = matched

        # Normalize score to [0, 1] confidence
        max_possible = max(len(search_keywords), 1)
        confidence = min(1.0, best_score / max_possible)

        if confidence < self.MIN_CONFIDENCE or best_runbook is None:
            return None

        return KnowledgeEvidence(
            related_incident_id=best_runbook.incident_id,
            incident_name=best_runbook.title,
            suggested_remediation=best_runbook.remediation[:500],
            confidence_score=round(confidence, 3),
            source_file=best_runbook.filename,
            keywords_matched=best_matched,
        )

    def _extract_keywords(self, metrics: dict[str, float]) -> list[str]:
        """Extract search keywords from metrics that are significantly elevated."""
        keywords: list[str] = []
        from aic.utils.constants import METRIC_TARGETS

        for metric_name, value in metrics.items():
            target = METRIC_TARGETS.get(metric_name, None)
            if target is None:
                continue
            # Check if metric is significantly off target
            if target == 0.0:
                if value > 1.0:
                    keywords.extend(_METRIC_KEYWORD_MAP.get(metric_name, []))
            else:
                pct_off = abs(value - target) / target
                if pct_off > 0.3:  # >30% from target
                    keywords.extend(_METRIC_KEYWORD_MAP.get(metric_name, []))

        return keywords

    def _score_runbook(
        self, runbook: RunbookEntry, search_keywords: list[str]
    ) -> tuple[float, list[str]]:
        """Score a runbook against search keywords. Returns (score, matched_keywords)."""
        matched: list[str] = []
        score = 0.0

        for kw in search_keywords:
            kw_lower = kw.lower()
            # Check keyword section (higher weight)
            if any(kw_lower in rk for rk in runbook.keywords):
                score += 2.0
                matched.append(kw)
            # Check full content (lower weight)
            elif kw_lower in runbook.content:
                score += 1.0
                matched.append(kw)

        return score, matched

    def get_runbook_count(self) -> int:
        """Return number of loaded runbooks."""
        return len(self._runbooks)

    def list_runbooks(self) -> list[str]:
        """Return list of loaded runbook IDs."""
        return [r.incident_id for r in self._runbooks]
