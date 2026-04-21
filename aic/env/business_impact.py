# aic/env/business_impact.py
"""
Business Impact Layer — maps system health metrics to business outcomes.

Revenue Loss Formula:
    L_rev = Σ(U_impacted × V_avg_transaction × E_rate)

Non-linear scaling:
    - error_rate < 5%:  U_impacted scales linearly (× 1.0)
    - 5% ≤ error_rate < 20%: U_impacted scales moderately (× 2.0)
    - error_rate ≥ 20%: U_impacted scales exponentially (user abandonment cascade)

Severity mapping: P1–P4 based on revenue_loss and compliance_risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Business constants
BASE_USERS: int = 50_000                # concurrent users at any time
V_AVG_TRANSACTION: float = 4.50         # average transaction value in USD
HEALTHY_SLA_COMPLIANCE: float = 99.9    # target SLA compliance percentage


@dataclass
class BusinessImpact:
    """Business impact metrics for a single step."""
    revenue_loss_per_minute: float
    users_impacted: int
    compliance_risk_score: float
    severity_level: str
    error_rate_pct: float
    sla_compliance_pct: float


def _compute_users_impacted(error_rate_pct: float) -> int:
    """
    Compute impacted users with non-linear scaling.

    - error_rate < 5%:  linear scaling
    - 5% ≤ error_rate < 20%: moderate scaling (×2)
    - error_rate ≥ 20%: exponential scaling (user abandonment cascade)
    """
    error_rate = max(0.0, error_rate_pct)

    if error_rate < 5.0:
        # Linear: mild impact
        fraction = (error_rate / 100.0) * 1.0
    elif error_rate < 20.0:
        # Moderate: users start noticing
        fraction = (error_rate / 100.0) * 2.0
    else:
        # Exponential: cascade of user abandonment
        # exp((error_rate - 20) / 10) grows rapidly above 20%
        multiplier = math.exp((error_rate - 20.0) / 10.0)
        fraction = (error_rate / 100.0) * multiplier

    # Cap at total user base
    impacted = int(min(BASE_USERS, BASE_USERS * fraction))
    return max(0, impacted)


def _compute_compliance_risk(sla_compliance_pct: float) -> float:
    """
    Map SLA compliance percentage to a risk score in [0.0, 1.0].

    99.9% → 0.0 (no risk)
    99.0% → ~0.1
    95.0% → ~0.5
    90.0% → ~0.7
    80.0% → ~0.9
    <70%  → 1.0
    """
    if sla_compliance_pct >= HEALTHY_SLA_COMPLIANCE:
        return 0.0

    # Normalize: how far below target (0% = at target, 100% = at 0%)
    deviation = (HEALTHY_SLA_COMPLIANCE - sla_compliance_pct) / HEALTHY_SLA_COMPLIANCE
    # Apply sigmoid-like curve for non-linear scaling
    risk = 1.0 - math.exp(-3.0 * deviation)
    return max(0.0, min(1.0, risk))


def _severity_from_impact(
    revenue_loss: float, compliance_risk: float
) -> str:
    """
    Classify incident severity based on revenue loss and compliance risk.

    P1: revenue_loss > $500/min OR compliance_risk > 0.8
    P2: revenue_loss > $100/min OR compliance_risk > 0.5
    P3: revenue_loss > $20/min
    P4: everything else
    """
    if revenue_loss > 500.0 or compliance_risk > 0.8:
        return "P1"
    elif revenue_loss > 100.0 or compliance_risk > 0.5:
        return "P2"
    elif revenue_loss > 20.0:
        return "P3"
    else:
        return "P4"


def compute_business_impact(
    metrics: dict[str, float],
    base_users: int = BASE_USERS,
    avg_transaction_value: float = V_AVG_TRANSACTION,
) -> BusinessImpact:
    """
    Compute full business impact from current system metrics.

    Args:
        metrics: Dict with at least 'error_rate_pct' and 'sla_compliance_pct'.
        base_users: Total concurrent user base.
        avg_transaction_value: Average transaction value in USD.

    Returns:
        BusinessImpact dataclass with all computed fields.
    """
    error_rate_pct = metrics.get("error_rate_pct", 0.0)
    sla_compliance_pct = metrics.get("sla_compliance_pct", HEALTHY_SLA_COMPLIANCE)

    # Compute impacted users (non-linear)
    users_impacted = _compute_users_impacted(error_rate_pct)

    # Revenue loss: L_rev = U_impacted × V_avg × E_rate
    e_rate = error_rate_pct / 100.0
    revenue_loss = users_impacted * avg_transaction_value * e_rate

    # Compliance risk
    compliance_risk = _compute_compliance_risk(sla_compliance_pct)

    # Severity classification
    severity = _severity_from_impact(revenue_loss, compliance_risk)

    return BusinessImpact(
        revenue_loss_per_minute=round(revenue_loss, 2),
        users_impacted=users_impacted,
        compliance_risk_score=round(compliance_risk, 4),
        severity_level=severity,
        error_rate_pct=error_rate_pct,
        sla_compliance_pct=sla_compliance_pct,
    )
