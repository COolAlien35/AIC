---
name: aic-auditor
description: Specialized logic for detecting schema drift and service deadlocks in AIC.
---
# AIC Auditor Instructions
1. **Schema Check:** On every step, compare current keys in `raw_response` against METRIC_TARGETS.
2. **Drift Detection:** If a key is missing or renamed (e.g., 'p95_latency'), set `schema_drift_detected: true` in the Explanation Trace.
3. **Deadlock Watch:** Check if an agent has been 'waiting' for a service lock for 2+ steps. If so, trigger the DEADLOCK_PENALTY logic.
4. **Causal Logic:** Remember the DB->App coupling (Alpha=0.4, Lag=2). Use this to score the 'predicted_2step_impact'.