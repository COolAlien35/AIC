# Canary Failure

## Symptoms
- error_rate_pct spikes for a subset of users (2-5% increase)
- p95_latency_ms increases significantly (>500ms above baseline)
- throughput_rps drops moderately
- Error rate telemetry may go dark (NaN) during observation window

## Root Cause
A canary deployment with a critical bug receives a portion of production traffic. The faulty code path causes increased errors and latency for affected users while healthy instances remain normal.

## Remediation Steps
1. **Immediate**: Roll back the canary deployment to the previous stable version
2. **Short-term**: Shift 100% of traffic to the stable deployment
3. **Medium-term**: Analyze canary error logs to identify the specific bug
4. **Long-term**: Strengthen canary analysis with automated rollback triggers

## Rollback
- If rollback fails, disable the canary service entirely
- Redirect all traffic through the stable deployment path

## Keywords
canary, deployment, rollback, error rate, latency spike, canary failure, deployment bug
