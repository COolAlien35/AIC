# Credential Compromise

## Symptoms
- error_rate_pct spikes sharply (>5% increase)
- auth_failure_rate increases dramatically
- suspicious_token_count rises
- throughput_rps drops as WAF throttles traffic
- sla_compliance_pct monitoring may go dark (NaN)

## Root Cause
Leaked API credentials trigger a brute-force attack or unauthorized access. Authentication failures spike from invalid credential attempts, and the Web Application Firewall (WAF) begins throttling traffic to protect the system.

## Remediation Steps
1. **Immediate**: Revoke all compromised API tokens and force re-authentication
2. **Short-term**: Block known compromised IP addresses at the WAF
3. **Medium-term**: Rotate all service-to-service credentials
4. **Long-term**: Implement credential scanning in CI/CD and short-lived token rotation

## Rollback
- Re-issue tokens for verified legitimate users
- Unblock IPs if false positives are detected
- Restore WAF rules to normal operating mode

## Keywords
credential, compromise, security, auth failure, token, API key, brute force, WAF, compromised IP
