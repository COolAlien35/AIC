# Regional Outage

## Symptoms
- All metrics degrade simultaneously across multiple services
- net_io_mbps shows anomalous readings or goes dark (NaN)
- packet_loss_pct spikes significantly
- regional_latency_ms increases above 200ms
- Error rates increase across all services uniformly

## Root Cause
An entire availability zone or region experiences infrastructure failure. This could be due to power outage, network partition, or cloud provider incident affecting the region.

## Remediation Steps
1. **Immediate**: Trigger DNS failover to route traffic to healthy regions
2. **Short-term**: Reroute traffic away from the affected availability zone
3. **Medium-term**: Enable cross-region request hedging for critical paths
4. **Long-term**: Implement active-active multi-region deployment

## Rollback
- Restore original DNS routing once the region recovers
- Re-enable the affected AZ gradually with traffic ramping

## Keywords
regional, outage, availability zone, AZ, DNS failover, packet loss, network, multi-region, failover
