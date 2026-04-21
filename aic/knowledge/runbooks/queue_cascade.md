# Queue Cascade

## Symptoms
- queue_depth explodes (>500)
- p95_latency_ms increases significantly
- error_rate_pct rises as consumers fail to keep up
- cpu_pct increases from consumer processing overhead
- queue_depth telemetry may show unit shift (reports in thousands instead of units)

## Root Cause
Message queue consumer group rebalance storm. A consumer crash triggers rebalancing, which causes other consumers to pause, leading to a cascade of message backlog. The queue depth spirals as producers continue sending while consumers are stuck rebalancing.

## Remediation Steps
1. **Immediate**: Scale up consumer instances to drain the backlog
2. **Short-term**: Increase consumer group session timeout to reduce rebalance frequency
3. **Medium-term**: Enable dead-letter queue for poison messages
4. **Long-term**: Implement consumer lag monitoring with auto-scaling triggers

## Rollback
- Scale down consumer instances once the backlog is drained
- Restore original session timeout values
- Reprocess dead-letter queue messages after root cause is fixed

## Keywords
queue, cascade, consumer, rebalance, message backlog, queue depth, dead letter, consumer group
