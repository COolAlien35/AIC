# Cache Stampede

## Symptoms
- db_latency_ms spikes above 500ms
- queue_depth increases rapidly (>200)
- conn_pool_pct saturates above 90%
- error_rate_pct increases moderately
- cache hit ratio drops to near zero

## Root Cause
All cache keys expire simultaneously (TTL alignment), causing a "thundering herd" of requests that bypass the cache layer and hit the database directly. The DB connection pool saturates under the sudden load spike.

## Remediation Steps
1. **Immediate**: Increase DB connection pool limit to absorb the surge
2. **Short-term**: Enable request coalescing to deduplicate identical cache-miss queries
3. **Medium-term**: Implement staggered TTL expiration with jitter (TTL ± 10%)
4. **Long-term**: Deploy a cache warming service that pre-populates keys before expiration

## Rollback
- Revert connection pool changes if memory pressure increases
- Disable request coalescing if it causes response delays
- Restore original TTL values

## Keywords
cache, stampede, thundering herd, TTL, cache miss, connection pool, db_latency, queue_depth
