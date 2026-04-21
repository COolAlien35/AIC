# Schema Migration Disaster

## Symptoms
- db_latency_ms explodes (>1000ms)
- replication_lag_ms increases rapidly (>500ms)
- conn_pool_pct saturates (>95%)
- db_latency_ms field may be renamed in telemetry (schema drift)
- Error rates increase due to query failures

## Root Cause
A botched database schema migration locks critical tables and introduces incompatible schema changes. Running queries fail against the new schema, replication breaks as replicas cannot apply the migration.

## Remediation Steps
1. **Immediate**: Pause the migration and release table locks
2. **Short-term**: Revert the schema change using the pre-migration backup
3. **Medium-term**: Throttle write throughput to allow replication to catch up
4. **Long-term**: Implement blue-green database deployments for zero-downtime migrations

## Rollback
- Execute the reverse migration script
- Restore from the pre-migration database snapshot if reverse migration fails

## Keywords
schema, migration, database, table lock, replication lag, db_latency, schema drift, migration disaster
