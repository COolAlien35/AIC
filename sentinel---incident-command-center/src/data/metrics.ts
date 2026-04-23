import type { DashboardStep } from './dashboardData';

const METRIC_ALIASES: Record<string, string[]> = {
  cpu: ['cpu_pct', 'cpu_load'],
  latency: ['p95_latency_ms', 'app_latency_ms'],
  dbLatency: ['db_latency_ms'],
  network: ['net_io_mbps', 'net_io'],
  errors: ['error_rate_pct', 'error_rate'],
  sla: ['sla_compliance_pct', 'sla_compliance'],
  queue: ['queue_depth'],
  throughput: ['throughput_rps'],
  memory: ['mem_pct'],
};

export function metric(step: DashboardStep | null, metricName: keyof typeof METRIC_ALIASES, fallback = 0): number {
  if (!step) return fallback;
  const keys = METRIC_ALIASES[metricName];
  for (const key of keys) {
    const value = step.metrics?.[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return fallback;
}

export function metricSeries(trajectory: DashboardStep[], metricName: keyof typeof METRIC_ALIASES) {
  return trajectory.map((step) => ({
    step: step.step,
    value: metric(step, metricName),
  }));
}

export function formatPct(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`;
}
