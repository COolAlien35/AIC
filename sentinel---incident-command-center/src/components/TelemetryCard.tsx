import { 
  AreaChart,
  Area,
  ResponsiveContainer,
  Tooltip
} from 'recharts';
import { Activity } from 'lucide-react';
import type { DashboardStep } from '@/src/data/dashboardData';
import { metric } from '@/src/data/metrics';

interface TelemetryCardProps {
  step: DashboardStep | null;
  trajectory: DashboardStep[];
  stepIndex: number;
}

export function TelemetryCard({ step, trajectory, stepIndex }: TelemetryCardProps) {
  const metrics = step?.metrics ?? {};
  const cpuLoad = metric(step, 'cpu');
  const appLatency = metric(step, 'latency');
  const errorRate = metric(step, 'errors');
  const history = trajectory.slice(0, stepIndex + 1).map((item) => ({
    step: item.step,
    cpu: metric(item, 'cpu'),
    latency: metric(item, 'latency'),
    errors: metric(item, 'errors'),
  }));
  const health = Number(step?.health ?? 0);

  return (
    <div className="panel-geometric p-6 flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Live Telemetry</h3>
        <Activity size={16} className="text-slate-500" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="card-geometric p-3">
          <p className="text-[10px] uppercase tracking-widest text-slate-500">CPU</p>
          <p className="text-lg font-semibold text-white mt-1">{cpuLoad.toFixed(1)}%</p>
        </div>
        <div className="card-geometric p-3">
          <p className="text-[10px] uppercase tracking-widest text-slate-500">P95 Latency</p>
          <p className="text-lg font-semibold text-white mt-1">{appLatency.toFixed(0)} ms</p>
        </div>
        <div className="card-geometric p-3">
          <p className="text-[10px] uppercase tracking-widest text-slate-500">Error Rate</p>
          <p className="text-lg font-semibold text-white mt-1">{errorRate.toFixed(2)}%</p>
        </div>
        <div className="card-geometric p-3">
          <p className="text-[10px] uppercase tracking-widest text-slate-500">Health</p>
          <p className="text-lg font-semibold text-white mt-1">{(health * 100).toFixed(1)}%</p>
        </div>
      </div>

      <div className="pt-2 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={history}>
            <defs>
              <linearGradient id="telemetry-cpu" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
            <Area type="monotone" dataKey="cpu" stroke="#3b82f6" fill="url(#telemetry-cpu)" strokeWidth={2} />
            <Area type="monotone" dataKey="latency" stroke="#10b981" fill="none" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
