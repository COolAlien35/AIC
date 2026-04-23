import { AlertCircle, AlertTriangle, Info, MoreHorizontal } from 'lucide-react';
import { cn } from '@/src/lib/utils';
import type { DashboardStep, Severity } from '@/src/data/dashboardData';
import { metricToSeverity } from '@/src/data/dashboardData';
import { metric } from '@/src/data/metrics';

function severityIcon(type: Severity) {
  if (type === 'critical') return AlertCircle;
  if (type === 'warning') return AlertTriangle;
  return Info;
}

export function LiveIncidentsFeed({ step }: { step: DashboardStep | null }) {
  const cpu = metric(step, 'cpu');
  const latency = metric(step, 'latency');
  const sla = metric(step, 'sla');
  const errors = metric(step, 'errors');
  const incidents = [
    {
      id: 1,
      type: metricToSeverity(1 - Number(step?.health ?? 1), 0.4),
      title: `Service Health ${((step?.health ?? 0) * 100).toFixed(1)}%`,
      meta: `Step ${step?.step ?? 0}`,
    },
    {
      id: 2,
      type: metricToSeverity(latency, 600),
      title: `P95 Latency ${latency.toFixed(0)} ms`,
      meta: `CPU ${cpu.toFixed(1)}%`,
    },
    {
      id: 3,
      type: metricToSeverity(100 - sla, 15),
      title: `SLA Compliance ${sla.toFixed(1)}%`,
      meta: `Error Rate ${errors.toFixed(2)}%`,
    },
  ];

  return (
    <div className="panel-geometric overflow-hidden flex flex-col h-full">
      <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-900/20">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Live Infrastructure Feed</h3>
        <span className="text-[10px] uppercase tracking-widest text-slate-500">Auto-generated from current step</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-left">
          <thead className="bg-slate-900/40 divide-y divide-slate-800">
            <tr className="text-[10px] text-slate-600 uppercase tracking-widest">
              <th className="px-6 py-3 font-bold">Severity</th>
              <th className="px-6 py-3 font-bold">Incident</th>
              <th className="px-6 py-3 font-bold">Duration</th>
              <th className="px-6 py-3 font-bold text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {incidents.map((incident) => (
              <tr key={incident.id} className="hover:bg-slate-800/20 transition-colors group cursor-pointer">
                <td className="px-6 py-3">
                  <span className={cn(
                    "px-2 py-0.5 rounded-full text-[9px] font-bold uppercase border",
                    incident.type === 'critical' ? 'bg-red-500/10 text-red-500 border-red-500/20' :
                    incident.type === 'warning' ? 'bg-amber-500/10 text-amber-500 border-amber-500/20' :
                    'bg-blue-500/10 text-blue-500 border-blue-500/20'
                  )}>
                    {incident.type}
                  </span>
                </td>
                <td className="px-6 py-3">
                  <div className="flex items-center gap-3">
                    {(() => {
                      const Icon = severityIcon(incident.type);
                      const color = incident.type === 'critical'
                        ? 'text-red-500'
                        : incident.type === 'warning'
                          ? 'text-amber-500'
                          : 'text-blue-500';
                      return <Icon size={14} className={cn('shrink-0', color)} />;
                    })()}
                    <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">{incident.title}</span>
                  </div>
                </td>
                <td className="px-6 py-3 text-xs text-slate-600 font-mono italic">
                  {incident.meta}
                </td>
                <td className="px-6 py-3 text-right">
                   <MoreHorizontal size={14} className="ml-auto text-slate-700" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
