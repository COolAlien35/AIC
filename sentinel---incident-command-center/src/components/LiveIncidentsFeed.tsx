import { AlertCircle, AlertTriangle, Info, MoreHorizontal } from 'lucide-react';
import { cn } from '@/src/lib/utils';

const incidents = [
  {
    id: 1,
    type: 'critical',
    title: 'CRITICAL: Payment Gateway Timeout',
    meta: 'Severity 1 - 2m ago',
    icon: AlertCircle,
    color: 'text-red-500',
    bg: 'bg-red-500/10'
  },
  {
    id: 2,
    type: 'warning',
    title: 'WARNING: Database Latency',
    meta: 'Severity 2 - 15m ago',
    icon: AlertTriangle,
    color: 'text-amber-500',
    bg: 'bg-amber-500/10'
  },
  {
    id: 3,
    type: 'info',
    title: 'INFO: New AI Policy Applied',
    meta: 'Severity 4 - 45m ago',
    icon: Info,
    color: 'text-blue-500',
    bg: 'bg-blue-500/10'
  }
];

export function LiveIncidentsFeed() {
  return (
    <div className="panel-geometric overflow-hidden flex flex-col h-full">
      <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-900/20">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Live Infrastructure Feed</h3>
        <button className="text-blue-500 text-[10px] font-bold uppercase tracking-widest hover:underline">View All Systems</button>
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
                    <incident.icon size={14} className={cn("shrink-0", incident.color)} />
                    <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">{incident.title}</span>
                  </div>
                </td>
                <td className="px-6 py-3 text-xs text-slate-600 font-mono italic">
                  {incident.meta.split(' - ')[1]}
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
