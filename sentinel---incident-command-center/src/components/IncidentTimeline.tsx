import { CheckCircle2, AlertTriangle, Zap } from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '@/src/lib/utils';
import type { DashboardEpisode } from '@/src/data/dashboardData';

export function IncidentTimeline({ episode, step }: { episode: DashboardEpisode | null; step: number }) {
  const trace = episode?.trajectory.slice(0, step + 1).slice(-8) ?? [];
  const steps = trace.map((item) => {
    const isRecovered = item.health > 0.5;
    const isCritical = item.health < 0.35;
    return {
      time: `Step ${item.step}`,
      event: item.trace?.action_taken ?? 'No action recorded',
      severity: isRecovered ? 'Success' : isCritical ? 'Critical' : 'Action',
      icon: isRecovered ? CheckCircle2 : isCritical ? AlertTriangle : Zap,
      color: isRecovered ? 'text-emerald-500' : isCritical ? 'text-red-500' : 'text-blue-500',
    };
  });

  return (
    <div className="panel-geometric p-6 flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Event Logging</h3>
        <span className="text-[10px] uppercase tracking-widest text-slate-600">Last 8 actions</span>
      </div>

      <div className="relative pl-4 space-y-6 flex-1 before:absolute before:left-[21px] before:top-2 before:bottom-2 before:w-px before:bg-slate-800">
        {steps.map((step, i) => (
          <motion.div 
            initial={{ x: -10, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: i * 0.1 }}
            key={i} 
            className="relative flex gap-6"
          >
            <div className={`w-3.5 h-3.5 rounded bg-[#0A0A0B] border border-slate-700 z-10 mt-1.5 flex items-center justify-center shrink-0`}>
              <div className={`w-1.5 h-1.5 rounded-sm ${step.color.replace('text', 'bg')} shadow-[0_0_8px_currentColor]`} />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-mono font-bold text-slate-500 tracking-tighter">{step.time}</span>
              </div>
              <p className="text-xs font-semibold text-slate-200 leading-tight tracking-tight">{step.event}</p>
              <p className={cn("text-[9px] font-bold uppercase tracking-widest mt-1", step.color)}>{step.severity}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-3 divide-x divide-slate-800 bg-slate-900/40 rounded-xl border border-slate-800 overflow-hidden">
        <div className="py-3 px-1 text-center">
          <p className="text-[8px] font-bold uppercase text-slate-600 tracking-tighter mb-1">Detector</p>
          <p className="text-xs font-bold text-white uppercase tracking-wider">{Math.max(1, Math.floor((episode?.mttr ?? 1) / 2))}m</p>
        </div>
        <div className="py-3 px-1 text-center">
          <p className="text-[8px] font-bold uppercase text-slate-600 tracking-tighter mb-1">Mitigated</p>
          <p className="text-xs font-bold text-white uppercase tracking-wider">{episode?.mttr ?? 0}m</p>
        </div>
        <div className="py-3 px-1 text-center">
          <p className="text-[8px] font-bold uppercase text-slate-600 tracking-tighter mb-1">Impact</p>
          <p className={cn(
            'text-xs font-bold uppercase tracking-widest',
            (episode?.final_health ?? 1) > 0.5 ? 'text-emerald-500' : 'text-red-500'
          )}>
            {(episode?.final_health ?? 1) > 0.5 ? 'Contained' : 'High'}
          </p>
        </div>
      </div>
    </div>
  );
}
