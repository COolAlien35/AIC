import { CheckCircle2, Clock, AlertTriangle, Zap, Play, RotateCcw } from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '@/src/lib/utils';

const steps = [
  { time: '08:15 AM', event: 'Alert Triggered - High CPU Usage', severity: 'Critical', icon: AlertTriangle, color: 'text-red-500' },
  { time: '08:21 AM', event: 'System Auto-Scaling Initiated', severity: 'Action', icon: Zap, color: 'text-blue-500' },
  { time: '08:30 AM', event: 'AI Anomaly Detection - Database', severity: 'Info', icon: Clock, color: 'text-slate-400' },
  { time: '08:45 AM', event: 'Manual Failover to Secondary Region', severity: 'Success', icon: CheckCircle2, color: 'text-emerald-500' },
];

export function IncidentTimeline() {
  return (
    <div className="panel-geometric p-6 flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Event Logging</h3>
        <RotateCcw size={14} className="text-slate-700 cursor-pointer hover:text-white" />
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
          <p className="text-xs font-bold text-white uppercase tracking-wider">6m</p>
        </div>
        <div className="py-3 px-1 text-center">
          <p className="text-[8px] font-bold uppercase text-slate-600 tracking-tighter mb-1">Mitigated</p>
          <p className="text-xs font-bold text-white uppercase tracking-wider">45m</p>
        </div>
        <div className="py-3 px-1 text-center">
          <p className="text-[8px] font-bold uppercase text-slate-600 tracking-tighter mb-1">Impact</p>
          <p className="text-xs font-bold text-red-500 uppercase tracking-widest">High</p>
        </div>
      </div>
    </div>
  );
}
