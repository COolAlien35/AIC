import { Award } from 'lucide-react';
import { motion } from 'motion/react';
import type { DashboardStep } from '@/src/data/dashboardData';
import { metric } from '@/src/data/metrics';

export function ReliabilityLeaderboard({ step }: { step: DashboardStep | null }) {
  const leaders = [
    { service: 'Gateway', score: 100 - metric(step, 'network') / 5 },
    { service: 'Application', score: 100 - metric(step, 'latency') / 40 },
    { service: 'Database', score: 100 - metric(step, 'dbLatency') / 15 },
    { service: 'Queue', score: 100 - metric(step, 'queue') / 18 },
    { service: 'SLA', score: metric(step, 'sla') },
  ]
    .map((service) => ({ ...service, score: Math.max(5, Math.min(99, Math.round(service.score))) }))
    .sort((a, b) => b.score - a.score)
    .map((service, index) => ({
      ...service,
      uptime: `${Math.max(90, Math.min(99.99, (service.score / 100) * 100)).toFixed(2)}%`,
      rank: index === 0 ? 'Gold' : index === 1 ? 'Silver' : index === 2 ? 'Bronze' : service.score > 70 ? 'Stable' : 'Needs Attention',
    }));

  return (
    <div className="panel-geometric p-6 flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Award size={16} className="text-blue-500" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Service Reliability</h3>
        </div>
      </div>

      <div className="space-y-4">
        {leaders.map((leader, i) => (
          <div key={leader.service} className="flex flex-col gap-2 group">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-600 font-mono">0{i + 1}</span>
                <span className="text-sm font-semibold text-slate-200 group-hover:text-blue-400 transition-colors tracking-tight">{leader.service}</span>
              </div>
              <span className={`text-[9px] font-bold uppercase tracking-widest ${
                leader.rank === 'Gold' ? 'text-blue-400' :
                leader.rank === 'Silver' ? 'text-slate-400' :
                leader.rank === 'Bronze' ? 'text-amber-600' :
                leader.rank === 'Stable' ? 'text-emerald-500' : 'text-red-500'
              }`}>
                {leader.rank}
              </span>
            </div>
            <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${leader.score}%` }}
                className={`h-full rounded-full transition-all duration-1000 ${
                  leader.score > 90 ? 'bg-blue-600 shadow-[0_0_8px_rgba(37,99,235,0.4)]' :
                  leader.score > 80 ? 'bg-slate-500' : 'bg-red-600'
                }`}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
