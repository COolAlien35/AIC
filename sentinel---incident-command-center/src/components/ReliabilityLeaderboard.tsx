import { Award } from 'lucide-react';
import { motion } from 'motion/react';

const leaders = [
  { service: 'Payments Service', uptime: '99.99%', rank: 'Gold', score: 98 },
  { service: 'User Authentication', uptime: '99.95%', rank: 'Silver', score: 95 },
  { service: 'Inventory', uptime: '99.95%', rank: 'Bronze', score: 92 },
  { service: 'Search', uptime: '99.92%', rank: 'Stable', score: 88 },
  { service: 'Order Processing', uptime: '99.85%', rank: 'Needs Attention', score: 72 },
];

export function ReliabilityLeaderboard() {
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
