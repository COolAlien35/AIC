import { motion } from 'motion/react';
import { ShieldCheck } from 'lucide-react';
import type { DashboardStep } from '@/src/data/dashboardData';

export function TrustScoreCard({ step }: { step: DashboardStep | null }) {
  const trustScores = Object.values(step?.trust_scores ?? {});
  const avgTrust = trustScores.length
    ? trustScores.reduce((a, b) => a + Number(b), 0) / trustScores.length
    : 0.72;
  const score = Math.round(avgTrust * 100);
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="panel-geometric p-6 flex flex-col justify-between relative overflow-hidden h-full">
      <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">AI Trust Score</h3>
      <div className="absolute top-6 right-6 text-slate-500"><ShieldCheck size={16} /></div>

      <div className="relative flex items-center justify-center my-8">
        <svg className="w-36 h-36 transform -rotate-90">
          <circle
            cx="72"
            cy="72"
            r={radius}
            className="stroke-slate-800 fill-none"
            strokeWidth="8"
          />
          <motion.circle
            cx="72"
            cy="72"
            r={radius}
            className="stroke-blue-600 fill-none"
            strokeWidth="8"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span 
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-3xl font-bold text-white"
          >
            {score}
            <span className="text-sm text-slate-500 font-medium">/100</span>
          </motion.span>
        </div>
      </div>

      <div className="text-center space-y-1">
        <div className="flex items-center gap-2 justify-center">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <p className="text-[11px] font-bold uppercase tracking-tighter text-emerald-400">
            {score >= 70 ? 'High Confidence' : score >= 45 ? 'Moderate Confidence' : 'Low Confidence'}
          </p>
        </div>
        <p className="text-[10px] text-slate-500">Automated remediation suggested</p>
      </div>

      <div className="mt-6 space-y-2">
        {Object.entries(step?.trust_scores ?? {}).map(([agent, value]) => (
          <div key={agent}>
            <div className="flex items-center justify-between text-[10px] uppercase tracking-widest text-slate-500 mb-1">
              <span>{agent.replace('_', ' ')}</span>
              <span>{Math.round(Number(value) * 100)}%</span>
            </div>
            <div className="w-full h-1.5 rounded bg-slate-800 overflow-hidden">
              <div className="h-full bg-blue-500" style={{ width: `${Math.round(Number(value) * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
