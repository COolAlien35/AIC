import { motion } from 'motion/react';
import { MoreVertical } from 'lucide-react';

export function TrustScoreCard() {
  const score = 72;
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="panel-geometric p-6 flex flex-col items-center justify-between relative overflow-hidden h-full">
      <div className="absolute top-6 right-6">
        <MoreVertical size={16} className="text-slate-600 cursor-pointer hover:text-white" />
      </div>

      <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">AI Trust Score</h3>

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
          <p className="text-[11px] font-bold uppercase tracking-tighter text-emerald-400">High Confidence</p>
        </div>
        <p className="text-[10px] text-slate-500">Automated remediation suggested</p>
      </div>

      <button className="w-full mt-6 py-2 rounded-xl bg-blue-600 text-[10px] font-bold uppercase tracking-widest text-white shadow-lg shadow-blue-900/20 hover:bg-blue-700 transition-all active:scale-[0.98]">
        Review Recommendations
      </button>
    </div>
  );
}
