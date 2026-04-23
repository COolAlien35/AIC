import { motion } from 'motion/react';
import { 
  Server, 
  Database, 
  Globe, 
  Lock, 
  CreditCard,
  AlertCircle,
  Activity,
} from 'lucide-react';
import { cn } from '@/src/lib/utils';
import type { DashboardStep } from '@/src/data/dashboardData';
import { metric } from '@/src/data/metrics';

interface NodeProps {
  icon: any;
  label: string;
  sublabel?: string;
  status: 'healthy' | 'warning' | 'critical';
  x: number;
  y: number;
}

function TopologyNode({ icon: Icon, label, sublabel, status, x, y }: NodeProps) {
  const statusColors = {
    healthy: 'bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.8)]',
    warning: 'bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.8)]',
    critical: 'bg-red-500 shadow-[0_0_20px_rgba(239,68,68,1)]'
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="absolute flex flex-col items-center gap-3 group"
      style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%, -50%)' }}
    >
      <div className="relative">
        <div className={cn(
          "w-12 h-12 rounded-xl flex items-center justify-center glass-panel group-hover:scale-110 transition-transform cursor-pointer",
          status === 'critical' ? 'border-red-500/50' : 'border-white/10'
        )}>
          <Icon size={24} className={cn(
            status === 'healthy' ? 'text-emerald-400' : 
            status === 'warning' ? 'text-amber-400' : 'text-red-400'
          )} />
        </div>
        <div className={cn("absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-[#16171d]", statusColors[status])} />
        
        {status === 'critical' && (
          <motion.div 
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.6, 0.3] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="absolute inset-0 bg-red-500/20 blur-xl rounded-full"
          />
        )}
      </div>
      
      <div className="text-center">
        <p className="text-[11px] font-semibold text-white/90 whitespace-nowrap">{label}</p>
        {sublabel && (
          <p className={cn(
            "text-[9px] uppercase tracking-tighter whitespace-nowrap",
            status === 'critical' ? 'text-red-400 font-bold' : 'text-slate-500'
          )}>
            {sublabel}
          </p>
        )}
      </div>
    </motion.div>
  );
}

function statusFromValue(value: number, warnThreshold: number, criticalThreshold: number): 'healthy' | 'warning' | 'critical' {
  if (value >= criticalThreshold) return 'critical';
  if (value >= warnThreshold) return 'warning';
  return 'healthy';
}

export function SystemTopology({ step }: { step: DashboardStep | null }) {
  const gatewayStatus = statusFromValue(metric(step, 'network'), 280, 360);
  const appStatus = statusFromValue(metric(step, 'latency'), 1500, 2200);
  const dbStatus = statusFromValue(metric(step, 'dbLatency'), 700, 1000);
  const queueStatus = statusFromValue(metric(step, 'queue'), 500, 800);
  const cpuStatus = statusFromValue(metric(step, 'cpu'), 70, 85);

  return (
    <div className="panel-geometric p-6 flex-1 flex flex-col relative overflow-hidden bg-slate-950/20">
      <div className="flex items-center justify-between mb-8">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Infrastructure Topology</h3>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px] uppercase text-slate-600 font-bold">
              Health {Math.round((step?.health ?? 0) * 100)}%
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 relative">
        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-5">
           <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
             <path d="M 40 0 L 0 0 0 40" fill="none" stroke="white" strokeWidth="0.5"/>
           </pattern>
           <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
          <path d="M 300 100 L 450 100" stroke="white" strokeWidth="1" fill="none" />
          <path d="M 450 100 L 450 250" stroke="white" strokeWidth="1" fill="none" strokeDasharray="4 4" />
          <path d="M 450 250 L 300 350" stroke="white" strokeWidth="1" fill="none" />
          <path d="M 150 250 L 300 350" stroke="#dc2626" strokeWidth="2" fill="none" />
          <path d="M 150 250 L 300 250" stroke="white" strokeWidth="1" fill="none" />
          <path d="M 600 250 L 750 250" stroke="white" strokeWidth="1" fill="none" />
          <path d="M 450 250 L 600 250" stroke="white" strokeWidth="1" fill="none" />
        </svg>

        <TopologyNode icon={CreditCard} label="Payment Gateway" status={gatewayStatus} x={30} y={30} />
        <TopologyNode icon={Lock} label="Auth Service" status={cpuStatus} x={50} y={30} />
        
        <TopologyNode icon={AlertCircle} label="Global Gateway" sublabel="Traffic" status={gatewayStatus} x={15} y={60} />
        <TopologyNode icon={AlertCircle} label="Auth API" status={appStatus} x={35} y={60} />
        <TopologyNode icon={Database} label="DB-Cluster-01" status={dbStatus} x={55} y={60} />
        <TopologyNode icon={Globe} label="Cloud CDN" status={gatewayStatus} x={75} y={60} />
        <TopologyNode icon={Server} label="Frontends" status={cpuStatus} x={90} y={60} />

        <TopologyNode icon={Activity} label="Service Cluster" status={queueStatus} x={45} y={85} />
      </div>
    </div>
  );
}
