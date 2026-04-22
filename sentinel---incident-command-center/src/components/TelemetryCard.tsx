import { 
  AreaChart, 
  Area, 
  ResponsiveContainer, 
  XAxis, 
  YAxis, 
  Tooltip,
  BarChart,
  Bar
} from 'recharts';
import { MoreHorizontal } from 'lucide-react';

const generateData = (points: number, min: number, max: number) => {
  return Array.from({ length: points }, (_, i) => ({
    time: i,
    value: Math.floor(Math.random() * (max - min + 1) + min)
  }));
};

const cpuData = generateData(20, 40, 90);
const latencyData = generateData(20, 200, 800);
const errorData = generateData(20, 0, 5);

function Sparkline({ data, color }: { data: any[], color: string }) {
  return (
    <div className="h-10 w-32">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`gradient-${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <Area 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            fillOpacity={1} 
            fill={`url(#gradient-${color})`} 
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TelemetryCard() {
  return (
    <div className="panel-geometric p-6 flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Live Telemetry</h3>
        <MoreHorizontal size={16} className="text-slate-600 cursor-pointer hover:text-white" />
      </div>

      <div className="space-y-6 flex-1">
        {/* CPU Usage */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] text-slate-500 font-bold uppercase tracking-tighter">CPU Load</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xl font-bold text-white">85%</span>
              <span className="text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-1 py-0.5 rounded">+2.4%</span>
            </div>
          </div>
          <Sparkline data={cpuData} color="#2563eb" />
        </div>

        {/* Latency */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] text-slate-500 font-bold uppercase tracking-tighter">Latency</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xl font-bold text-white">540ms</span>
              <span className="text-[10px] text-red-400 font-bold bg-red-500/10 px-1 py-0.5 rounded">+12%</span>
            </div>
          </div>
          <Sparkline data={latencyData} color="#059669" />
        </div>
      </div>

      <div className="pt-4 border-t border-slate-800">
        <div className="h-12 flex items-end gap-1 px-1">
          {Array.from({ length: 32 }).map((_, i) => (
            <div 
              key={i} 
              className="flex-1 bg-blue-600/20 rounded-t-sm transition-all hover:bg-blue-600/50"
              style={{ height: `${20 + Math.random() * 80}%` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
