import { motion } from 'motion/react';
import { 
  LayoutDashboard, 
  AlertTriangle, 
  BrainCircuit, 
  LineChart, 
  Settings,
  ChevronRight
} from 'lucide-react';
import { cn } from '@/src/lib/utils';
import { useState } from 'react';

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', id: 'dashboard' },
  { icon: AlertTriangle, label: 'Incidents', id: 'incidents' },
  { icon: BrainCircuit, label: 'AI Actions', id: 'ai' },
  { icon: LineChart, label: 'Analytics', id: 'analytics' },
  { icon: Settings, label: 'Settings', id: 'settings' },
];

export function Sidebar() {
  const [active, setActive] = useState('dashboard');

  return (
    <aside className="w-64 h-screen flex flex-col bg-[#0F0F12] border-r border-slate-800 z-50">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-900/20">
          <BrainCircuit size={20} className="text-white" />
        </div>
        <span className="font-bold text-xl tracking-tight text-white">Sentinel</span>
      </div>
      
      <nav className="flex-1 px-4 space-y-1 mt-4">
        {navItems.map((item) => {
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActive(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-medium transition-all duration-200",
                isActive ? "nav-active" : "nav-hover"
              )}
            >
              <item.icon size={20} />
              <span className="text-sm">
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>

      <div className="p-4">
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
          <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-2">System Status</p>
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-white font-medium">Uptime: 99.98%</p>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: '99%' }}
              className="bg-blue-600 h-full" 
            />
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-slate-800 flex items-center gap-3">
        <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-xs font-bold text-white uppercase">
          SK
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-white leading-none">Sarah K.</p>
          <p className="text-[10px] text-slate-500 mt-1 uppercase">Account Admin</p>
        </div>
      </div>
    </aside>
  );
}
