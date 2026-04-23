import { Activity, BrainCircuit, ShieldCheck, Siren } from 'lucide-react';

export function Sidebar() {
  return (
    <aside className="w-64 h-screen flex flex-col bg-[#0F0F12] border-r border-slate-800 z-50">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-900/20">
          <BrainCircuit size={20} className="text-white" />
        </div>
        <span className="font-bold text-xl tracking-tight text-white">Sentinel</span>
      </div>

      <div className="px-4 mt-2 space-y-3">
        <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">War Room Scope</p>
          <div className="space-y-3 text-sm text-slate-300">
            <div className="flex items-center gap-2"><Activity size={14} className="text-emerald-400" /> Live metrics + health timeline</div>
            <div className="flex items-center gap-2"><Siren size={14} className="text-red-400" /> Incident feed from current step</div>
            <div className="flex items-center gap-2"><ShieldCheck size={14} className="text-blue-400" /> Trust and override visibility</div>
          </div>
        </div>
      </div>

      <div className="mt-auto p-4 border-t border-slate-800 text-xs text-slate-500">
        AIC data-backed dashboard
      </div>
    </aside>
  );
}
