import { Search, Bell, Settings } from 'lucide-react';

export function Navbar() {
  return (
    <header className="h-16 border-b border-slate-800 bg-[#0A0A0B]/80 backdrop-blur-md flex items-center justify-between px-8 z-40">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-slate-500">Sentinel</span>
        <span className="text-slate-700">/</span>
        <span className="text-white font-medium tracking-tight">Incident Command Center</span>
      </div>

      <div className="flex items-center gap-6">
        <div className="relative group">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-hover:text-blue-400 transition-colors" size={14} />
          <input 
            type="text" 
            placeholder="Quick search infrastructure..." 
            className="bg-slate-900 border border-slate-800 rounded-full py-1.5 pl-10 pr-4 text-xs w-64 focus:outline-none focus:border-blue-600 transition-all text-slate-300"
          />
        </div>

        <div className="flex items-center gap-4 pl-6 border-l border-slate-800">
          <button className="relative p-2 text-slate-400 hover:text-white transition-colors">
            <Bell size={18} />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-600 rounded-full border-2 border-[#0A0A0B]" />
          </button>
          <div className="w-px h-4 bg-slate-800" />
          <button className="p-2 text-slate-400 hover:text-white transition-colors">
            <Settings size={18} />
          </button>
        </div>
      </div>
    </header>
  );
}
