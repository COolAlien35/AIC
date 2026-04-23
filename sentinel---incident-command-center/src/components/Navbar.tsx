import { CalendarRange, Gauge, RefreshCcw } from 'lucide-react';

interface NavbarProps {
  mode: 'trained' | 'untrained';
  availableEpisodes: number[];
  episodeId: number;
  onEpisodeChange: (value: number) => void;
  onModeChange: (value: 'trained' | 'untrained') => void;
  step: number;
  maxStep: number;
  onStepChange: (value: number) => void;
  generatedAt?: string;
  onReload: () => void;
}

export function Navbar({
  mode,
  availableEpisodes,
  episodeId,
  onEpisodeChange,
  onModeChange,
  step,
  maxStep,
  onStepChange,
  generatedAt,
  onReload,
}: NavbarProps) {
  return (
    <header className="border-b border-slate-800 bg-[#0A0A0B]/90 backdrop-blur-md px-6 py-4 z-40">
      <div className="flex flex-wrap items-end gap-4 justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Adaptive Incident Choreographer</p>
          <h1 className="text-xl font-semibold text-white tracking-tight">Sentinel Incident Command Center</h1>
          {generatedAt && <p className="text-xs text-slate-500 mt-1">Data refreshed: {new Date(generatedAt).toLocaleString()}</p>}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 bg-slate-900/40">
            <Gauge size={14} className="text-slate-400" />
            <select
              className="bg-transparent text-sm text-slate-200 outline-none"
              value={mode}
              onChange={(e) => onModeChange(e.target.value as 'trained' | 'untrained')}
            >
              <option value="trained">Trained</option>
              <option value="untrained">Untrained</option>
            </select>
          </div>

          <div className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 bg-slate-900/40">
            <CalendarRange size={14} className="text-slate-400" />
            <select
              className="bg-transparent text-sm text-slate-200 outline-none"
              value={episodeId}
              onChange={(e) => onEpisodeChange(Number(e.target.value))}
            >
              {availableEpisodes.map((ep) => (
                <option key={ep} value={ep}>Episode {ep}</option>
              ))}
            </select>
          </div>

          <button
            onClick={onReload}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800/60"
          >
            <RefreshCcw size={14} />
            Reload JSON
          </button>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/30 px-4 py-3">
        <span className="text-xs uppercase tracking-widest text-slate-500">Timeline Step</span>
        <div className="flex items-center gap-3 flex-1">
          <input
            type="range"
            min={0}
            max={Math.max(maxStep, 0)}
            value={Math.min(step, Math.max(maxStep, 0))}
            onChange={(e) => onStepChange(Number(e.target.value))}
            className="w-full accent-blue-600"
          />
          <span className="font-mono text-sm text-slate-200 min-w-20 text-right">{step} / {Math.max(maxStep, 0)}</span>
        </div>
      </div>
    </header>
  );
}
