import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { SystemTopology } from './components/SystemTopology';
import { TelemetryCard } from './components/TelemetryCard';
import { TrustScoreCard } from './components/TrustScoreCard';
import { LiveIncidentsFeed } from './components/LiveIncidentsFeed';
import { IncidentTimeline } from './components/IncidentTimeline';
import { ReliabilityLeaderboard } from './components/ReliabilityLeaderboard';
import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';
import { LayoutGrid, List } from 'lucide-react';

export default function App() {
  const [view, setView] = useState<'topology' | 'investigation'>('topology');

  return (
    <div className="flex h-screen overflow-hidden bg-[#0A0A0B] text-slate-300 font-sans">
      <Sidebar />
      
      <main className="flex-1 flex flex-col min-w-0 bg-[#0A0A0B]">
        <Navbar />
        
        {/* View Switcher Overlay - Geometric Style */}
        <div className="absolute top-[88px] right-8 z-30 flex items-center gap-1 p-1 bg-slate-900 border border-slate-800 rounded-lg shadow-xl">
          <button 
            onClick={() => setView('topology')}
            className={`p-1.5 rounded transition-all ${view === 'topology' ? 'bg-blue-600 text-white shadow-md shadow-blue-900/20' : 'text-slate-500 hover:text-slate-200'}`}
          >
            <LayoutGrid size={16} />
          </button>
          <button 
             onClick={() => setView('investigation')}
            className={`p-1.5 rounded transition-all ${view === 'investigation' ? 'bg-blue-600 text-white shadow-md shadow-blue-900/20' : 'text-slate-500 hover:text-slate-200'}`}
          >
            <List size={16} />
          </button>
        </div>

        <div className="flex-1 p-8 pt-6 overflow-y-auto overflow-x-hidden">
          <AnimatePresence mode="wait">
            {view === 'topology' ? (
              <motion.div 
                key="topology"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-12 grid-rows-12 gap-6 h-[calc(100vh-140px)]"
              >
                {/* Main Content Area */}
                <div className="col-span-12 lg:col-span-8 row-span-8 flex flex-col">
                  <SystemTopology />
                </div>

                {/* Right Column Top */}
                <div className="col-span-12 lg:col-span-4 row-span-6">
                  <TelemetryCard />
                </div>

                {/* Right Column Bottom */}
                <div className="col-span-12 lg:col-span-4 row-span-6">
                  <TrustScoreCard />
                </div>

                {/* Bottom Left Feed */}
                <div className="col-span-12 lg:col-span-8 row-span-4">
                  <LiveIncidentsFeed />
                </div>
              </motion.div>
            ) : (
              <motion.div 
                key="investigation"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-12 gap-6"
              >
                <div className="col-span-12 lg:col-span-3">
                  <IncidentTimeline />
                </div>
                <div className="col-span-12 lg:col-span-6 space-y-6">
                  <div className="panel-geometric p-6 min-h-[300px]">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-6">AI Remediation Recommendations</h3>
                    <div className="space-y-4">
                      {[1, 2, 3].map(i => (
                        <div key={i} className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 hover:bg-slate-800/20 transition-colors flex items-center justify-between group">
                          <div>
                            <p className="text-sm font-semibold text-white group-hover:text-blue-400 transition-colors tracking-tight">Recommendation 0{i}: Cluster Node Scale-out</p>
                            <p className="text-[10px] text-slate-600 mt-1 uppercase font-bold tracking-wider">Priority: High • Impact: Low Risk</p>
                          </div>
                          <div className="flex items-center gap-2">
                             <button className="px-4 py-1.5 rounded-lg bg-blue-600 text-[10px] font-bold uppercase tracking-widest text-white shadow-lg shadow-blue-900/20 hover:bg-blue-700 transition-all">Execute</button>
                             <button className="px-4 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-[10px] font-bold uppercase tracking-widest text-slate-400 hover:text-white transition-all">Audit</button>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="mt-8">
                       <h4 className="text-[11px] font-bold uppercase text-slate-600 tracking-widest mb-4">Service Correlation - Historical Latency</h4>
                       <div className="h-44 bg-slate-950/40 rounded-xl border border-slate-800 relative overflow-hidden">
                          <div className="p-4 flex items-end justify-between gap-1 h-full">
                            {Array.from({length: 60}).map((_, j) => (
                              <div 
                                key={j} 
                                className="flex-1 bg-slate-800 rounded-t-sm hover:bg-blue-600 transition-colors" 
                                style={{ height: `${10 + Math.random() * 80}%` }}
                              />
                            ))}
                          </div>
                       </div>
                    </div>
                  </div>
                </div>
                <div className="col-span-12 lg:col-span-3 space-y-6">
                  <ReliabilityLeaderboard />
                  <div className="panel-geometric p-6">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Command Personnel</h3>
                    <div className="space-y-4">
                      {['Sarah K.', 'Alex Chen', 'Mike Wu'].map(name => (
                        <div key={name} className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-900 transition-colors cursor-pointer">
                          <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-[10px] font-bold text-slate-400 uppercase">
                            {name.split(' ').map(n => n[0]).join('')}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-semibold text-white tracking-tight">{name}</p>
                            <p className="text-[10px] text-slate-600 uppercase font-bold tracking-tighter">On Call • Duty Shift A</p>
                          </div>
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                        </div>
                      ))}
                    </div>
                    <button className="w-full mt-6 py-2.5 bg-blue-600 text-[10px] font-bold uppercase tracking-widest text-white rounded-lg shadow-lg shadow-blue-900/20 hover:bg-blue-700 transition-all">Join Secure War Room</button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
