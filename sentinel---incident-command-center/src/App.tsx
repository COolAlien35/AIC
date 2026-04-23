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
import { useDashboardData } from './data/useDashboardData';
import { metric } from './data/metrics';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export default function App() {
  const [view, setView] = useState<'overview' | 'investigation'>('overview');
  const {
    payload,
    mode,
    setMode,
    availableEpisodes,
    episodeId,
    setEpisodeId,
    episode,
    currentStep,
    stepIndex,
    setStepIndex,
    maxStep,
    error,
  } = useDashboardData();

  if (error) {
    return (
      <div className="min-h-screen bg-[#0A0A0B] text-slate-200 flex items-center justify-center p-6">
        <div className="panel-geometric p-8 max-w-xl">
          <h2 className="text-lg font-semibold text-white">Dashboard data unavailable</h2>
          <p className="text-sm text-slate-400 mt-2">{error}</p>
          <p className="text-xs text-slate-500 mt-4">
            Run <code>python scripts/export_sentinel_data.py</code> from project root, then restart the UI.
          </p>
        </div>
      </div>
    );
  }

  const trajectory = episode?.trajectory ?? [];
  const chartData = trajectory.map((step) => ({
    step: step.step,
    health: Number((step.health * 100).toFixed(2)),
    reward: Number(step.reward?.total ?? 0),
    latency: metric(step, 'latency'),
    errors: metric(step, 'errors'),
    cpu: metric(step, 'cpu'),
  }));

  const selectedReward = Number(currentStep?.reward?.total ?? 0);

  return (
    <div className="flex h-screen overflow-hidden bg-[#0A0A0B] text-slate-300 font-sans">
      <Sidebar />
      
      <main className="flex-1 flex flex-col min-w-0 bg-[#0A0A0B]">
        <Navbar
          mode={mode}
          availableEpisodes={availableEpisodes}
          episodeId={episodeId ?? 0}
          onEpisodeChange={setEpisodeId}
          onModeChange={setMode}
          step={stepIndex}
          maxStep={maxStep}
          onStepChange={setStepIndex}
          generatedAt={payload?.generated_at}
          onReload={() => window.location.reload()}
        />
        
        <div className="absolute top-[142px] right-8 z-30 flex items-center gap-1 p-1 bg-slate-900 border border-slate-800 rounded-lg shadow-xl">
          <button 
            onClick={() => setView('overview')}
            className={`p-1.5 rounded transition-all ${view === 'overview' ? 'bg-blue-600 text-white shadow-md shadow-blue-900/20' : 'text-slate-500 hover:text-slate-200'}`}
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

        <div className="flex-1 p-6 overflow-y-auto overflow-x-hidden">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div className="card-geometric p-4">
              <p className="text-[10px] uppercase tracking-widest text-slate-500">Scenario</p>
              <p className="text-sm text-white mt-2">{episode?.scenario_name ?? 'Unknown'}</p>
            </div>
            <div className="card-geometric p-4">
              <p className="text-[10px] uppercase tracking-widest text-slate-500">Health</p>
              <p className="text-2xl font-semibold text-white mt-1">{((currentStep?.health ?? 0) * 100).toFixed(1)}%</p>
            </div>
            <div className="card-geometric p-4">
              <p className="text-[10px] uppercase tracking-widest text-slate-500">Step Reward</p>
              <p className={`text-2xl font-semibold mt-1 ${selectedReward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {selectedReward.toFixed(2)}
              </p>
            </div>
            <div className="card-geometric p-4">
              <p className="text-[10px] uppercase tracking-widest text-slate-500">Episode Total Reward</p>
              <p className={`text-2xl font-semibold mt-1 ${(episode?.total_reward ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(episode?.total_reward ?? 0).toFixed(2)}
              </p>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {view === 'overview' ? (
              <motion.div 
                key="overview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-12 gap-4"
              >
                <div className="col-span-12 lg:col-span-8">
                  <SystemTopology step={currentStep} />
                </div>
                <div className="col-span-12 lg:col-span-4">
                  <TrustScoreCard step={currentStep} />
                </div>
                <div className="col-span-12 lg:col-span-8 panel-geometric p-4 h-[280px]">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Health & Reward Timeline</h3>
                  </div>
                  <ResponsiveContainer width="100%" height="92%">
                    <LineChart data={chartData.slice(0, stepIndex + 1)}>
                      <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                      <XAxis dataKey="step" stroke="#64748b" />
                      <YAxis yAxisId="left" stroke="#64748b" />
                      <YAxis yAxisId="right" orientation="right" stroke="#64748b" />
                      <Tooltip contentStyle={{ background: '#020617', border: '1px solid #334155' }} />
                      <Line yAxisId="left" type="monotone" dataKey="health" stroke="#3b82f6" strokeWidth={2} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="reward" stroke="#10b981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="col-span-12 lg:col-span-4">
                  <TelemetryCard step={currentStep} trajectory={trajectory} stepIndex={stepIndex} />
                </div>

                <div className="col-span-12 lg:col-span-8">
                  <LiveIncidentsFeed step={currentStep} />
                </div>
                <div className="col-span-12 lg:col-span-4">
                  <ReliabilityLeaderboard step={currentStep} />
                </div>
              </motion.div>
            ) : (
              <motion.div 
                key="investigation"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-12 gap-4"
              >
                <div className="col-span-12 lg:col-span-3">
                  <IncidentTimeline episode={episode} step={stepIndex} />
                </div>
                <div className="col-span-12 lg:col-span-6 space-y-4">
                  <div className="panel-geometric p-6">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Action Reasoning</h3>
                    <p className="text-sm text-white leading-relaxed">{currentStep?.trace?.action_taken ?? currentStep?.action ?? 'No action available'}</p>
                    <p className="text-xs text-slate-400 mt-3 leading-relaxed">{currentStep?.trace?.reasoning ?? 'No reasoning recorded for this step.'}</p>
                    {!!currentStep?.trace?.override_reason && (
                      <div className="mt-3 text-xs rounded-lg border border-amber-700/50 bg-amber-900/20 p-3 text-amber-200">
                        Override: {currentStep.trace.override_reason}
                      </div>
                    )}
                  </div>
                  <div className="panel-geometric p-4 h-[260px]">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Latency vs Errors</h3>
                    <ResponsiveContainer width="100%" height="88%">
                      <BarChart data={chartData.slice(0, stepIndex + 1)}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                        <XAxis dataKey="step" stroke="#64748b" />
                        <YAxis stroke="#64748b" />
                        <Tooltip contentStyle={{ background: '#020617', border: '1px solid #334155' }} />
                        <Bar dataKey="latency" fill="#2563eb" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="errors" fill="#ef4444" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="panel-geometric p-6">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Predicted 2-Step Impact</h3>
                    <div className="space-y-2 max-h-44 overflow-auto pr-2">
                      {Object.entries(currentStep?.trace?.predicted_2step_impact ?? {}).length === 0 && (
                        <p className="text-xs text-slate-500">No predicted impact values at this step.</p>
                      )}
                      {Object.entries(currentStep?.trace?.predicted_2step_impact ?? {}).map(([name, value]) => (
                        <div key={name} className="flex items-center justify-between rounded-md border border-slate-800 px-3 py-2">
                          <span className="text-xs text-slate-300">{name}</span>
                          <span className={`text-xs font-mono ${Number(value) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {Number(value).toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="col-span-12 lg:col-span-3 space-y-4">
                  <ReliabilityLeaderboard step={currentStep} />
                  <div className="panel-geometric p-6">
                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Trace Flags</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Schema Drift</span>
                        <span className={currentStep?.trace?.schema_drift_detected ? 'text-red-400' : 'text-emerald-400'}>
                          {currentStep?.trace?.schema_drift_detected ? 'Detected' : 'No'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Override Applied</span>
                        <span className={currentStep?.override_applied ? 'text-amber-400' : 'text-emerald-400'}>
                          {currentStep?.override_applied ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Adversary Correct</span>
                        <span className={currentStep?.adv_was_correct ? 'text-red-400' : 'text-emerald-400'}>
                          {currentStep?.adv_was_correct ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>
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
