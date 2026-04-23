import { useEffect, useMemo, useState } from 'react';
import type { DashboardDataPayload, DashboardEpisode, DashboardStep } from './dashboardData';

const FALLBACK_ERROR = 'No dashboard data found. Run data export script first.';

export function useDashboardData() {
  const [payload, setPayload] = useState<DashboardDataPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'trained' | 'untrained'>('trained');
  const [episodeId, setEpisodeId] = useState<number | null>(null);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      try {
        const response = await fetch('/data/dashboard-data.json');
        if (!response.ok) {
          throw new Error(FALLBACK_ERROR);
        }
        const data = (await response.json()) as DashboardDataPayload;
        if (!isMounted) return;
        setPayload(data);
        setMode(data.mode ?? 'trained');
        const defaultEpisode = data.modes?.[data.mode]?.selected_episode_id ?? 0;
        setEpisodeId(defaultEpisode);
      } catch (e) {
        if (!isMounted) return;
        setError(e instanceof Error ? e.message : FALLBACK_ERROR);
      }
    };
    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const modeData = useMemo(() => payload?.modes?.[mode] ?? null, [payload, mode]);

  const availableEpisodes = modeData?.available_episodes ?? [];
  const resolvedEpisodeId = useMemo(() => {
    if (!availableEpisodes.length) return null;
    if (episodeId !== null && availableEpisodes.includes(episodeId)) return episodeId;
    return availableEpisodes[0];
  }, [availableEpisodes, episodeId]);

  const episode: DashboardEpisode | null = useMemo(() => {
    if (!modeData || resolvedEpisodeId === null) return null;
    return modeData.episodes[String(resolvedEpisodeId)] ?? null;
  }, [modeData, resolvedEpisodeId]);

  const currentStep: DashboardStep | null = useMemo(() => {
    if (!episode || episode.trajectory.length === 0) return null;
    const idx = Math.max(0, Math.min(stepIndex, episode.trajectory.length - 1));
    return episode.trajectory[idx];
  }, [episode, stepIndex]);

  useEffect(() => {
    setStepIndex(0);
  }, [mode, resolvedEpisodeId]);

  return {
    payload,
    mode,
    setMode,
    availableEpisodes,
    episodeId: resolvedEpisodeId,
    setEpisodeId,
    episode,
    currentStep,
    stepIndex,
    setStepIndex,
    maxStep: (episode?.trajectory.length ?? 1) - 1,
    error,
  };
}
