export type Severity = 'critical' | 'warning' | 'info';

export interface DashboardTrace {
  step?: number;
  followed_agent?: string | null;
  action_taken?: string;
  reasoning?: string;
  override_applied?: boolean;
  override_reason?: string | null;
  predicted_2step_impact?: Record<string, number>;
  schema_drift_detected?: boolean;
  schema_drift_field?: string | null;
}

export interface RewardRecord {
  step: number;
  r1: number;
  r2: number;
  r3: number;
  r4: number;
  total: number;
  [key: string]: number;
}

export interface DashboardStep {
  step: number;
  metrics: Record<string, number>;
  health: number;
  action?: string;
  override_applied?: boolean;
  adv_was_correct?: boolean;
  trust_scores: Record<string, number>;
  reward?: RewardRecord;
  trace?: DashboardTrace;
  drift_active?: boolean;
}

export interface DashboardEpisode {
  episode_id: number;
  total_reward: number;
  r2_bonus?: number;
  final_health: number;
  mttr: number;
  scenario_name: string;
  reward_history?: RewardRecord[];
  trust_evolution?: Array<Record<string, number>>;
  trajectory: DashboardStep[];
}

export interface DashboardModeData {
  available_episodes: number[];
  selected_episode_id: number;
  episodes: Record<string, DashboardEpisode>;
}

export interface DashboardDataPayload {
  generated_at: string;
  source: string;
  mode: 'trained' | 'untrained';
  modes: {
    trained: DashboardModeData;
    untrained: DashboardModeData;
  };
}

export function metricToSeverity(value: number, target = 0.7): Severity {
  if (value >= target * 1.35) return 'critical';
  if (value >= target * 1.1) return 'warning';
  return 'info';
}
