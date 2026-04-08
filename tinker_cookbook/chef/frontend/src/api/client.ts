/** API client for Tinker Chef backend. */

import type {
  CheckpointRecord,
  IterationInfo,
  LogtreeResponse,
  MetricsResponse,
  RolloutDetail,
  RolloutsResponse,
  RunInfo,
  TimingResponse,
} from './types';

const BASE = '/api';

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  // Runs
  listRuns: () => fetchJSON<RunInfo[]>(`${BASE}/runs`),
  getRun: (runId: string) => fetchJSON<RunInfo>(`${BASE}/runs/${runId}`),
  getConfig: (runId: string) => fetchJSON<Record<string, unknown>>(`${BASE}/runs/${runId}/config`),

  // Iterations
  listIterations: (runId: string) =>
    fetchJSON<IterationInfo[]>(`${BASE}/runs/${runId}/iterations`),

  // Metrics
  getMetrics: (runId: string, keys?: string) => {
    const params = keys ? `?keys=${encodeURIComponent(keys)}` : '';
    return fetchJSON<MetricsResponse>(`${BASE}/runs/${runId}/metrics${params}`);
  },
  getMetricKeys: (runId: string) =>
    fetchJSON<string[]>(`${BASE}/runs/${runId}/metrics/keys`),

  // Rollouts
  getRollouts: (
    runId: string,
    iteration: number,
    params?: { split?: string; label?: string; tag?: string; min_reward?: number; max_reward?: number }
  ) => {
    const searchParams = new URLSearchParams();
    if (params?.split) searchParams.set('split', params.split);
    if (params?.label) searchParams.set('label', params.label);
    if (params?.tag) searchParams.set('tag', params.tag);
    if (params?.min_reward !== undefined) searchParams.set('min_reward', String(params.min_reward));
    if (params?.max_reward !== undefined) searchParams.set('max_reward', String(params.max_reward));
    const qs = searchParams.toString();
    return fetchJSON<RolloutsResponse>(
      `${BASE}/runs/${runId}/iterations/${iteration}/rollouts${qs ? '?' + qs : ''}`
    );
  },
  getRolloutDetail: (runId: string, iteration: number, groupIdx: number, trajIdx: number) =>
    fetchJSON<RolloutDetail>(
      `${BASE}/runs/${runId}/iterations/${iteration}/rollouts/${groupIdx}/${trajIdx}`
    ),

  // Logtree
  getLogtree: (runId: string, iteration: number, baseName?: string) => {
    const params = baseName ? `?base_name=${encodeURIComponent(baseName)}` : '';
    return fetchJSON<LogtreeResponse>(
      `${BASE}/runs/${runId}/iterations/${iteration}/logtree${params}`
    );
  },

  // Timing
  getTiming: (runId: string, stepStart?: number, stepEnd?: number) => {
    const params = new URLSearchParams();
    if (stepStart !== undefined) params.set('step_start', String(stepStart));
    if (stepEnd !== undefined) params.set('step_end', String(stepEnd));
    const qs = params.toString();
    return fetchJSON<TimingResponse>(`${BASE}/runs/${runId}/timing${qs ? '?' + qs : ''}`);
  },

  // Checkpoints
  getCheckpoints: (runId: string) =>
    fetchJSON<CheckpointRecord[]>(`${BASE}/runs/${runId}/checkpoints`),

  // Eval scores matched to training checkpoints
  getEvalScores: (runId: string) =>
    fetchJSON<import('./types').EvalScorePoint[]>(`${BASE}/runs/${runId}/eval-scores`),

  // SSE stream URL (for EventSource)
  metricsStreamUrl: (runId: string) => `${BASE}/runs/${runId}/metrics/stream`,

  // Eval benchmarks
  listEvalRuns: () =>
    fetchJSON<import('./types').EvalRunSummary[]>(`${BASE}/eval/runs`),
  getEvalRun: (evalRunId: string) =>
    fetchJSON<import('./types').EvalRunDetail>(`${BASE}/eval/runs/${evalRunId}`),
  getEvalTrajectories: (evalRunId: string, benchmark: string, params?: { correct_only?: boolean; incorrect_only?: boolean; errors_only?: boolean }) => {
    const searchParams = new URLSearchParams();
    if (params?.correct_only) searchParams.set('correct_only', 'true');
    if (params?.incorrect_only) searchParams.set('incorrect_only', 'true');
    if (params?.errors_only) searchParams.set('errors_only', 'true');
    const qs = searchParams.toString();
    return fetchJSON<import('./types').EvalTrajectoriesResponse>(
      `${BASE}/eval/runs/${evalRunId}/${benchmark}/trajectories${qs ? '?' + qs : ''}`
    );
  },
  getEvalTrajectoryDetail: (evalRunId: string, benchmark: string, idx: number) =>
    fetchJSON<import('./types').EvalTrajectoryDetail>(
      `${BASE}/eval/runs/${evalRunId}/${benchmark}/trajectories/${idx}`
    ),
  getScoresTable: () =>
    fetchJSON<import('./types').ScoresTableRow[]>(`${BASE}/eval/scores`),
};
