// Typed HTTP client for the token DB viewer server.
//
// The server runs in one of two modes:
//   - single-run mode: endpoints at /api/* and the live socket at /ws
//   - registry mode: /api/runs lists runs, /api/dashboard aggregates them,
//     and every single-run endpoint is also mounted at /api/runs/{run_id}/*
// `detectMode()` probes once at app start; `apiBase(runId)` / `wsPath(runId)`
// pick the right prefix for data calls.

export interface StepRow {
  run_id: string;
  run_attempt: number;
  split: string;
  iteration: number;
  sampling_client_step: number | null;
  group_idx: number;
  traj_idx: number;
  step_idx: number;
  tags: string[];
  env_row_id: string | null;
  ts: string;
  source: string;
  ob_tokens: number[];
  ob_is_delta: boolean;
  ac_tokens: number[];
  ac_logprobs: number[] | null;
  stop_reason: string | null;
  reward: number;
  episode_done: boolean;
  total_reward: number;
  final_reward: number;
  ob_text: string | null;
  ac_text: string | null;
  metrics: string;
  logs: string;
  extra: string;
  filtered_reason: string | null;
  superseded: boolean;
  // Added by the detail endpoint:
  ob_full_tokens?: number[];
  ac_token_strs?: string[];
}

export interface TrajectoryRow {
  run_id: string;
  run_attempt: number;
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  n_steps: number;
  n_ac_tokens: number;
  total_reward: number;
  final_reward: number;
  stop_reason: string | null;
  filtered_reason: string | null;
  env_row_id: string | null;
  tags: string[];
  source: string;
  sampling_client_step: number | null;
  ac_preview: string | null;
  superseded: boolean;
  ts: string;
}

export interface Label {
  run_id: string | null;
  split: string | null;
  iteration: number | null;
  group_idx: number | null;
  traj_idx: number | null;
  step_idx: number | null;
  label_key: string;
  label_value: unknown;
  author: string;
  ts: string;
  note: string | null;
}

export interface RolloutDetail {
  steps: StepRow[];
  labels: Label[];
  group_traj_idxs: number[];
}

export interface RunInfo {
  run_id: string;
  run_attempt: number;
  context?: { model_name?: string };
}

export interface RewardPoint {
  iteration: number;
  mean_total_reward: number | null;
}

export interface DashboardRun {
  run_id: string;
  run_attempt: number;
  log_path: string;
  model_name: string | null;
  recipe_name: string | null;
  started_at: string | null;
  live: boolean;
  last_activity_ts: number | null;
  latest_iteration: number | null;
  n_rows: number | null;
  n_filtered_rows: number | null;
  mean_recent_reward: number | null;
  reward_series: RewardPoint[];
  error?: string;
}

export type Mode = "single" | "registry";

/** Probe the server once at app start: registry mode serves /api/runs. */
export async function detectMode(): Promise<Mode> {
  try {
    const resp = await fetch("/api/runs");
    return resp.ok ? "registry" : "single";
  } catch {
    return "single";
  }
}

/** Prefix for data endpoints: /api (single-run) or /api/runs/{run_id}. */
export function apiBase(runId?: string): string {
  return runId ? `/api/runs/${encodeURIComponent(runId)}` : "/api";
}

/** Live-row websocket path: /ws (single-run) or the per-run mount. */
export function wsPath(runId?: string): string {
  return runId ? `/api/runs/${encodeURIComponent(runId)}/ws` : "/ws";
}

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let message = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body.error) message = body.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return resp.json() as Promise<T>;
}

export function getJSON<T>(path: string, params?: Record<string, string>): Promise<T> {
  const qs = params && Object.keys(params).length > 0 ? `?${new URLSearchParams(params)}` : "";
  return fetch(path + qs).then((r) => handle<T>(r));
}

export function postJSON<T>(path: string, body: unknown): Promise<T> {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => handle<T>(r));
}
