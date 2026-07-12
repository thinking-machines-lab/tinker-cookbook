// Typed HTTP client for the token DB viewer server.
//
// The server runs in one of two modes:
//   - single-run mode: endpoints at /api/* and the live socket at /ws
//   - registry mode: /api/runs lists runs, /api/dashboard aggregates them,
//     and every single-run endpoint is also mounted at /api/runs/{run_id}/*
// `detectMode()` probes once at app start; `apiBase(runId)` / `chatWsPath(runId)`
// pick the right prefix for data calls.

/** One structured tool call of a turn (schema v2 `tool_calls` entry). */
export interface ToolCallEntry {
  name: string;
  args_json: string;
  error_type: string | null;
  should_stop: boolean;
}

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
  // Typed maps (schema v2). NaN metric values arrive as null (JSON has no NaN).
  metrics: Record<string, number | null>;
  attrs: Record<string, string>;
  token_metrics: Record<string, (number | null)[]>;
  tool_calls: ToolCallEntry[] | null;
  logs: string;
  extra: string;
  filtered_reason: string | null;
  superseded: boolean;
  // Added by the detail endpoint:
  ob_full_tokens?: number[];
  ac_token_strs?: string[];
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
  context?: { model_name?: string; recipe_name?: string };
}

/** One /api/runs record (registry mode): registry entry + liveness probe. */
export interface RegistryRun {
  run_id: string;
  run_attempt: number;
  model_name: string | null;
  recipe_name: string | null;
  status?: { live: boolean };
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

export interface AgentConfig {
  provider: string;
  model: string;
  has_key: boolean;
  /**
   * Model suggestions per provider (the dropdown options). The tinker list
   * comes from server capabilities and is only populated when the config is
   * fetched with the tinker provider in play (?provider=tinker).
   */
  models: Record<string, string[]>;
  /** The server-side default model per provider. */
  default_model: Record<string, string>;
  /** Why the tinker model list is empty (missing TINKER_API_KEY, fetch failure). */
  tinker_models_error?: string | null;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  n_records: number;
  mtime: number | null;
}

/** One JSONL transcript line: a conversation message or a replayable UI event. */
export interface ChatRecord {
  kind: string; // "message" | "event"
  ts?: string;
  // kind == "message":
  role?: string;
  content?: string;
  tool_calls?: { id: string; name: string; arguments: Record<string, unknown> }[];
  tool_call_id?: string;
  // kind == "event" (e.g. visual_published):
  type?: string;
  name?: string;
  url?: string;
  title?: string;
  description?: string;
}

export interface VisualInfo {
  name: string;
  url: string;
  size: number | null;
  mtime: number | null;
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

/** Chat websocket path: per-run mount, or /api/chat (single-run and registry-global). */
export function chatWsPath(runId?: string): string {
  return runId ? `/api/runs/${encodeURIComponent(runId)}/chat` : "/api/chat";
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
