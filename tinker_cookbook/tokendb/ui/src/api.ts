// HTTP + websocket client for the token DB viewer server.

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
  const qs = params ? `?${new URLSearchParams(params)}` : "";
  return fetch(path + qs).then((r) => handle<T>(r));
}

export function postJSON<T>(path: string, body: unknown): Promise<T> {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => handle<T>(r));
}

export interface WsHandlers {
  onRow: (row: StepRow) => void;
  onLabelsChanged?: () => void;
  onStatus?: (status: string) => void;
}

/** Subscribe to live rows; returns a close function. Reconnects on drop. */
export function subscribe(filters: Record<string, unknown>, handlers: WsHandlers): () => void {
  let ws: WebSocket | null = null;
  let closed = false;

  const connect = () => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
      handlers.onStatus?.("live");
      ws?.send(JSON.stringify({ type: "subscribe", filters }));
    };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "row") handlers.onRow(msg.row);
      else if (msg.type === "labels_changed") handlers.onLabelsChanged?.();
    };
    ws.onclose = () => {
      handlers.onStatus?.("disconnected");
      if (!closed) setTimeout(connect, 2000);
    };
  };
  connect();

  return () => {
    closed = true;
    ws?.close();
  };
}
