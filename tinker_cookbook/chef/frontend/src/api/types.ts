/** TypeScript types matching the Tinker Chef backend API responses. */

// ── Unified conversation types (shared across training, eval, chat) ───────

/** A single content part within a message — matches renderer's ContentPart. */
export interface ContentPart {
  type: 'text' | 'thinking' | 'image';
  text?: string;
  thinking?: string;
  image?: string;
}

/** A parsed tool invocation — matches renderer's ToolCall. */
export interface ToolCallInfo {
  type: 'function';
  id?: string;
  function: { name: string; arguments: string };
}

/** A conversation message with structured content — matches message_to_jsonable() output.
 *  Content accepts string for backward compat with old logtree data. */
export interface ConversationMessage {
  role: string;
  content: string | ContentPart[];
  tool_calls?: ToolCallInfo[];
  unparsed_tool_calls?: Array<{ raw_text: string; error: string }>;
  tool_call_id?: string;
  name?: string;
  token_count?: number;
}

export interface RunInfo {
  run_id: string;
  prefix: string;
  has_config: boolean;
  has_metrics: boolean;
  has_checkpoints: boolean;
  has_timing: boolean;
  iteration_count: number;
  status: 'running' | 'completed' | 'idle';
  last_updated: number | null;
  training_type: 'rl' | 'sl' | 'dpo' | null;
  config_summary?: Record<string, unknown>;
  latest_step?: number;
  total_steps?: number;
  config?: Record<string, unknown>;
  eval_scores?: Record<string, number>;
}

export interface MetricsResponse {
  run_id: string;
  total_records: number;
  records: MetricRecord[];
}

export interface MetricRecord {
  step?: number;
  [key: string]: number | string | undefined;
}

export interface IterationInfo {
  iteration: number;
  has_train_rollouts: boolean;
  has_train_logtree: boolean;
  eval_labels: string[];
}

export interface EvalScorePoint {
  eval_run_id: string;
  checkpoint_name: string | null;
  checkpoint_path: string | null;
  step: number | null;
  scores: Record<string, number>;
  benchmarks: string[];
  timestamp: string;
}

export interface RolloutSummary {
  group_idx: number;
  traj_idx: number;
  tags: string[];
  total_reward: number;
  final_reward: number;
  num_steps: number;
  total_tokens: number;
  final_ob_len: number;
  sampling_client_step: number | null;
  status: string | null;
  error_type: string | null;
  stop_reason: string | null;
}

export interface RolloutsResponse {
  run_id: string;
  iteration: number;
  split: string;
  total: number;
  available_tags: string[];
  rollouts: RolloutSummary[];
}

export interface RolloutStep {
  step_idx: number;
  ob_len: number;
  ac_len: number;
  reward: number;
  episode_done: boolean;
  metrics: Record<string, number>;
  logs: Record<string, unknown>;
}

export interface RolloutDetail {
  schema_version: number;
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  tags: string[];
  sampling_client_step: number | null;
  model_name?: string | null;
  total_reward: number;
  final_reward: number;
  trajectory_metrics: Record<string, number>;
  /** Aggregated conversation messages across all steps (schema v3+). */
  conversation?: ConversationMessage[] | null;
  steps: RolloutStep[];
  final_ob_len: number;
  status: string | null;
  error_type: string | null;
  error_message: string | null;
  stop_reason: string | null;
}

/** A flat timing span from the /timing/flat endpoint. */
export interface TimingSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
}

/** A nested timing record from the /timing endpoint (one per step). */
export interface TimingStepRecord {
  step: number;
  spans: TimingSpan[];
}

export interface TimingResponse {
  run_id: string;
  total_records: number;
  records: TimingStepRecord[];
}

export interface TimingFlatResponse {
  run_id: string;
  total_spans: number;
  spans: TimingSpan[];
}

export interface CheckpointRecord {
  name: string;
  batch?: number;
  epoch?: number;
  final?: boolean;
  kind?: string;
  state_path?: string;
  sampler_path?: string;
  sampler_weights_path?: string;
  timestamp?: number;
  loop_state?: { epoch?: number; batch?: number };
  extra?: Record<string, unknown>;
}

export interface LogtreeNode {
  tag: string;
  attrs?: Record<string, string>;
  children?: (string | LogtreeNode)[];
  data?: Record<string, unknown>;
}

export interface LogtreeResponse {
  title: string;
  started_at: string;
  root: LogtreeNode;
}

// Eval benchmark types

export interface EvalRunSummary {
  eval_run_id: string;
  model_name: string;
  checkpoint_path?: string;
  checkpoint_name?: string;
  timestamp?: string;
  benchmarks: string[];
  scores: Record<string, number>;
}

export interface EvalRunDetail {
  eval_run_id: string;
  metadata: Record<string, unknown>;
  benchmarks: string[];
  results: Record<string, EvalBenchmarkResult>;
}

export interface EvalBenchmarkResult {
  name: string;
  score: number;
  num_examples: number;
  num_correct: number;
  num_errors: number;
  num_truncated: number;
  metrics: Record<string, number>;
  time_seconds: number;
  pass_at_k?: Record<string, number>;
}

export interface EvalTrajectorySummary {
  idx: number;
  example_id?: string;
  reward: number;
  num_turns: number;
  time_seconds: number;
  error?: string;
  logs: Record<string, unknown>;
}

export interface EvalTrajectoriesResponse {
  eval_run_id: string;
  benchmark: string;
  total: number;
  trajectories: EvalTrajectorySummary[];
}

export interface EvalTrajectoryTurn extends ConversationMessage {
  token_count: number;
  metadata: Record<string, unknown>;
}

export interface EvalTrajectoryDetail {
  idx: number;
  benchmark: string;
  example_id?: string;
  turns: EvalTrajectoryTurn[];
  reward: number;
  metrics: Record<string, number>;
  logs: Record<string, unknown>;
  error?: string;
  time_seconds: number;
}

export interface ScoresTableRow {
  run_id: string;
  model_name: string;
  checkpoint_name?: string;
  timestamp?: string;
  scores: Record<string, number>;
}

export interface DataSource {
  url: string;
  type: 'local' | 'cloud';
}
