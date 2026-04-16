/**
 * TypeScript types for the Tinker Chef API.
 *
 * These types mirror the JSON structures returned by the FastAPI backend.
 * The backend reads training artifacts from disk (JSONL, JSON files written
 * by the RL/SL/DPO training loops) and serves them via REST endpoints.
 *
 * ## Data Model Hierarchy
 *
 * ```
 * Storage (local or cloud)
 * └── RunRegistry discovers training runs
 *     ├── TrainingRun (one per log directory)
 *     │   ├── config.json           → RunInfo.config
 *     │   ├── metrics.jsonl         → MetricRecord[]
 *     │   ├── checkpoints.jsonl     → CheckpointRecord[]
 *     │   ├── timing_spans.jsonl    → TimingSpan[]
 *     │   ├── chat_sessions/        → ChatSession[]
 *     │   └── iteration_NNNNNN/
 *     │       ├── train_rollout_summaries.jsonl  → RolloutDetail[]
 *     │       └── train_logtree.json             → LogtreeResponse
 *     └── EvalStore (optional)
 *         ├── runs.jsonl            → EvalRunSummary[]
 *         └── runs/{id}/
 *             ├── metadata.json     → EvalRunDetail
 *             └── {benchmark}/
 *                 ├── result.json   → EvalBenchmarkResult
 *                 └── trajectories.jsonl → EvalTrajectoryDetail[]
 * ```
 *
 * ## Rollout Schema Versions
 *
 * - **v1-v2**: Basic rollout summaries with steps (ob_len, ac_len, reward).
 *   Conversations only available via logtree extraction (first few groups).
 * - **v3** (current): Adds `conversation` (aggregated messages) and
 *   `model_name`. Per-step `logs._conversation` contains messages for
 *   each step, enabling interleaved step+conversation views.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Conversation types (shared across training rollouts, eval, and chat)
// ═══════════════════════════════════════════════════════════════════════════

/** Known content part types. Extensible — unknown types render as collapsible JSON. */
export type ContentPartType = 'text' | 'thinking' | 'image';

/**
 * A single content part within a message.
 *
 * Produced by `content_to_jsonable()` in the renderer. Each part has a `type`
 * discriminator and a corresponding data field:
 * - `{type: 'text', text: '...'}` — plain text
 * - `{type: 'thinking', thinking: '...'}` — model's chain-of-thought (collapsible in UI)
 * - `{type: 'image', image: 'https://...' | 'data:...'}` — image URL or data URI
 */
export interface ContentPart {
  type: ContentPartType | string;
  text?: string;
  thinking?: string;
  image?: string;
}

/**
 * A parsed tool invocation attached to an assistant message.
 *
 * Matches the OpenAI-style tool_call format used by Tinker renderers.
 */
export interface ToolCallInfo {
  type: 'function';
  id?: string;
  function: { name: string; arguments: string };
}

/** Known message roles. */
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool' | 'environment' | 'grader';

/**
 * A conversation message with structured content.
 *
 * Produced by `message_to_jsonable()` in `tinker_cookbook/renderers/base.py`.
 * Used consistently across training rollouts, eval trajectories, and chat sessions.
 *
 * Content can be a plain string (legacy/simple) or an array of ContentPart
 * objects (structured, supports thinking blocks, images, etc.).
 */
export interface ConversationMessage {
  /** Message author: 'user', 'assistant', 'system', 'tool', 'environment', 'grader'. */
  role: MessageRole | string;
  /** Plain text (legacy) or structured content parts. */
  content: string | ContentPart[];
  /** Tool calls made by the assistant (OpenAI-style). */
  tool_calls?: ToolCallInfo[];
  /** Tool calls that failed to parse from model output. */
  unparsed_tool_calls?: Array<{ raw_text: string; error: string }>;
  /** For role='tool': the tool_call.id this result corresponds to. */
  tool_call_id?: string;
  /** For role='tool': the tool function name. */
  name?: string;
  /** Number of tokens in this message (set by renderer). */
  token_count?: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// Training run types
// ═══════════════════════════════════════════════════════════════════════════

/** Run lifecycle status, inferred from file modification times. */
export type RunStatus = 'running' | 'completed' | 'idle';

/** Training paradigm, inferred from config keys. */
export type TrainingType = 'rl' | 'sl' | 'dpo';

/**
 * Summary of a discovered training run.
 *
 * Returned by `GET /api/runs`. Each run corresponds to a directory containing
 * `config.json` and/or `metrics.jsonl`.
 */
export interface RunInfo {
  /** Directory name used as the run identifier. */
  run_id: string;
  /** Storage path prefix (e.g., local path or cloud URI). */
  prefix: string;
  /** Whether config.json exists. */
  has_config: boolean;
  /** Whether metrics.jsonl exists. */
  has_metrics: boolean;
  /** Whether checkpoints.jsonl exists. */
  has_checkpoints: boolean;
  /** Whether timing_spans.jsonl exists. */
  has_timing: boolean;
  /** Number of iteration_NNNNNN directories found. */
  iteration_count: number;
  /** 'running' if files updated recently, 'completed' if final checkpoint exists, else 'idle'. */
  status: RunStatus;
  /** Unix timestamp of most recent file modification, or null. */
  last_updated: number | null;
  /** 'rl' if config has loss_fn, 'dpo' if has dpo_beta, 'sl' otherwise. Null if no config. */
  training_type: TrainingType | null;
  /** Subset of config keys (model_name, learning_rate, etc.) for dashboard display. */
  config_summary?: Record<string, unknown>;
  /** Step number from the most recent metrics record. */
  latest_step?: number;
  /** Total number of metrics records. */
  total_steps?: number;
  /** Full config (only on detail endpoint). */
  config?: Record<string, unknown>;
  /** Best eval scores across this run's checkpoints, keyed by benchmark name. */
  eval_scores?: Record<string, number>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Metrics types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Response from `GET /api/runs/{run_id}/metrics`.
 */
export interface MetricsResponse {
  run_id: string;
  /** Total records in the file (may exceed records.length if `limit` is set). */
  total_records: number;
  records: MetricRecord[];
}

/**
 * A single metrics record from metrics.jsonl.
 *
 * Each line in metrics.jsonl is one record, keyed by metric name with
 * numeric values. The 'step' field is the training iteration index.
 *
 * Common keys: `env/all/reward/total`, `optim/kl_sample_train_v1`,
 * `time/total`, `env/{tag}/correct`, etc.
 */
export interface MetricRecord {
  step?: number;
  [key: string]: number | string | undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// Iteration and rollout types
// ═══════════════════════════════════════════════════════════════════════════

/** Metadata about a single iteration directory (iteration_NNNNNN/). */
export interface IterationInfo {
  iteration: number;
  has_train_rollouts: boolean;
  has_train_logtree: boolean;
  /** Eval labels found (e.g., ['gsm8k', 'math'] from eval_gsm8k_rollout_summaries.jsonl). */
  eval_labels: string[];
}

/** Rollout trajectory status. */
export type RolloutStatus = 'ok' | 'error' | 'timeout';

/** Why the model stopped generating. */
export type StopReason = 'stop' | 'length';

/**
 * Summary of a single trajectory, returned in rollout listing.
 *
 * This is a lightweight projection of RolloutDetail — no conversation
 * data or per-step details. Used for the rollout browser table.
 */
export interface RolloutSummary {
  /** Index of the GRPO group (trajectories in the same group share a problem). */
  group_idx: number;
  /** Index within the group. */
  traj_idx: number;
  /** Logging tags from EnvGroupBuilder.logging_tags() (e.g., ['math', 'gsm8k']). */
  tags: string[];
  /** Sum of per-step rewards + final group reward. */
  total_reward: number;
  /** Group-level reward from EnvGroupBuilder.compute_group_rewards(). */
  final_reward: number;
  /** Number of env.step() calls in this trajectory. */
  num_steps: number;
  /** Total action tokens generated across all steps. */
  total_tokens: number;
  /** Token length of the final observation (approximate context window usage). */
  final_ob_len: number;
  /** Training step of the sampling client that generated this trajectory. */
  sampling_client_step: number | null;
  /** 'ok', 'error' (rollout failed), or 'timeout' (hit max_tokens). */
  status: RolloutStatus | null;
  /** Exception class name if status='error'. */
  error_type: string | null;
  /** 'stop' (hit stop sequence) or 'length' (hit max_tokens). */
  stop_reason: StopReason | null;
}

/** Response from `GET /api/runs/{run_id}/iterations/{iter}/rollouts`. */
export interface RolloutsResponse {
  run_id: string;
  iteration: number;
  split: string;
  /** Total matching rollouts (before pagination). */
  total: number;
  /** All tags across rollouts in this iteration (for filter dropdown). */
  available_tags: string[];
  rollouts: RolloutSummary[];
}

/**
 * A single step within a trajectory.
 *
 * Maps to one Transition in the RL training loop: the model saw `ob_len`
 * tokens of context, generated `ac_len` tokens, and received `reward`.
 *
 * For multi-turn agent rollouts, each step corresponds to one conversation
 * turn (prompt → model response → environment feedback).
 */
export interface RolloutStep {
  step_idx: number;
  /** Observation length in tokens (the prompt the model saw). */
  ob_len: number;
  /** Action length in tokens (the model's response). */
  ac_len: number;
  /** Immediate reward from env.step(). */
  reward: number;
  /** Whether this step ended the episode. */
  episode_done: boolean;
  /** Numeric metrics from env.step() (e.g., {correct: 1.0, format: 1.0}). */
  metrics: Record<string, number>;
  /**
   * Diagnostic logs from env.step().
   *
   * User-facing keys are strings/numbers. Framework keys are prefixed with '_':
   * - `_conversation`: ConversationMessage[] for this step (schema v3).
   *   Contains the decoded messages exchanged during this step.
   */
  logs: Record<string, unknown>;
}

/**
 * Full detail for a single trajectory.
 *
 * Stored as one line in `{iteration}/train_rollout_summaries.jsonl`.
 * Returned by `GET /api/runs/{run_id}/iterations/{iter}/rollouts/{group}/{traj}`.
 */
export interface RolloutDetail {
  /** Schema version: 1-2 (legacy), 3 (current, has conversation + model_name). */
  schema_version: number;
  /** 'train' or 'eval/{label}'. */
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  tags: string[];
  sampling_client_step: number | null;
  /** Model that generated this trajectory (schema v3+). Useful for distillation. */
  model_name?: string | null;
  /** Sum of per-step rewards + final group reward. */
  total_reward: number;
  /** Group-level reward from compute_group_rewards(). */
  final_reward: number;
  /** Metrics from compute_group_rewards() (e.g., pairwise comparison scores). */
  trajectory_metrics: Record<string, number>;
  /**
   * Aggregated conversation messages across all steps (schema v3+).
   *
   * Flat list of all messages in order: initial prompt, model response,
   * tool results, next prompt, etc. Built from per-step `logs._conversation`.
   * Null for schema v1-v2 data (use logtree extraction as fallback).
   */
  conversation?: ConversationMessage[] | null;
  steps: RolloutStep[];
  /** Final observation length — approximates total context window usage. */
  final_ob_len: number;
  status: RolloutStatus | null;
  error_type: string | null;
  error_message: string | null;
  stop_reason: StopReason | null;
}

// ═══════════════════════════════════════════════════════════════════════════
// Timing types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A flat timing span from the /timing/flat endpoint.
 *
 * Each span represents one measured operation during training (e.g.,
 * 'policy_sample', 'env_step', 'train_step'). Spans can nest —
 * parent/child relationships are inferred from wall_start/wall_end containment.
 */
export interface TimingSpan {
  step: number;
  /** Span name (e.g., 'policy_sample', 'env_step', 'train_step'). */
  name: string;
  /** Duration in seconds (perf_counter based, not wall-clock). */
  duration: number;
  /** Wall-clock start time (time.time()). */
  wall_start: number;
  /** Wall-clock end time. */
  wall_end: number;
  /** Optional attributes (e.g., {group_idx: 0} for per-group spans). */
  attributes?: Record<string, unknown>;
}

/** A timing record from timing_spans.jsonl (one per training step). */
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

// ═══════════════════════════════════════════════════════════════════════════
// Checkpoint types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A checkpoint record from checkpoints.jsonl.
 *
 * Written by `TrainingRunStore.write_checkpoint()` each time training
 * saves model weights. The `sampler_path` is needed for interactive chat.
 */
export interface CheckpointRecord {
  /** Human-readable name (e.g., '000050', 'final'). */
  name: string;
  /** Training batch/step when this checkpoint was saved. */
  batch?: number;
  epoch?: number;
  /** Whether this is the final checkpoint (kept indefinitely, no TTL). */
  final?: boolean;
  /** 'both' (state + sampler), 'state', or 'sampler'. */
  kind?: string;
  /** Tinker state path for resuming training. */
  state_path?: string;
  /** Tinker sampler path for inference/chat. */
  sampler_path?: string;
  /** Tinker sampler weights path. */
  sampler_weights_path?: string;
  /** Unix timestamp when saved. */
  timestamp?: number;
  /** Training loop state at save time. */
  loop_state?: { epoch?: number; batch?: number };
  extra?: Record<string, unknown>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Logtree types (legacy conversation source for schema v1-v2)
// ═══════════════════════════════════════════════════════════════════════════

/** A node in the logtree visualization tree. */
export interface LogtreeNode {
  tag: string;
  attrs?: Record<string, string>;
  children?: (string | LogtreeNode)[];
  /** Data payload — conversation nodes have {type: 'conversation', messages: [...]}. */
  data?: Record<string, unknown>;
}

export interface LogtreeResponse {
  title: string;
  started_at: string;
  root: LogtreeNode;
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Eval score progression point — matches eval runs to training checkpoints.
 *
 * Returned by `GET /api/runs/{run_id}/eval-scores`. Used to plot eval
 * score curves alongside training metrics.
 */
export interface EvalScorePoint {
  eval_run_id: string;
  checkpoint_name: string | null;
  checkpoint_path: string | null;
  /** Training step of the checkpoint, matched by path or name. */
  step: number | null;
  /** Benchmark name → score (0-1). */
  scores: Record<string, number>;
  benchmarks: string[];
  timestamp: string;
}

/** Summary of an eval run (from EvalStore.list_runs()). */
export interface EvalRunSummary {
  eval_run_id: string;
  model_name: string;
  checkpoint_path?: string;
  checkpoint_name?: string;
  timestamp?: string;
  benchmarks: string[];
  scores: Record<string, number>;
}

/** Full detail of an eval run with per-benchmark results. */
export interface EvalRunDetail {
  eval_run_id: string;
  metadata: Record<string, unknown>;
  benchmarks: string[];
  results: Record<string, EvalBenchmarkResult>;
}

/** Aggregated results for one benchmark within an eval run. */
export interface EvalBenchmarkResult {
  name: string;
  /** Overall score (0-1). */
  score: number;
  num_examples: number;
  num_correct: number;
  num_errors: number;
  num_truncated: number;
  /** Additional benchmark-specific metrics. */
  metrics: Record<string, number>;
  /** Total wall-clock time for this benchmark. */
  time_seconds: number;
  /** Pass@k scores (e.g., {1: 0.45, 5: 0.72}). Only for multi-sample benchmarks. */
  pass_at_k?: Record<string, number>;
}

/** Summary of one eval trajectory in a listing. */
export interface EvalTrajectorySummary {
  idx: number;
  example_id?: string;
  /** 1.0 for correct, 0.0 for incorrect. */
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

/** A single turn in an eval trajectory (extends ConversationMessage with metadata). */
export interface EvalTrajectoryTurn extends ConversationMessage {
  token_count: number;
  metadata: Record<string, unknown>;
}

/** Full detail of one eval trajectory. */
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

/** Row in the cross-eval scores table. */
export interface ScoresTableRow {
  run_id: string;
  model_name: string;
  checkpoint_name?: string;
  timestamp?: string;
  scores: Record<string, number>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Data source types
// ═══════════════════════════════════════════════════════════════════════════

/** A data source URI with its type (local path or cloud storage). */
export interface DataSource {
  /** Local path, s3://, or gs:// URI. */
  url: string;
  type: 'local' | 'cloud';
}
