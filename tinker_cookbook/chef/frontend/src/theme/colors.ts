/** Centralized color palettes — single source of truth for all visualizations. */

/** General-purpose series colors for line/bar charts. */
export const SERIES_COLORS = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

/** Colors for comparing runs (up to 4 runs). */
export const RUN_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444'];

/** Named colors for known timing span types. */
export const SPAN_COLORS: Record<string, string> = {
  sampling: '#8bbe3a',
  group_rollout: '#a78bfa',
  policy_sample: '#e5a11c',
  env_step: '#e85850',
  env_observe: '#6aad7a',
  compute_group_rewards: '#ec4899',
  train_step: '#06b6d4',
  prepare_minibatch: '#f97316',
  assemble_training_data: '#64748b',
  save_checkpoint: '#14b8a6',
  train_fwd_bwd_enqueue: '#06b6d4',
  train_fwd_bwd_consume: '#0891b2',
  train_optim_enqueue: '#f97316',
  train_optim_consume: '#ea580c',
  compute_kl_penalty: '#64748b',
  compute_kl_metrics: '#64748b',
  run_evaluations: '#14b8a6',
};

/** Get a color for a span name, with hash-based fallback for unknown names. */
export function getSpanColor(name: string, fallbackIndex?: number): string {
  if (SPAN_COLORS[name]) return SPAN_COLORS[name];
  // Hash-based fallback for unknown span names
  if (fallbackIndex != null) return SERIES_COLORS[fallbackIndex % SERIES_COLORS.length];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return SERIES_COLORS[Math.abs(hash) % SERIES_COLORS.length];
}

/** Get a color for a span by its position in a list of known names. */
export function getSpanColorByList(name: string, names: string[]): string {
  const idx = names.indexOf(name);
  return SERIES_COLORS[idx % SERIES_COLORS.length];
}
