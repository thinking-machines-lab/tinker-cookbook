/** Shared timing span types and color utilities. */

export interface FlatSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
}

const SPAN_PALETTE = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

export function getSpanColor(name: string, names: string[]): string {
  const idx = names.indexOf(name);
  return SPAN_PALETTE[idx % SPAN_PALETTE.length];
}
