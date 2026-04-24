/** Shared timing span types and color utilities. */

import { getSpanColorByList } from '../theme/colors';

export interface FlatSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
}

export function getSpanColor(name: string, names: string[]): string {
  return getSpanColorByList(name, names);
}
