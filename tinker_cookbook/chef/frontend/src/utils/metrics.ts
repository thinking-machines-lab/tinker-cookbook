/** Shared metrics utilities — EMA smoothing and metric grouping. */

import type { MetricRecord } from '../api/types';

/** Compute EMA-smoothed data with debias correction (same as W&B / Adam). */
export function applyEMA(data: MetricRecord[], keys: string[], alpha: number): MetricRecord[] {
  if (data.length === 0) return data;
  const result: MetricRecord[] = [];
  const ema: Record<string, number> = {};
  const count: Record<string, number> = {};

  for (const record of data) {
    const smoothed: MetricRecord = { step: record.step };
    for (const key of keys) {
      const raw = record[key];
      if (typeof raw !== 'number') continue;
      count[key] = (count[key] ?? 0) + 1;
      if (ema[key] === undefined) {
        ema[key] = (1 - alpha) * raw;
      } else {
        ema[key] = alpha * ema[key] + (1 - alpha) * raw;
      }
      smoothed[key] = ema[key] / (1 - Math.pow(alpha, count[key]));
    }
    result.push(smoothed);
  }
  return result;
}

const METRIC_GROUPS: [string, (key: string) => boolean][] = [
  ['Reward & Correctness', (k) => k.includes('reward') || k.includes('correct') || k.includes('format') || k.includes('by_group')],
  ['KL & Entropy', (k) => k.includes('kl') || k.includes('entropy')],
  ['Learning Rate', (k) => k.includes('/lr') || k === 'optim/lr'],
  ['Per-Turn Rates', (k) => k.includes('per_turn') || k.includes('per_episode')],
  ['Totals', (k) => k.includes('total_') && !k.startsWith('time/')],
  ['Progress', (k) => k.startsWith('progress/')],
  ['Timing', (k) => k.startsWith('time/')],
];

/** Group metric keys into semantic categories. */
export function groupMetricKeys(keys: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const key of keys) {
    if (key.endsWith(':total') || key.endsWith(':count')) continue;
    let assigned = false;
    for (const [groupName, matcher] of METRIC_GROUPS) {
      if (matcher(key)) {
        if (!groups.has(groupName)) groups.set(groupName, []);
        groups.get(groupName)!.push(key);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      const slashIdx = key.indexOf('/');
      const prefix = slashIdx > 0 ? key.substring(0, slashIdx) : 'Other';
      if (!groups.has(prefix)) groups.set(prefix, []);
      groups.get(prefix)!.push(key);
    }
  }
  return groups;
}
