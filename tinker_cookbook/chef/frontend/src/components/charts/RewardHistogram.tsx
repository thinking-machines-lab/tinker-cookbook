/** Reward distribution histogram using Recharts BarChart. */

import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { TOOLTIP_CONTENT_STYLE } from './ChartTooltip';

interface Props {
  rewards: number[];
  bins?: number;
  /** Called when a histogram bin is clicked, with the bin range. */
  onBinClick?: (min: number, max: number) => void;
  height?: number;
}

interface BinData {
  label: string;
  count: number;
  min: number;
  max: number;
}

function computeBins(rewards: number[], numBins: number): BinData[] {
  if (rewards.length === 0) return [];
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);
  // Handle edge case where all rewards are the same
  if (min === max) {
    return [{ label: min.toFixed(2), count: rewards.length, min, max: max + 0.01 }];
  }
  const binWidth = (max - min) / numBins;
  const bins: BinData[] = [];
  for (let i = 0; i < numBins; i++) {
    const lo = min + i * binWidth;
    const hi = i === numBins - 1 ? max + 0.001 : min + (i + 1) * binWidth;
    bins.push({
      label: lo.toFixed(2),
      count: rewards.filter((r) => r >= lo && r < hi).length,
      min: lo,
      max: hi,
    });
  }
  return bins;
}

/** Color bins by reward range: low=red, mid=yellow, high=green. */
function binColor(bin: BinData): string {
  const mid = (bin.min + bin.max) / 2;
  if (mid >= 0.8) return 'var(--reward-high)';
  if (mid >= 0.3) return 'var(--reward-mid)';
  return 'var(--reward-low)';
}

export function RewardHistogram({ rewards, bins = 15, onBinClick, height = 100 }: Props) {
  const data = useMemo(() => computeBins(rewards, bins), [rewards, bins]);

  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        margin={{ left: 0, right: 10, top: 5, bottom: 0 }}
        onClick={(e) => {
          if (onBinClick && e?.activePayload?.[0]?.payload) {
            const bin = e.activePayload[0].payload as BinData;
            onBinClick(bin.min, bin.max);
          }
        }}
        style={onBinClick ? { cursor: 'pointer' } : undefined}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis dataKey="label" stroke="var(--text-muted)" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
        <YAxis stroke="var(--text-muted)" tick={{ fontSize: 9 }} width={30} allowDecimals={false} />
        <Tooltip
          contentStyle={TOOLTIP_CONTENT_STYLE}
          formatter={(value: number) => [`${value} trajectories`, 'Count']}
          labelFormatter={(label: string) => `Reward: ${label}`}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((bin, i) => (
            <Cell key={i} fill={binColor(bin)} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
