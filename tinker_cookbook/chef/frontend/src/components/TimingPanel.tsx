import { useEffect, useMemo, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Flamegraph } from './Flamegraph';
import { getSpanColor } from './Waterfall';
import type { FlatSpan } from './Waterfall';

interface Props {
  runId: string;
  jumpToStep?: number | null;
}

export function TimingPanel({ runId, jumpToStep }: Props) {
  const [spans, setSpans] = useState<FlatSpan[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [hiddenSpanNames, setHiddenSpanNames] = useState<Set<string>>(new Set());
  const [minDuration, setMinDuration] = useState<string>('');
  const [treeData, setTreeData] = useState<{ root: any; total_duration: number } | null>(null);

  // Jump to a specific step when linked from metrics chart
  useEffect(() => {
    if (jumpToStep != null) setSelectedStep(jumpToStep);
  }, [jumpToStep]);

  useEffect(() => {
    fetch(`/api/runs/${runId}/timing/flat`)
      .then((r) => r.json())
      .then((data) => setSpans(data.spans ?? []))
      .catch(() => setSpans([]))
      .finally(() => setLoading(false));
  }, [runId]);

  // Filter spans by visibility and min duration
  const filteredSpans = useMemo(() => {
    const minDur = minDuration !== '' ? Number(minDuration) : 0;
    return spans.filter((s) => !hiddenSpanNames.has(s.name) && s.duration >= minDur);
  }, [spans, hiddenSpanNames, minDuration]);

  const steps = useMemo(() => [...new Set(filteredSpans.map((s) => s.step))].sort((a, b) => a - b), [filteredSpans]);
  const allSpanNames = useMemo(() => [...new Set(spans.map((s) => s.name))].sort(), [spans]);
  const spanNames = useMemo(() => [...new Set(filteredSpans.map((s) => s.name))].sort(), [filteredSpans]);

  // Per-step total duration for the overview chart
  const stepDurations = useMemo(() => {
    const map = new Map<number, Record<string, number>>();
    for (const s of filteredSpans) {
      if (!map.has(s.step)) map.set(s.step, { step: s.step });
      const row = map.get(s.step)!;
      row[s.name] = (row[s.name] || 0) + s.duration;
    }
    return Array.from(map.values()).sort((a, b) => (a.step as number) - (b.step as number));
  }, [spans]);

  // Per-step wall-clock duration (from first span start to last span end)
  const wallDurations = useMemo(() => {
    const map = new Map<number, { step: number; wall: number; sum: number }>();
    for (const s of filteredSpans) {
      const existing = map.get(s.step);
      if (!existing) {
        map.set(s.step, { step: s.step, wall: s.wall_end - s.wall_start, sum: s.duration });
      } else {
        existing.wall = Math.max(existing.wall, s.wall_end) - Math.min(0, s.wall_start);
        existing.sum += s.duration;
      }
    }
    // Recalculate wall properly
    for (const step of map.keys()) {
      const stepSpans = spans.filter((s) => s.step === step);
      const minW = Math.min(...stepSpans.map((s) => s.wall_start));
      const maxW = Math.max(...stepSpans.map((s) => s.wall_end));
      map.get(step)!.wall = maxW - minW;
    }
    return Array.from(map.values()).sort((a, b) => a.step - b.step);
  }, [spans]);

  const displayStep = selectedStep ?? steps[0] ?? 0;

  // Fetch tree data for selected step
  useEffect(() => {
    if (steps.length === 0) return;
    fetch(`/api/runs/${runId}/timing/tree/${displayStep}`)
      .then((r) => r.json())
      .then((data) => setTreeData(data))
      .catch(() => setTreeData(null));
  }, [runId, displayStep, steps.length]);
  const stepSpans = useMemo(
    () => filteredSpans.filter((s) => s.step === displayStep).sort((a, b) => a.wall_start - b.wall_start),
    [filteredSpans, displayStep]
  );

  if (loading) return <div className="loading">Loading timing data...</div>;
  if (spans.length === 0) return <div className="empty-state">No timing data available</div>;

  return (
    <div>
      {/* Per-step wall time trend */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span className="card-title">Wall Time per Step</span>
          <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            Click a bar to inspect that step's waterfall
          </span>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={wallDurations}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} unit="s" width={45} />
            <Tooltip
              contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.75rem' }}
              formatter={(value: unknown) => [`${Number(value).toFixed(3)}s`]}
            />
            <Bar dataKey="wall" fill="var(--accent)" radius={[2, 2, 0, 0]} cursor="pointer"
              onClick={(_data: unknown, index: number) => setSelectedStep(wallDurations[index]?.step ?? 0)}
            >
              {wallDurations.map((entry) => (
                <Cell
                  key={entry.step}
                  fill={entry.step === displayStep ? 'var(--accent)' : 'var(--border-bright)'}
                  opacity={entry.step === displayStep ? 1 : 0.6}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stacked duration breakdown over steps */}
      {stepDurations.length > 1 && (
        <div className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="card-header">
            <span className="card-title">Duration Breakdown by Span</span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={stepDurations}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} unit="s" width={45} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.7rem' }}
                formatter={(value: unknown, name: unknown) => [`${Number(value).toFixed(3)}s`, String(name)]}
              />
              {spanNames.map((name) => (
                <Bar key={name} dataKey={name} stackId="a" fill={getSpanColor(name, spanNames)} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          {/* Legend */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem 0.75rem', marginTop: '0.375rem' }}>
            {spanNames.map((name) => (
              <span key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.625rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: getSpanColor(name, spanNames), flexShrink: 0 }} />
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Step selector */}
      <div className="filters-bar">
        <div className="filter-group" style={{ flex: '1 1 auto' }}>
          <span className="filter-label">Step</span>
          <button className="theme-toggle" style={{ padding: '0.1875rem 0.375rem' }}
            onClick={() => { const i = steps.indexOf(displayStep); if (i > 0) setSelectedStep(steps[i - 1]); }}
            disabled={displayStep === steps[0]}
          >Prev</button>
          <input type="range" min={0} max={steps.length - 1}
            value={steps.indexOf(displayStep)}
            onChange={(e) => setSelectedStep(steps[Number(e.target.value)])}
            style={{ flex: 1, minWidth: '80px', accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <button className="theme-toggle" style={{ padding: '0.1875rem 0.375rem' }}
            onClick={() => { const i = steps.indexOf(displayStep); if (i < steps.length - 1) setSelectedStep(steps[i + 1]); }}
            disabled={displayStep === steps[steps.length - 1]}
          >Next</button>
          <span className="mono" style={{ fontSize: '0.75rem', minWidth: '40px' }}>{displayStep}</span>
        </div>
        <div className="filter-group">
          <span className="filter-label">Min dur</span>
          <input
            type="number"
            placeholder="0"
            value={minDuration}
            onChange={(e) => setMinDuration(e.target.value)}
            step="0.01"
            style={{ width: '55px' }}
          />
          <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)' }}>s</span>
        </div>
        <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {stepSpans.length} spans
          {hiddenSpanNames.size > 0 && ` (${hiddenSpanNames.size} hidden)`}
          {stepSpans.length > 0 && (() => {
            const minW = Math.min(...stepSpans.map((s) => s.wall_start));
            const maxW = Math.max(...stepSpans.map((s) => s.wall_end));
            return ` · ${(maxW - minW).toFixed(2)}s wall`;
          })()}
        </div>
      </div>

      {/* Flamegraph — combined call hierarchy + waterfall */}
      {treeData?.root && (
        <div className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="card-header">
            <span className="card-title">Step {displayStep}</span>
            <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
              {treeData.total_duration.toFixed(2)}s total · click a span for details
            </span>
          </div>
          <Flamegraph root={treeData.root} totalDuration={treeData.total_duration} runId={runId} step={displayStep} />
        </div>
      )}

      {/* Summary table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Aggregate Statistics</span>
          <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            Click a span name to show/hide it
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Span</th>
              <th>Count</th>
              <th>Total</th>
              <th>Mean</th>
              <th>Max</th>
              <th>% of Total</th>
            </tr>
          </thead>
          <tbody>
            {(() => {
              const stats = new Map<string, { total: number; count: number; max: number }>();
              let grandTotal = 0;
              for (const s of filteredSpans) {
                if (!stats.has(s.name)) stats.set(s.name, { total: 0, count: 0, max: 0 });
                const st = stats.get(s.name)!;
                st.total += s.duration;
                st.count += 1;
                st.max = Math.max(st.max, s.duration);
                grandTotal += s.duration;
              }
              return [...stats.entries()]
                .sort((a, b) => b[1].total - a[1].total)
                .map(([name, s]) => (
                  <tr key={name} style={{ cursor: 'pointer' }}
                    onClick={() => setHiddenSpanNames((prev) => {
                      const next = new Set(prev);
                      if (next.has(name)) next.delete(name);
                      else next.add(name);
                      return next;
                    })}
                  >
                    <td style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: hiddenSpanNames.has(name) ? 0.4 : 1 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: getSpanColor(name, allSpanNames), flexShrink: 0 }} />
                      <span className="mono" style={{ textDecoration: hiddenSpanNames.has(name) ? 'line-through' : 'none' }}>{name}</span>
                    </td>
                    <td className="mono">{s.count}</td>
                    <td className="mono">{s.total.toFixed(3)}s</td>
                    <td className="mono">{(s.total / s.count).toFixed(3)}s</td>
                    <td className="mono">{s.max.toFixed(3)}s</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                        <div style={{
                          width: `${(s.total / grandTotal) * 100}%`,
                          maxWidth: '80px',
                          height: 6,
                          borderRadius: 3,
                          background: getSpanColor(name, allSpanNames),
                          minWidth: 2,
                        }} />
                        <span className="mono text-muted" style={{ fontSize: '0.6875rem' }}>
                          {((s.total / grandTotal) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ));
            })()}
          </tbody>
        </table>
      </div>
    </div>
  );
}
