import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { EvalScorePoint } from '../api/types';
import { SERIES_COLORS } from '../theme/colors';

export function EvalTabPanel({ runId }: { runId: string }) {
  const [evalScores, setEvalScores] = useState<EvalScorePoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getEvalScores(runId)
      .then(setEvalScores)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading eval scores...</div>;
  if (evalScores.length === 0) return <div className="empty-state">No eval runs linked to this training run's checkpoints</div>;

  // Collect all benchmark names
  const benchmarks = [...new Set(evalScores.flatMap((e) => Object.keys(e.scores)))].sort();

  return (
    <div>
      {/* Score progression — all benchmarks on one chart */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>
          Eval Score Progression ({evalScores.length} checkpoints, {benchmarks.length} benchmarks)
        </div>
        {(() => {
          const COLORS = SERIES_COLORS;
          // Build chart data: one row per checkpoint with all benchmark scores
          const chartData = evalScores
            .map((e) => ({
              name: e.checkpoint_name ?? `step ${e.step ?? 0}`,
              ...Object.fromEntries(benchmarks.map((b) => [b, e.scores[b] != null ? Math.round(e.scores[b] * 1000) / 10 : null])),
            }));
          return (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} width={40} unit="%" />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '0.75rem' }}
                  formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`]}
                />
                {benchmarks.map((bench, i) => (
                  <Line key={bench} type="monotone" dataKey={bench} stroke={COLORS[i % COLORS.length]}
                    strokeWidth={2} dot={{ r: 4 }} name={bench} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          );
        })()}
      </div>

      {/* Eval runs table */}
      <div className="card" style={{ overflow: 'auto' }}>
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Eval Runs</div>
        <table>
          <thead>
            <tr>
              <th>Checkpoint</th>
              <th>Step</th>
              {benchmarks.map((b) => <th key={b}>{b}</th>)}
              <th></th>
            </tr>
          </thead>
          <tbody>
            {evalScores.map((e, i) => {
              const prev = i > 0 ? evalScores[i - 1] : undefined;
              return (
                <tr key={e.eval_run_id}>
                  <td className="mono" style={{ fontWeight: 600 }}>{e.checkpoint_name ?? e.eval_run_id}</td>
                  <td className="mono">{e.step ?? '-'}</td>
                  {benchmarks.map((b) => {
                    const score = e.scores[b];
                    const prevScore = prev?.scores[b];
                    const delta = score != null && prevScore != null ? score - prevScore : undefined;
                    return (
                      <td key={b} className="mono">
                        {score != null ? (
                          <>
                            <span style={{ color: score > 0.7 ? 'var(--success)' : score > 0.4 ? 'var(--warning)' : 'var(--error)' }}>
                              {(score * 100).toFixed(1)}%
                            </span>
                            {delta != null && delta !== 0 && (
                              <span style={{ fontSize: '0.5625rem', marginLeft: '0.25rem', color: delta > 0 ? 'var(--success)' : 'var(--error)' }}>
                                {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}
                              </span>
                            )}
                          </>
                        ) : '-'}
                      </td>
                    );
                  })}
                  <td>
                    <Link to={`/eval/${e.eval_run_id}`} style={{ fontSize: '0.75rem', fontWeight: 600 }}>
                      Details →
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
