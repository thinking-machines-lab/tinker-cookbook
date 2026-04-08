import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { TimingPanel } from '../components/TimingPanel';
import { StatusBadge, TypeBadge } from '../utils/shared';
import { useUrlParam } from '../utils/useUrlParam';
import type { CheckpointRecord, EvalScorePoint, IterationInfo, MetricRecord, RunInfo } from '../api/types';

type Tab = 'metrics' | 'rollouts' | 'timing' | 'eval' | 'checkpoints' | 'config';

/** Pick the latest value and trend direction for a metric key. */
function getStatFromMetrics(records: MetricRecord[], key: string): { value: number | null; trend: 'up' | 'down' | 'flat'; sparkData: number[] } {
  const values = records.map((r) => r[key]).filter((v): v is number => typeof v === 'number');
  if (values.length === 0) return { value: null, trend: 'flat', sparkData: [] };
  const latest = values[values.length - 1];
  const prev = values.length > 1 ? values[values.length - 2] : latest;
  const trend: 'up' | 'down' | 'flat' = latest > prev + 0.001 ? 'up' : latest < prev - 0.001 ? 'down' : 'flat';
  return { value: latest, trend, sparkData: values.slice(-30) };
}

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const h = 28;
  const w = 80;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 4) - 2}`).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function formatValue(v: number | null, isPercent?: boolean): string {
  if (v === null) return '-';
  if (isPercent) return `${(v * 100).toFixed(1)}%`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

const TREND_ARROWS: Record<string, string> = { up: '\u2191', down: '\u2193', flat: '' };
const TREND_COLORS: Record<string, string> = { up: 'var(--accent)', down: 'var(--error)', flat: 'var(--text-muted)' };

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [activeTab, setActiveTab] = useUrlParam('tab', 'metrics');
  const [urlIter, setUrlIter] = useUrlParam('iter', '');
  const [urlStep, setUrlStep] = useUrlParam('step', '');
  const [run, setRun] = useState<RunInfo | null>(null);
  const [iterations, setIterations] = useState<IterationInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [visitedTabs, setVisitedTabs] = useState<Set<Tab>>(new Set([activeTab as Tab]));
  const [jumpToStep, setJumpToStep] = useState<number | null>(urlIter ? Number(urlIter) : null);

  useEffect(() => {
    if (!runId) return;
    Promise.all([
      api.getRun(runId),
      api.listIterations(runId),
      api.getMetrics(runId),
    ])
      .then(([runData, iters, metricsResp]) => {
        setRun(runData);
        setIterations(iters);
        setMetrics(metricsResp.records);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const switchTab = (tab: Tab) => {
    setActiveTab(tab);
    setVisitedTabs((prev) => new Set(prev).add(tab));
  };

  const handleMetricStepClick = (step: number, group: string) => {
    if (group === 'Timing') {
      setUrlStep(String(step));
      switchTab('timing');
    } else {
      setUrlIter(String(step));
      setJumpToStep(step);
      switchTab('rollouts');
    }
  };

  // Auto-detect stat card metric keys from available metrics
  const allKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const r of metrics) for (const k of Object.keys(r)) if (k !== 'step') keys.add(k);
    return keys;
  }, [metrics]);

  const findKey = (patterns: string[]) => {
    for (const p of patterns) {
      for (const k of allKeys) {
        if (p.includes('*') ? k.includes(p.replace(/\*/g, '')) : k === p) return k;
      }
    }
    return null;
  };

  const rewardKey = findKey(['env/all/reward/total', 'reward/total', 'reward']);
  const correctKey = findKey(['env/all/correct', 'correct']);
  const klKey = findKey(['optim/kl_sample_train_v1', 'kl/forward', 'kl']);
  const speedKey = findKey(['time/total']);

  const reward = useMemo(() => rewardKey ? getStatFromMetrics(metrics, rewardKey) : null, [metrics, rewardKey]);
  const correct = useMemo(() => correctKey ? getStatFromMetrics(metrics, correctKey) : null, [metrics, correctKey]);
  const kl = useMemo(() => klKey ? getStatFromMetrics(metrics, klKey) : null, [metrics, klKey]);
  const speed = useMemo(() => speedKey ? getStatFromMetrics(metrics, speedKey) : null, [metrics, speedKey]);

  if (loading) return <div className="loading">Loading run...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!run || !runId) return <div className="error-msg">Run not found</div>;

  const hasIterations = iterations.some((it) => it.has_train_rollouts);
  const hasTiming = run.has_timing;

  const hasCheckpoints = run.has_checkpoints;

  const tabs: { id: Tab; label: string; disabled?: boolean }[] = [
    { id: 'metrics', label: 'Metrics' },
    { id: 'rollouts', label: 'Rollouts', disabled: !hasIterations },
    { id: 'timing', label: 'Timing', disabled: !hasTiming },
    { id: 'eval', label: 'Eval' },
    { id: 'checkpoints', label: 'Checkpoints', disabled: !hasCheckpoints },
    { id: 'config', label: 'Config' },
  ];

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span>/</span>
        <span>{runId}</span>
      </div>

      {/* Run header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h2 className="page-title">{runId}</h2>
          <TypeBadge type={run.training_type} />
          <StatusBadge status={run.status} />
        </div>
        <div className="text-muted" style={{ fontSize: '0.8125rem' }}>
          {run.config_summary?.model_name != null && String(run.config_summary.model_name)}
          {run.total_steps != null && <span> · step {run.total_steps}</span>}
        </div>
      </div>

      {/* Stat cards — answer "is training working?" at a glance */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.5rem', marginBottom: '1rem' }}>
        {reward && <StatCard label="Reward" value={formatValue(reward.value)} trend={reward.trend} spark={reward.sparkData} color="var(--accent)" />}
        {correct && <StatCard label="Correct" value={formatValue(correct.value, true)} trend={correct.trend} spark={correct.sparkData} color="var(--cyan)" />}
        {kl && <StatCard label="KL Divergence" value={formatValue(kl.value)} trend={kl.trend} spark={kl.sparkData} color="var(--warning)" invertTrend />}
        {speed && <StatCard label="Step Time" value={speed.value !== null ? `${speed.value.toFixed(1)}s` : '-'} trend={speed.trend} spark={speed.sparkData} color="var(--purple)" invertTrend />}
      </div>

      {/* 4 tabs */}
      <div className="tabs">
        {tabs.map(({ id, label, disabled }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => !disabled && switchTab(id)}
            style={disabled ? { opacity: 0.3, cursor: 'default' } : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: activeTab === 'metrics' ? 'block' : 'none' }}>
        {visitedTabs.has('metrics') && <MetricsPanel runId={runId} onStepClick={handleMetricStepClick} />}
      </div>
      <div style={{ display: activeTab === 'rollouts' ? 'block' : 'none' }}>
        {visitedTabs.has('rollouts') && <RolloutBrowser runId={runId} iterations={iterations} jumpToStep={jumpToStep} />}
      </div>
      <div style={{ display: activeTab === 'timing' ? 'block' : 'none' }}>
        {visitedTabs.has('timing') && <TimingPanel runId={runId} jumpToStep={urlStep ? Number(urlStep) : null} />}
      </div>
      <div style={{ display: activeTab === 'eval' ? 'block' : 'none' }}>
        {visitedTabs.has('eval') && <EvalPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'checkpoints' ? 'block' : 'none' }}>
        {visitedTabs.has('checkpoints') && <CheckpointsPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'config' ? 'block' : 'none' }}>
        {visitedTabs.has('config') && <ConfigPanel config={run.config ?? {}} />}
      </div>
    </div>
  );
}

function StatCard({ label, value, trend, spark, color, invertTrend }: {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'flat';
  spark: number[];
  color: string;
  invertTrend?: boolean;
}) {
  // For KL and speed, "up" is bad (red) and "down" is good (green)
  const trendColor = invertTrend
    ? (trend === 'up' ? 'var(--error)' : trend === 'down' ? 'var(--accent)' : 'var(--text-muted)')
    : TREND_COLORS[trend];

  return (
    <div className="card" style={{ padding: '0.625rem 0.75rem' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
        {label}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <span className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color }}>
            {value}
          </span>
          {trend !== 'flat' && (
            <span style={{ fontSize: '0.8125rem', marginLeft: '0.25rem', color: trendColor, fontWeight: 600 }}>
              {TREND_ARROWS[trend]}
            </span>
          )}
        </div>
        <MiniSparkline data={spark} color={color} />
      </div>
    </div>
  );
}

function ConfigPanel({ config }: { config: Record<string, unknown> }) {
  const [search, setSearch] = useState('');
  const configStr = JSON.stringify(config, null, 2);
  const lines = configStr.split('\n');
  const filtered = search
    ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase()))
    : lines;

  const topLevelKeys = Object.keys(config);
  const scalarKeys = topLevelKeys.filter((k) => typeof config[k] !== 'object' || config[k] === null);
  const objectKeys = topLevelKeys.filter((k) => typeof config[k] === 'object' && config[k] !== null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  return (
    <div>
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span>Configuration</span>
          <input type="text" placeholder="Search..." value={search} onChange={(e) => setSearch(e.target.value)} style={{ width: '180px' }} />
        </div>
        {search ? (
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.6875rem' }}>{filtered.join('\n')}</pre>
        ) : (
          <div style={{ fontSize: '0.8125rem' }}>
            <table style={{ marginBottom: '0.75rem' }}>
              <tbody>
                {scalarKeys.map((key) => (
                  <tr key={key} style={{ cursor: 'default' }}>
                    <td className="mono text-muted" style={{ width: '200px', fontSize: '0.75rem' }}>{key}</td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>{String(config[key] ?? 'null')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {objectKeys.map((key) => (
              <div key={key} style={{ marginBottom: '0.375rem' }}>
                <div
                  onClick={() => setExpandedSections((prev) => { const n = new Set(prev); if (n.has(key)) n.delete(key); else n.add(key); return n; })}
                  style={{ cursor: 'pointer', padding: '0.375rem 0.5rem', background: 'var(--bg-elevated)', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.375rem' }}
                >
                  <span>{expandedSections.has(key) ? '\u25bc' : '\u25b6'}</span>
                  <span>{key}</span>
                  <span className="text-muted" style={{ fontWeight: 400, fontSize: '0.625rem' }}>
                    {Array.isArray(config[key]) ? `[${(config[key] as unknown[]).length} items]` : `{${Object.keys(config[key] as object).length} fields}`}
                  </span>
                </div>
                {expandedSections.has(key) && (
                  <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '0.6875rem', padding: '0.5rem', margin: '0.25rem 0 0 1rem', background: 'var(--bg-elevated)', borderRadius: '4px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(config[key], null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CheckpointsPanel({ runId }: { runId: string }) {
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getCheckpoints(runId)
      .then(setCheckpoints)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading checkpoints...</div>;
  if (checkpoints.length === 0) return <div className="empty-state">No checkpoints saved</div>;

  return (
    <div className="card" style={{ overflow: 'auto' }}>
      <div className="card-title" style={{ marginBottom: '0.5rem' }}>
        Checkpoints ({checkpoints.length})
      </div>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Step</th>
            <th>Final</th>
            <th>State Path</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map((ckpt, i) => (
            <tr key={i} style={{ cursor: 'default' }}>
              <td className="mono" style={{ fontWeight: 600 }}>{ckpt.name}</td>
              <td className="mono">{ckpt.batch ?? ckpt.loop_state?.batch ?? '-'}</td>
              <td>{ckpt.final ? 'Yes' : ''}</td>
              <td className="mono" style={{ fontSize: '0.6875rem', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {ckpt.state_path ?? '-'}
              </td>
              <td>
                {ckpt.sampler_path && (
                  <Link
                    to={`/runs/${runId}/chat?checkpoint=${ckpt.name}`}
                    onClick={(e) => e.stopPropagation()}
                    style={{ fontSize: '0.75rem', fontWeight: 600 }}
                  >
                    Chat →
                  </Link>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvalPanel({ runId }: { runId: string }) {
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
          const COLORS = ['#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a', '#ec4899', '#06b6d4', '#f97316'];
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
