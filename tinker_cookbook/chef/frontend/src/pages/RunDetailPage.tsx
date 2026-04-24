import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { CheckpointsTabPanel } from '../components/CheckpointsTabPanel';
import { ConfigTabPanel } from '../components/ConfigTabPanel';
import { EvalTabPanel } from '../components/EvalTabPanel';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { StatCard } from '../components/StatCard';
import { TimingPanel } from '../components/TimingPanel';
import { StatusBadge, TypeBadge } from '../utils/shared';
import { useUrlParam } from '../utils/useUrlParam';
import type { IterationInfo, MetricRecord, RunInfo } from '../api/types';

type Tab = 'metrics' | 'rollouts' | 'timing' | 'eval' | 'checkpoints' | 'config';

interface TabConfig {
  id: string;
  label: string;
  disabled?: boolean;
  render: () => React.ReactNode;
}

/** Pick the latest value and trend direction for a metric key. */
function getStatFromMetrics(records: MetricRecord[], key: string): { value: number | null; trend: 'up' | 'down' | 'flat'; sparkData: number[] } {
  const values = records.map((r) => r[key]).filter((v): v is number => typeof v === 'number');
  if (values.length === 0) return { value: null, trend: 'flat', sparkData: [] };
  const latest = values[values.length - 1];
  const prev = values.length > 1 ? values[values.length - 2] : latest;
  const trend: 'up' | 'down' | 'flat' = latest > prev + 0.001 ? 'up' : latest < prev - 0.001 ? 'down' : 'flat';
  return { value: latest, trend, sparkData: values.slice(-30) };
}

function formatValue(v: number | null, isPercent?: boolean): string {
  if (v === null) return '-';
  if (isPercent) return `${(v * 100).toFixed(1)}%`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

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

  const tabs: TabConfig[] = [
    { id: 'metrics', label: 'Metrics', render: () => <MetricsPanel runId={runId} onStepClick={handleMetricStepClick} /> },
    { id: 'rollouts', label: 'Rollouts', disabled: !hasIterations, render: () => <RolloutBrowser runId={runId} iterations={iterations} jumpToStep={jumpToStep} /> },
    { id: 'timing', label: 'Timing', disabled: !hasTiming, render: () => <TimingPanel runId={runId} jumpToStep={urlStep ? Number(urlStep) : null} /> },
    { id: 'eval', label: 'Eval', render: () => <EvalTabPanel runId={runId} /> },
    { id: 'checkpoints', label: 'Checkpoints', disabled: !hasCheckpoints, render: () => <CheckpointsTabPanel runId={runId} /> },
    { id: 'config', label: 'Config', render: () => <ConfigTabPanel config={run.config ?? {}} /> },
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

      <div className="tabs">
        {tabs.map(({ id, label, disabled }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => !disabled && switchTab(id as Tab)}
            style={disabled ? { opacity: 0.3, cursor: 'default' } : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      {tabs.map(({ id, render }) => (
        <div key={id} style={{ display: activeTab === id ? 'block' : 'none' }}>
          {visitedTabs.has(id as Tab) && render()}
        </div>
      ))}
    </div>
  );
}
