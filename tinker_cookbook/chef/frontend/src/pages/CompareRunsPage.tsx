import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';
import type { MetricRecord } from '../api/types';
import { RUN_COLORS } from '../theme/colors';

interface RunData {
  runId: string;
  records: MetricRecord[];
  keys: Set<string>;
}

function groupKeys(allKeys: Set<string>): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const key of allKeys) {
    if (key.endsWith(':total') || key.endsWith(':count')) continue;
    const slashIdx = key.indexOf('/');
    const prefix = slashIdx > 0 ? key.substring(0, slashIdx) : 'general';
    if (!groups.has(prefix)) groups.set(prefix, []);
    groups.get(prefix)!.push(key);
  }
  return groups;
}

export function CompareRunsPage() {
  const [searchParams] = useSearchParams();
  const runIds = searchParams.get('runs')?.split(',').filter(Boolean) ?? [];
  const [runData, setRunData] = useState<RunData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  useEffect(() => {
    if (runIds.length === 0) {
      setLoading(false);
      return;
    }
    Promise.all(
      runIds.map(async (runId) => {
        const [metricsResp, keys] = await Promise.all([
          api.getMetrics(runId),
          api.getMetricKeys(runId),
        ]);
        return {
          runId,
          records: metricsResp.records,
          keys: new Set(keys),
        };
      })
    )
      .then(setRunData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [searchParams.get('runs')]);

  // Collect all unique metric keys across all runs
  const allKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const rd of runData) {
      for (const k of rd.keys) keys.add(k);
    }
    return keys;
  }, [runData]);

  const groups = useMemo(() => groupKeys(allKeys), [allKeys]);

  if (loading) return <div className="loading">Loading comparison...</div>;
  if (runIds.length === 0) {
    return (
      <div className="empty-state">
        <p>Select 2-4 runs from the run list to compare.</p>
        <Link to="/">Back to runs</Link>
      </div>
    );
  }

  // For the selected metric, build an overlay chart
  const allMetrics = Array.from(allKeys).filter(
    (k) => !k.endsWith(':total') && !k.endsWith(':count')
  ).sort();

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Runs</Link>
        <span>/</span>
        <span>Compare</span>
      </div>

      <h2 className="page-title">Compare Runs</h2>
      <div className="page-subtitle">
        {runData.map((rd, i) => (
          <span key={rd.runId}>
            <span style={{ color: RUN_COLORS[i], fontWeight: 600 }}>{rd.runId}</span>
            {i < runData.length - 1 && <span className="text-muted"> vs </span>}
          </span>
        ))}
      </div>

      {/* Metric selector for overlay view */}
      <div className="filters-bar" style={{ marginTop: '0.75rem' }}>
        <div className="filter-group">
          <span className="filter-label">Overlay Metric</span>
          <select
            value={selectedMetric ?? ''}
            onChange={(e) => setSelectedMetric(e.target.value || null)}
          >
            <option value="">Select a metric...</option>
            {allMetrics.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Single-metric overlay chart */}
      {selectedMetric && (
        <div className="card" style={{ marginBottom: '1rem' }}>
          <div className="card-header">
            <span className="card-title">{selectedMetric}</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} type="number" allowDuplicatedCategory={false} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '0.75rem' }}
                labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
              />
              {runData.map((rd, i) => (
                <Line
                  key={rd.runId}
                  data={rd.records}
                  type="monotone"
                  dataKey={selectedMetric}
                  stroke={RUN_COLORS[i]}
                  dot={false}
                  strokeWidth={2}
                  name={rd.runId}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-group comparison charts */}
      <div className="charts-grid">
        {Array.from(groups.entries()).map(([_prefix, groupKeys]) => {
          return groupKeys.slice(0, 4).map((metricKey) => (
            <div key={metricKey} className="chart-card">
              <div className="chart-title">
                {metricKey.includes('/') ? metricKey.split('/').slice(1).join('/') : metricKey}
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 9 }} type="number" allowDuplicatedCategory={false} />
                  <YAxis stroke="var(--text-muted)" tick={{ fontSize: 9 }} width={50} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '0.7rem' }}
                  />
                  {runData.map((rd, i) => (
                    <Line
                      key={rd.runId}
                      data={rd.records}
                      type="monotone"
                      dataKey={metricKey}
                      stroke={RUN_COLORS[i]}
                      dot={false}
                      strokeWidth={1.5}
                      name={rd.runId}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              {/* Legend */}
              <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.625rem', marginTop: '0.25rem' }}>
                {runData.map((rd, i) => (
                  <span key={rd.runId} style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: RUN_COLORS[i] }} />
                    <span className="text-muted">{rd.runId}</span>
                  </span>
                ))}
              </div>
            </div>
          ));
        })}
      </div>
    </div>
  );
}
