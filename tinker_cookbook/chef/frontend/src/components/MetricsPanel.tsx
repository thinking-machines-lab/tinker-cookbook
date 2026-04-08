import { useEffect, useMemo, useRef, useState } from 'react';
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
import { applyEMA, groupMetricKeys } from '../utils/metrics';

const COLORS = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

interface Props {
  runId: string;
  onStepClick?: (step: number, group: string) => void;
}

const EMA_OPTIONS = [
  { label: 'Off', alpha: 0 },
  { label: '0.6', alpha: 0.6 },
  { label: '0.9', alpha: 0.9 },
  { label: '0.95', alpha: 0.95 },
  { label: '0.99', alpha: 0.99 },
];

function MetricChart({ prefix, metricKeys, data, onStepClick }: {
  prefix: string;
  metricKeys: string[];
  data: MetricRecord[];
  onStepClick?: (step: number, group: string) => void;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [emaAlpha, setEmaAlpha] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  const toggleKey = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const shortName = (key: string) => key.includes('/') ? key.split('/').slice(1).join('/') : key;

  const smoothedData = useMemo(
    () => emaAlpha > 0 ? applyEMA(data, metricKeys, emaAlpha) : null,
    [data, metricKeys, emaAlpha]
  );

  // When smoothing is active, merge raw + smoothed into one array so Recharts
  // uses a single data source and the x-axis domain stays stable.
  const chartData = useMemo(() => {
    if (!smoothedData) return data;
    return data.map((raw, i) => {
      const merged: MetricRecord = { step: raw.step };
      for (const key of metricKeys) {
        if (typeof raw[key] === 'number') merged[`_raw_${key}`] = raw[key];
      }
      const sm = smoothedData[i];
      if (sm) {
        for (const key of metricKeys) {
          if (typeof sm[key] === 'number') merged[key] = sm[key];
        }
      }
      return merged;
    });
  }, [data, smoothedData, metricKeys]);

  return (
    <div className="chart-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
        <div
          className="chart-title"
          onClick={() => setCollapsed(!collapsed)}
          style={{ cursor: 'pointer', userSelect: 'none' }}
        >
          <span style={{ fontSize: '0.625rem', marginRight: '0.25rem' }}>{collapsed ? '\u25b6' : '\u25bc'}</span>
          {prefix}
          <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', marginLeft: '0.375rem', fontWeight: 400 }}>
            {metricKeys.length} metric{metricKeys.length > 1 ? 's' : ''}
          </span>
        </div>
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.5625rem', color: 'var(--text-muted)' }}>smooth:</span>
            <div style={{ display: 'flex', gap: '1px' }}>
              {EMA_OPTIONS.map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => setEmaAlpha(opt.alpha)}
                  style={{
                    padding: '1px 5px',
                    border: 'none',
                    borderRadius: '3px',
                    fontSize: '0.5625rem',
                    fontFamily: 'var(--font-mono)',
                    cursor: 'pointer',
                    background: emaAlpha === opt.alpha ? 'var(--accent-dim)' : 'transparent',
                    color: emaAlpha === opt.alpha ? 'var(--accent)' : 'var(--text-muted)',
                    fontWeight: emaAlpha === opt.alpha ? 600 : 400,
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            {onStepClick && (
              <span style={{ fontSize: '0.5rem', color: 'var(--text-muted)', fontStyle: 'italic', marginLeft: '0.25rem' }}>
                click → rollouts
              </span>
            )}
          </div>
        )}
      </div>
      {collapsed ? null : (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={chartData}
          onClick={onStepClick ? (e: unknown) => {
            const ev = e as { activeLabel?: string | number; activePayload?: { payload?: { step?: number } }[] };
            // activeLabel is the x-axis value (step) at the click position
            const step = ev?.activePayload?.[0]?.payload?.step ?? (typeof ev?.activeLabel === 'number' ? ev.activeLabel : undefined);
            if (step != null) {
              onStepClick(step, prefix);
            }
          } : undefined}
          style={onStepClick ? { cursor: 'pointer' } : undefined}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} width={55} />
          <Tooltip
            content={({ active, label, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              // Filter to visible, non-raw, smoothed (or raw if no smoothing) lines
              const visible = payload.filter((p) =>
                !String(p.dataKey).startsWith('_raw_') && !hidden.has(String(p.dataKey))
                && typeof p.value === 'number'
              );
              if (visible.length === 0) return null;
              // Show only the hovered key, or all if <=2 visible
              const show = hoveredKey
                ? visible.filter((p) => String(p.dataKey) === hoveredKey)
                : visible.length <= 2 ? visible : [];
              if (show.length === 0) return null;
              return (
                <div style={{
                  background: 'var(--bg-surface)', border: '1px solid var(--border)',
                  borderRadius: '6px', fontSize: '0.6875rem', padding: '0.1875rem 0.5rem',
                  pointerEvents: 'none',
                }}>
                  <span className="mono" style={{ color: 'var(--text-muted)', marginRight: '0.375rem' }}>
                    {label}
                  </span>
                  {show.map((item) => (
                    <span key={String(item.dataKey)} style={{ color: String(item.color) }}>
                      {String(item.name)}: <span className="mono">{typeof item.value === 'number' ? item.value.toPrecision(4) : item.value}</span>
                    </span>
                  ))}
                </div>
              );
            }}
          />
          {/* Raw data lines (faded when smoothing active) */}
          {metricKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={smoothedData ? `_raw_${key}` : key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={smoothedData ? 0.5 : (hoveredKey === key ? 2.5 : 1.5)}
              strokeOpacity={hidden.has(key) ? 0 : (smoothedData ? 0.3 : (hoveredKey && hoveredKey !== key ? 0.15 : 1))}
              name={smoothedData ? `${shortName(key)} (raw)` : shortName(key)}
              connectNulls
              hide={hidden.has(key)}
            />
          ))}
          {/* Smoothed lines (same data source, no domain conflict) */}
          {smoothedData && metricKeys.map((key, i) => (
            <Line
              key={`${key}_ema`}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={hoveredKey === key ? 3 : 2}
              strokeOpacity={hidden.has(key) ? 0 : (hoveredKey && hoveredKey !== key ? 0.15 : 1)}
              name={`${shortName(key)} (EMA)`}
              connectNulls
              hide={hidden.has(key)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      )}
      {/* Legend — click to hide, hover to focus tooltip */}
      {!collapsed && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.625rem', marginTop: '0.375rem', paddingLeft: '0.25rem' }}>
          {metricKeys.map((key, i) => (
            <button
              key={key}
              onClick={() => toggleKey(key)}
              onMouseEnter={() => setHoveredKey(key)}
              onMouseLeave={() => setHoveredKey(null)}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.25rem',
                background: hoveredKey === key ? 'var(--bg-elevated)' : 'none',
                border: 'none', cursor: 'pointer',
                fontSize: '0.6875rem', fontFamily: 'var(--font-mono)',
                color: hidden.has(key) ? 'var(--text-muted)' : 'var(--text-secondary)',
                opacity: hidden.has(key) ? 0.5 : (hoveredKey && hoveredKey !== key ? 0.4 : 1),
                padding: '0.125rem 0.25rem',
                borderRadius: '4px',
                textDecoration: hidden.has(key) ? 'line-through' : 'none',
              }}
            >
              <span style={{
                width: 8, height: 8, borderRadius: '50%',
                background: COLORS[i % COLORS.length],
                opacity: hidden.has(key) ? 0.3 : 1, flexShrink: 0,
              }} />
              {shortName(key)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function MetricsPanel({ runId, onStepClick }: Props) {
  const [records, setRecords] = useState<MetricRecord[]>([]);
  const [keys, setKeys] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    Promise.all([api.getMetrics(runId), api.getMetricKeys(runId)])
      .then(([metricsResp, metricKeys]) => {
        setRecords(metricsResp.records);
        setKeys(metricKeys);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    function handleMessage(event: MessageEvent) {
      try {
        const record = JSON.parse(event.data) as MetricRecord;
        setRecords((prev) => [...prev, record]);
        const newKeys = Object.keys(record).filter((k) => k !== 'step');
        setKeys((prev) => {
          const existing = new Set(prev);
          const added = newKeys.filter((k) => !existing.has(k));
          return added.length > 0 ? [...prev, ...added] : prev;
        });
      } catch { /* ignore keepalives */ }
    }

    function connect() {
      const es = new EventSource(api.metricsStreamUrl(runId));
      esRef.current = es;
      es.onmessage = handleMessage;
      es.onerror = () => {
        es.close();
        if (esRef.current === es) {
          esRef.current = null;
          setTimeout(() => { if (esRef.current === null) connect(); }, 5000);
        }
      };
    }

    connect();
    return () => { if (esRef.current) { esRef.current.close(); esRef.current = null; } };
  }, [runId]);

  const groups = useMemo(() => groupMetricKeys(keys), [keys]);

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (records.length === 0) return <div className="empty-state">No metrics data yet</div>;

  return (
    <div className="charts-grid">
      {Array.from(groups.entries()).map(([prefix, groupKeys]) => (
        <MetricChart key={prefix} prefix={prefix} metricKeys={groupKeys} data={records} onStepClick={onStepClick} />
      ))}
    </div>
  );
}
