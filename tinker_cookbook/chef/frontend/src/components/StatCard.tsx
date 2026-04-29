const TREND_ARROWS: Record<string, string> = { up: '\u2191', down: '\u2193', flat: '' };
const TREND_COLORS: Record<string, string> = { up: 'var(--accent)', down: 'var(--error)', flat: 'var(--text-muted)' };

export function MiniSparkline({ data, color }: { data: number[]; color: string }) {
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

export function StatCard({ label, value, trend, spark, color, invertTrend }: {
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
