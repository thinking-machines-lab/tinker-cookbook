/** Shared interactive chart legend — click to hide, hover to highlight. */

interface Props {
  keys: string[];
  colors: string[];
  hidden: Set<string>;
  onToggle: (key: string) => void;
  hoveredKey?: string | null;
  onHover?: (key: string | null) => void;
  /** Shorten keys by removing the shared prefix (e.g., "env/all/reward" → "reward"). */
  shortName?: (key: string) => string;
}

function defaultShortName(key: string): string {
  return key.includes('/') ? key.split('/').slice(1).join('/') : key;
}

export function ChartLegend({
  keys,
  colors,
  hidden,
  onToggle,
  hoveredKey,
  onHover,
  shortName = defaultShortName,
}: Props) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.625rem', marginTop: '0.375rem', paddingLeft: '0.25rem' }}>
      {keys.map((key, i) => (
        <button
          key={key}
          onClick={() => onToggle(key)}
          onMouseEnter={() => onHover?.(key)}
          onMouseLeave={() => onHover?.(null)}
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
            background: colors[i % colors.length],
            opacity: hidden.has(key) ? 0.3 : 1, flexShrink: 0,
          }} />
          {shortName(key)}
        </button>
      ))}
    </div>
  );
}
