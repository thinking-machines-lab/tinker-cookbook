/** Unified tooltip for Recharts charts — consistent styling across all panels. */

import type { TooltipProps } from 'recharts';

const TOOLTIP_STYLE: React.CSSProperties = {
  background: 'var(--bg-surface)',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  fontSize: '0.6875rem',
  padding: '0.1875rem 0.5rem',
  pointerEvents: 'none',
};

interface ChartTooltipItem {
  label: string;
  value: string | number;
  color: string;
}

interface Props {
  /** Override default Recharts tooltip rendering with custom items. */
  items?: ChartTooltipItem[];
  /** Format a numeric value for display. Defaults to 4 significant digits. */
  formatValue?: (value: number) => string;
  /** Show only specific dataKeys (e.g., the hovered one). */
  filterKeys?: Set<string>;
  /** Keys to exclude from display. */
  hideKeys?: Set<string>;
  /** Unit suffix (e.g., "%", "ms"). */
  unit?: string;
}

function defaultFormat(v: number): string {
  return v.toPrecision(4);
}

/**
 * Recharts-compatible tooltip content renderer.
 *
 * Usage with Recharts:
 * ```tsx
 * <Tooltip content={<ChartTooltipContent hideKeys={hidden} />} />
 * ```
 */
export function ChartTooltipContent({
  formatValue = defaultFormat,
  filterKeys,
  hideKeys,
  unit,
}: Props) {
  return function TooltipContent({ active, label, payload }: TooltipProps<number, string>) {
    if (!active || !payload || payload.length === 0) return null;

    const visible = payload.filter((p) => {
      const key = String(p.dataKey);
      if (key.startsWith('_raw_')) return false;
      if (hideKeys?.has(key)) return false;
      if (filterKeys && !filterKeys.has(key)) return false;
      return typeof p.value === 'number';
    });

    if (visible.length === 0) return null;

    return (
      <div style={TOOLTIP_STYLE}>
        <span className="mono" style={{ color: 'var(--text-muted)', marginRight: '0.375rem' }}>
          {label}
        </span>
        {visible.map((item) => (
          <span key={String(item.dataKey)} style={{ color: String(item.color), marginLeft: '0.25rem' }}>
            {String(item.name)}: <span className="mono">{formatValue(Number(item.value))}{unit ?? ''}</span>
          </span>
        ))}
      </div>
    );
  };
}

/** Simple tooltip style for use with Recharts contentStyle prop. */
export const TOOLTIP_CONTENT_STYLE: React.CSSProperties = {
  background: 'var(--bg-surface)',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  fontSize: '0.75rem',
};
