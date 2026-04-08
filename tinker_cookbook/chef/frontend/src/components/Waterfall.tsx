/** Waterfall visualization for timing spans — shows bars on a timeline. */

export interface FlatSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
}

const SPAN_PALETTE = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

export function getSpanColor(name: string, names: string[]): string {
  const idx = names.indexOf(name);
  return SPAN_PALETTE[idx % SPAN_PALETTE.length];
}

interface Props {
  spans: FlatSpan[];
  spanNames: string[];
  hoveredSpan: FlatSpan | null;
  onHover: (span: FlatSpan | null) => void;
  onClick?: (span: FlatSpan) => void;
}

export function Waterfall({ spans, spanNames, hoveredSpan, onHover, onClick }: Props) {
  const minWall = Math.min(...spans.map((s) => s.wall_start));
  const maxWall = Math.max(...spans.map((s) => s.wall_end));
  const totalDuration = maxWall - minWall || 1;

  // Assign lanes to avoid overlap — greedy lane packing
  const lanes: { end: number }[] = [];
  const laneAssignments: number[] = [];

  for (const span of spans) {
    let assigned = false;
    for (let i = 0; i < lanes.length; i++) {
      if (span.wall_start >= lanes[i].end - 0.001) {
        lanes[i].end = span.wall_end;
        laneAssignments.push(i);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      lanes.push({ end: span.wall_end });
      laneAssignments.push(lanes.length - 1);
    }
  }

  const rowHeight = 26;
  const totalHeight = lanes.length * rowHeight + 24;

  const numTicks = 6;
  const ticks = Array.from({ length: numTicks + 1 }, (_, i) => ({
    time: (i / numTicks) * totalDuration,
    x: (i / numTicks) * 100,
  }));

  return (
    <div style={{ position: 'relative', overflow: 'hidden' }}>
      <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: totalHeight, pointerEvents: 'none' }}>
        {ticks.map((tick, i) => (
          <line key={i} x1={`${tick.x}%`} y1="0" x2={`${tick.x}%`} y2={totalHeight - 20}
            stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 4" />
        ))}
      </svg>

      <div style={{ position: 'relative', height: totalHeight - 20 }}>
        {spans.map((span, idx) => {
          const left = ((span.wall_start - minWall) / totalDuration) * 100;
          const width = ((span.wall_end - span.wall_start) / totalDuration) * 100;
          const lane = laneAssignments[idx];
          const isHovered = hoveredSpan === span;
          const color = getSpanColor(span.name, spanNames);

          return (
            <div
              key={idx}
              onMouseEnter={() => onHover(span)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onClick?.(span)}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${Math.max(width, 0.3)}%`,
                top: lane * rowHeight,
                height: rowHeight - 4,
                borderRadius: 4,
                background: color,
                opacity: isHovered ? 1 : 0.8,
                border: isHovered ? '2px solid var(--text-primary)' : '1px solid transparent',
                boxSizing: 'border-box',
                display: 'flex',
                alignItems: 'center',
                padding: '0 6px',
                cursor: 'pointer',
                transition: 'opacity 0.1s, border 0.1s',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
              }}
            >
              {width > 8 && (
                <span style={{ fontSize: '0.6rem', color: 'white', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {span.name}
                </span>
              )}
              {width > 15 && (
                <span style={{ fontSize: '0.55rem', color: 'rgba(255,255,255,0.7)', marginLeft: '0.375rem', fontFamily: 'var(--font-mono)' }}>
                  {span.duration.toFixed(3)}s
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.5625rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', paddingTop: 4 }}>
        {ticks.map((tick, i) => (
          <span key={i}>{tick.time.toFixed(2)}s</span>
        ))}
      </div>
    </div>
  );
}
