/** Flamegraph: combined call hierarchy + waterfall in one visualization.
 *
 * Each span is a bar positioned by wall_start/wall_end.
 * Children render directly below their parent, creating a hierarchical
 * flamegraph where depth = nesting level.
 */

import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

interface SpanNode {
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
  children: SpanNode[];
  _concurrent?: number;
}

interface Props {
  root: SpanNode | null;
  totalDuration: number;
  runId: string;
  step: number;
}

const SPAN_COLORS: Record<string, string> = {
  sampling: '#8bbe3a',
  group_rollout: '#a78bfa',
  policy_sample: '#e5a11c',
  env_step: '#e85850',
  env_initial_observation: '#6aad7a',
  compute_group_rewards: '#ec4899',
  train_step: '#06b6d4',
  prepare_minibatch: '#f97316',
  assemble_training_data: '#64748b',
  save_checkpoint: '#14b8a6',
};

function getColor(name: string): string {
  if (SPAN_COLORS[name]) return SPAN_COLORS[name];
  // Hash-based fallback
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 60%, 55%)`;
}

const ROW_HEIGHT = 28;
const MIN_BAR_WIDTH_PX = 3;

export function Flamegraph({ root, totalDuration, runId, step }: Props) {
  const [selected, setSelected] = useState<SpanNode | null>(null);

  if (!root || totalDuration === 0) {
    return <div className="empty-state">No timing data for this step</div>;
  }

  const globalStart = root.wall_start;
  const globalEnd = root.wall_end;
  const range = globalEnd - globalStart || 1;

  type FlatEntry = { node: SpanNode; row: number };

  const { entries, maxRow } = useMemo(() => {
    const flatEntries: FlatEntry[] = [];
    const rowEnds: number[] = [];

    function findRow(start: number, minRow: number): number {
      for (let r = minRow; r < rowEnds.length; r++) {
        if (start >= rowEnds[r] - 0.001) return r;
      }
      rowEnds.push(0);
      return rowEnds.length - 1;
    }

    function walk(node: SpanNode, minRow: number) {
      if (node.name === 'iteration') {
        for (const child of node.children) walk(child, minRow);
        return;
      }
      const row = findRow(node.wall_start, minRow);
      rowEnds[row] = Math.max(rowEnds[row] || 0, node.wall_end);
      flatEntries.push({ node, row });
      for (const child of node.children) walk(child, row + 1);
    }

    walk(root, 0);
    return { entries: flatEntries, maxRow: rowEnds.length };
  }, [root]);

  const totalHeight = maxRow * ROW_HEIGHT + 30;

  // Time axis
  const numTicks = 8;
  const ticks = Array.from({ length: numTicks + 1 }, (_, i) => ({
    time: (i / numTicks) * range,
    pct: (i / numTicks) * 100,
  }));

  return (
    <div>
      <div style={{ position: 'relative', minHeight: totalHeight, overflow: 'hidden' }}>
        {/* Time grid */}
        <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: maxRow * ROW_HEIGHT, pointerEvents: 'none' }}>
          {ticks.map((t, i) => (
            <line key={i} x1={`${t.pct}%`} y1="0" x2={`${t.pct}%`} y2={maxRow * ROW_HEIGHT}
              stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 4" />
          ))}
        </svg>

        {/* Span bars */}
        {entries.map(({ node, row }, idx) => {
          const left = ((node.wall_start - globalStart) / range) * 100;
          const width = ((node.wall_end - node.wall_start) / range) * 100;
          const top = row * ROW_HEIGHT;
          const color = getColor(node.name);
          const isSelected = selected === node;
          const groupIdx = node.attributes?.group_idx;

          return (
            <div
              key={idx}
              onClick={(e) => { e.stopPropagation(); setSelected(isSelected ? null : node); }}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `max(${width}%, ${MIN_BAR_WIDTH_PX}px)`,
                top,
                height: ROW_HEIGHT - 3,
                borderRadius: 3,
                background: color,
                opacity: isSelected ? 1 : selected ? 0.5 : 0.85,
                border: isSelected ? '2px solid var(--text-primary)' : '1px solid rgba(0,0,0,0.1)',
                boxSizing: 'border-box',
                cursor: 'pointer',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
                display: 'flex',
                alignItems: 'center',
                padding: '0 4px',
                transition: 'opacity 0.1s',
              }}
              title={`${node.name} (${node.duration.toFixed(3)}s)${groupIdx != null ? ` [group ${groupIdx}]` : ''}`}
            >
              {width > 5 && (
                <span style={{ fontSize: '0.575rem', color: 'white', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {node.name}
                  {node._concurrent ? ` x${node._concurrent}` : ''}
                </span>
              )}
              {width > 12 && (
                <span style={{ fontSize: '0.5rem', color: 'rgba(255,255,255,0.7)', marginLeft: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                  {node.duration.toFixed(2)}s
                </span>
              )}
            </div>
          );
        })}

        {/* Time axis */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          display: 'flex', justifyContent: 'space-between',
          fontSize: '0.5625rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
        }}>
          {ticks.map((t, i) => (
            <span key={i}>{t.time.toFixed(2)}s</span>
          ))}
        </div>
      </div>

      {/* Selected span detail */}
      {selected && (
        <div className="card" style={{ marginTop: '0.5rem', borderColor: getColor(selected.name) }}>
          <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', fontSize: '0.8125rem', alignItems: 'center' }}>
            <div>
              <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Span </span>
              <span className="mono" style={{ fontWeight: 700, color: getColor(selected.name) }}>{selected.name}</span>
            </div>
            <span className="mono" style={{ fontWeight: 600 }}>{selected.duration.toFixed(4)}s</span>
            <span className="mono text-muted">{selected.wall_start.toFixed(3)}s – {selected.wall_end.toFixed(3)}s</span>
            {selected.attributes?.group_idx != null && (
              <Link
                to={`/runs/${runId}/iterations/${step}/rollouts/${selected.attributes.group_idx}/0`}
                style={{ fontWeight: 600 }}
                onClick={() => setSelected(null)}
              >
                View rollout (group {String(selected.attributes.group_idx)}) →
              </Link>
            )}
            <button
              onClick={() => setSelected(null)}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.75rem' }}
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
