/** Detail card for a selected/hovered timing span. */
import { Link } from 'react-router-dom';
import type { FlatSpan } from './Waterfall';
import { getSpanColor } from './Waterfall';

interface Props {
  span: FlatSpan;
  spanNames: string[];
  isPinned: boolean;
  onDismiss: () => void;
  /** URL base for rollout links, e.g. "/runs/{id}/iterations/{step}/rollouts" */
  rolloutLinkBase?: string;
}

export function SpanDetail({ span, spanNames, isPinned, onDismiss, rolloutLinkBase }: Props) {
  const color = getSpanColor(span.name, spanNames);
  return (
    <div className="card" style={{ marginBottom: '0.75rem', borderColor: color }}>
      <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', fontSize: '0.8125rem', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Span</div>
          <div className="mono" style={{ fontWeight: 700, color }}>{span.name}</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Duration</div>
          <div className="mono" style={{ fontWeight: 600 }}>{span.duration.toFixed(4)}s</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Wall Start</div>
          <div className="mono">{span.wall_start.toFixed(4)}s</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Wall End</div>
          <div className="mono">{span.wall_end.toFixed(4)}s</div>
        </div>
        {span.attributes?.group_idx != null && rolloutLinkBase && (
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Rollout</div>
            <Link
              to={`${rolloutLinkBase}/${span.attributes.group_idx}/0`}
              style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', fontWeight: 600 }}
            >
              Group {String(span.attributes.group_idx)} →
            </Link>
          </div>
        )}
        {isPinned && (
          <button
            onClick={onDismiss}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.75rem' }}
          >
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
