import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { RewardHistogram } from './charts';
import { SortableTable } from './SortableTable';
import type { IterationInfo, RolloutSummary } from '../api/types';

/** Returns a debounced version of `value`, updating only after `delay` ms of inactivity. */
function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => {
    timerRef.current = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timerRef.current);
  }, [value, delay]);
  return debounced;
}

interface Props {
  runId: string;
  iterations: IterationInfo[];
  jumpToStep?: number | null;  // set externally to jump to the nearest iteration
}

function rewardClass(reward: number): string {
  if (reward >= 0.8) return 'high';
  if (reward >= 0.3) return 'mid';
  return 'low';
}

export function RolloutBrowser({ runId, iterations, jumpToStep }: Props) {
  const navigate = useNavigate();
  const [selectedIter, setSelectedIter] = useState<number | null>(null);
  const [rollouts, setRollouts] = useState<RolloutSummary[]>([]);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [tagFilter, setTagFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [minReward, setMinReward] = useState<string>('');
  const [maxReward, setMaxReward] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const debouncedMinReward = useDebouncedValue(minReward, 300);
  const debouncedMaxReward = useDebouncedValue(maxReward, 300);

  const iterationsWithRollouts = iterations.filter((it) => it.has_train_rollouts);

  useEffect(() => {
    if (iterationsWithRollouts.length > 0 && selectedIter === null) {
      setSelectedIter(iterationsWithRollouts[0].iteration);
    }
  }, [iterations, selectedIter]);

  // Jump to the closest iteration when linked from metrics chart
  useEffect(() => {
    if (jumpToStep == null || iterationsWithRollouts.length === 0) return;
    const closest = iterationsWithRollouts.reduce((prev, curr) =>
      Math.abs(curr.iteration - jumpToStep) < Math.abs(prev.iteration - jumpToStep) ? curr : prev
    );
    setSelectedIter(closest.iteration);
  }, [jumpToStep]);

  useEffect(() => {
    if (selectedIter === null) return;
    setLoading(true);
    api
      .getRollouts(runId, selectedIter, {
        tag: tagFilter || undefined,
        min_reward: debouncedMinReward !== '' ? Number(debouncedMinReward) : undefined,
        max_reward: debouncedMaxReward !== '' ? Number(debouncedMaxReward) : undefined,
      })
      .then((resp) => {
        setRollouts(resp.rollouts);
        setAvailableTags(resp.available_tags);
      })
      .catch(() => setRollouts([]))
      .finally(() => setLoading(false));
  }, [runId, selectedIter, tagFilter, debouncedMinReward, debouncedMaxReward]);

  if (iterationsWithRollouts.length === 0) {
    return <div className="empty-state">
      {iterations.length === 0 ? 'No iteration data available' : 'No iterations have rollout data'}
    </div>;
  }

  // Status counts for summary line
  const statusCounts = {
    ok: rollouts.filter((r) => r.status === 'ok' || !r.status).length,
    error: rollouts.filter((r) => r.status === 'error').length,
    timeout: rollouts.filter((r) => r.status === 'timeout').length,
    zeroReward: rollouts.filter((r) => r.total_reward === 0).length,
  };

  const columns = [
    {
      key: 'group',
      label: 'Group',
      render: (r: RolloutSummary) => (
        <Link
          to={`/runs/${runId}/iterations/${selectedIter}/groups/${r.group_idx}`}
          onClick={(e) => e.stopPropagation()}
          className="mono"
          style={{ fontWeight: 600 }}
        >
          {r.group_idx}
        </Link>
      ),
      sortValue: (r: RolloutSummary) => r.group_idx,
    },
    {
      key: 'traj',
      label: 'Traj',
      render: (r: RolloutSummary) => <span className="mono">{r.traj_idx}</span>,
      sortValue: (r: RolloutSummary) => r.traj_idx,
    },
    {
      key: 'status',
      label: 'Status',
      render: (r: RolloutSummary) => {
        if (r.status === 'error') return <span className="tag" style={{ background: 'rgba(239,68,68,0.15)', color: 'var(--error)' }}>Error</span>;
        if (r.status === 'timeout') return <span className="tag" style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--warning)' }}>Timeout</span>;
        if (r.total_reward === 0) return <span className="tag" style={{ background: 'rgba(100,116,139,0.15)', color: 'var(--text-muted)' }}>Zero</span>;
        return null;
      },
    },
    {
      key: 'tags',
      label: 'Tags',
      render: (r: RolloutSummary) => (
        <>{r.tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</>
      ),
    },
    {
      key: 'steps',
      label: 'Steps',
      render: (r: RolloutSummary) => <span className="mono">{r.num_steps}</span>,
      sortValue: (r: RolloutSummary) => r.num_steps,
    },
    {
      key: 'tokens',
      label: 'Tokens',
      render: (r: RolloutSummary) => <span className="mono">{r.total_tokens}</span>,
      sortValue: (r: RolloutSummary) => r.total_tokens,
    },
    {
      key: 'total_reward',
      label: 'Reward',
      render: (r: RolloutSummary) => (
        <span className={`reward-badge ${rewardClass(r.total_reward)}`}>
          {r.total_reward.toFixed(3)}
        </span>
      ),
      sortValue: (r: RolloutSummary) => r.total_reward,
    },
    {
      key: 'context',
      label: 'Context',
      render: (r: RolloutSummary) => <span className="mono">{r.final_ob_len}</span>,
      sortValue: (r: RolloutSummary) => r.final_ob_len,
    },
  ];

  return (
    <div>
      <div className="filters-bar">
        <div className="filter-group" style={{ flex: '1 1 auto', minWidth: '200px' }}>
          <span className="filter-label">Iteration</span>
          <button
            className="tab"
            onClick={() => {
              const idx = iterationsWithRollouts.findIndex((it) => it.iteration === selectedIter);
              if (idx > 0) setSelectedIter(iterationsWithRollouts[idx - 1].iteration);
            }}
            disabled={selectedIter === iterationsWithRollouts[0]?.iteration}
            style={{ opacity: selectedIter === iterationsWithRollouts[0]?.iteration ? 0.3 : 1, padding: '0.1875rem 0.375rem' }}
          >
            Prev
          </button>
          <input
            type="range"
            min={0}
            max={iterationsWithRollouts.length - 1}
            value={iterationsWithRollouts.findIndex((it) => it.iteration === selectedIter)}
            onChange={(e) => setSelectedIter(iterationsWithRollouts[Number(e.target.value)]?.iteration ?? 0)}
            style={{ flex: 1, minWidth: '100px', accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <button
            className="tab"
            onClick={() => {
              const idx = iterationsWithRollouts.findIndex((it) => it.iteration === selectedIter);
              if (idx < iterationsWithRollouts.length - 1) setSelectedIter(iterationsWithRollouts[idx + 1].iteration);
            }}
            disabled={selectedIter === iterationsWithRollouts[iterationsWithRollouts.length - 1]?.iteration}
            style={{ opacity: selectedIter === iterationsWithRollouts[iterationsWithRollouts.length - 1]?.iteration ? 0.3 : 1, padding: '0.1875rem 0.375rem' }}
          >
            Next
          </button>
          <input
            type="number"
            value={selectedIter ?? ''}
            onChange={(e) => {
              const val = Number(e.target.value);
              const closest = iterationsWithRollouts.reduce((prev, curr) =>
                Math.abs(curr.iteration - val) < Math.abs(prev.iteration - val) ? curr : prev
              );
              setSelectedIter(closest.iteration);
            }}
            style={{ width: '60px' }}
          />
          <span className="text-muted" style={{ fontSize: '0.625rem', whiteSpace: 'nowrap' }}>
            / {iterationsWithRollouts[iterationsWithRollouts.length - 1]?.iteration ?? 0}
          </span>
        </div>

        <div className="filter-group">
          <span className="filter-label">Status</span>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">All</option>
            <option value="error">Errors</option>
            <option value="timeout">Timeouts</option>
            <option value="zero">Zero Reward</option>
          </select>
        </div>

        {availableTags.length > 0 && (
          <div className="filter-group">
            <span className="filter-label">Tag</span>
            <select value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
              <option value="">All</option>
              {availableTags.map((tag) => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>
        )}

        <div className="filter-group">
          <span className="filter-label">Reward</span>
          <input
            type="number"
            placeholder="min"
            value={minReward}
            onChange={(e) => setMinReward(e.target.value)}
            step="0.1"
            style={{
              width: '60px',
              padding: '0.3125rem 0.375rem',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              fontSize: '0.8125rem',
            }}
          />
          <span className="text-muted" style={{ fontSize: '0.75rem' }}>to</span>
          <input
            type="number"
            placeholder="max"
            value={maxReward}
            onChange={(e) => setMaxReward(e.target.value)}
            step="0.1"
            style={{
              width: '60px',
              padding: '0.3125rem 0.375rem',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              fontSize: '0.8125rem',
            }}
          />
        </div>

        <div style={{ marginLeft: 'auto', fontSize: '0.6875rem', color: 'var(--text-muted)', display: 'flex', gap: '0.5rem' }}>
          <span>{rollouts.length} rollouts</span>
          {statusCounts.error > 0 && <span style={{ color: 'var(--error)' }}>{statusCounts.error} errors</span>}
          {statusCounts.timeout > 0 && <span style={{ color: 'var(--warning)' }}>{statusCounts.timeout} timeouts</span>}
          {statusCounts.zeroReward > 0 && <span>{statusCounts.zeroReward} zero-reward</span>}
        </div>
      </div>

      {/* Reward distribution histogram */}
      {!loading && rollouts.length > 0 && (
        <div className="card" style={{ marginBottom: '0.5rem', padding: '0.5rem 0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Reward Distribution
            </span>
            <span className="text-muted" style={{ fontSize: '0.625rem' }}>
              click bin to filter
            </span>
          </div>
          <RewardHistogram
            rewards={rollouts.map((r) => r.total_reward)}
            onBinClick={(min, max) => {
              setMinReward(min.toFixed(2));
              setMaxReward(max.toFixed(2));
            }}
          />
        </div>
      )}

      {loading ? (
        <div className="loading">Loading rollouts...</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'auto' }}>
          <SortableTable
            columns={columns}
            data={statusFilter
              ? rollouts.filter((r) =>
                  statusFilter === 'error' ? r.status === 'error'
                  : statusFilter === 'timeout' ? r.status === 'timeout'
                  : statusFilter === 'zero' ? r.total_reward === 0
                  : true)
              : rollouts}
            rowKey={(r) => `${r.group_idx}-${r.traj_idx}`}
            onRowClick={(r) =>
              navigate(`/runs/${runId}/iterations/${selectedIter}/rollouts/${r.group_idx}/${r.traj_idx}`)
            }
          />
        </div>
      )}
    </div>
  );
}
