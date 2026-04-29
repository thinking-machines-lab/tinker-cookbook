import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { SortableTable } from '../components/SortableTable';
import { ResultTag, scoreColor } from '../utils/shared';
import { useUrlParam } from '../utils/useUrlParam';
import type { EvalRunDetail, EvalTrajectorySummary } from '../api/types';

export function EvalRunDetailPage() {
  const { evalRunId } = useParams<{ evalRunId: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<EvalRunDetail | null>(null);
  const [urlBenchmark, setUrlBenchmark] = useUrlParam('benchmark', '');
  const [selectedBenchmark, setSelectedBenchmarkState] = useState<string | null>(urlBenchmark || null);
  const [trajectories, setTrajectories] = useState<EvalTrajectorySummary[]>([]);
  const [filter, setFilter] = useState<'all' | 'correct' | 'incorrect' | 'errors'>('all');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  const setSelectedBenchmark = (b: string | null) => {
    setSelectedBenchmarkState(b);
    setUrlBenchmark(b ?? '');
  };

  useEffect(() => {
    if (!evalRunId) return;
    api.getEvalRun(evalRunId)
      .then((data) => {
        setRun(data);
        const initial = urlBenchmark && data.benchmarks.includes(urlBenchmark) ? urlBenchmark : data.benchmarks[0];
        if (initial) setSelectedBenchmark(initial);
      })
      .catch(() => setRun(null))
      .finally(() => setLoading(false));
  }, [evalRunId]);

  useEffect(() => {
    if (!evalRunId || !selectedBenchmark) return;
    api.getEvalTrajectories(evalRunId, selectedBenchmark, {
      correct_only: filter === 'correct',
      incorrect_only: filter === 'incorrect',
      errors_only: filter === 'errors',
    })
      .then((resp) => setTrajectories(resp.trajectories))
      .catch(() => setTrajectories([]));
  }, [evalRunId, selectedBenchmark, filter]);

  if (loading) return <div className="loading">Loading eval run...</div>;
  if (!run || !evalRunId) return <div className="empty-state">Eval run not found</div>;

  const metadata = run.metadata as Record<string, unknown>;

  const trajColumns = [
    {
      key: 'idx',
      label: '#',
      render: (t: EvalTrajectorySummary) => <span className="mono">{t.idx}</span>,
      sortValue: (t: EvalTrajectorySummary) => t.idx,
    },
    {
      key: 'example_id',
      label: 'Example',
      render: (t: EvalTrajectorySummary) => (
        <span className="mono" style={{ fontSize: '0.6875rem' }}>
          {t.example_id ? t.example_id.slice(0, 12) : '-'}
        </span>
      ),
    },
    {
      key: 'reward',
      label: 'Reward',
      render: (t: EvalTrajectorySummary) => (
        <span style={{ color: t.reward > 0 ? 'var(--success)' : 'var(--error)', fontWeight: 600 }}>
          {t.reward.toFixed(1)}
        </span>
      ),
      sortValue: (t: EvalTrajectorySummary) => t.reward,
    },
    {
      key: 'turns',
      label: 'Turns',
      render: (t: EvalTrajectorySummary) => <span className="mono">{t.num_turns}</span>,
      sortValue: (t: EvalTrajectorySummary) => t.num_turns,
    },
    {
      key: 'time',
      label: 'Time',
      render: (t: EvalTrajectorySummary) => <span className="mono">{t.time_seconds.toFixed(1)}s</span>,
      sortValue: (t: EvalTrajectorySummary) => t.time_seconds,
    },
    {
      key: 'status',
      label: 'Status',
      render: (t: EvalTrajectorySummary) => {
        if (t.error) return <ResultTag variant="error">Error</ResultTag>;
        if (t.reward > 0) return <ResultTag variant="correct">Correct</ResultTag>;
        return <span className="tag">Wrong</span>;
      },
    },
  ];

  // Client-side search filtering
  const searched = search
    ? trajectories.filter((t) => {
        const q = search.toLowerCase();
        return (
          (t.example_id && t.example_id.toLowerCase().includes(q)) ||
          (t.error && t.error.toLowerCase().includes(q)) ||
          String(t.idx).includes(q)
        );
      })
    : trajectories;

  const correctCount = searched.filter((t) => t.reward > 0).length;
  const errorCount = searched.filter((t) => t.error).length;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span>/</span>
        <span>{evalRunId}</span>
      </div>

      <h2 className="page-title">{evalRunId}</h2>
      <div className="page-subtitle">
        {String(metadata.model_name ?? 'Unknown model')} · {String(metadata.checkpoint_name ?? 'No checkpoint')}
      </div>

      {/* Benchmark result cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.625rem', marginBottom: '1rem' }}>
        {Object.entries(run.results).map(([name, result]) => (
          <div
            key={name}
            className="card"
            style={{
              cursor: 'pointer',
              borderColor: selectedBenchmark === name ? 'var(--accent)' : undefined,
              padding: '0.75rem',
            }}
            onClick={() => setSelectedBenchmark(name)}
          >
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              {name}
            </div>
            <div style={{ fontSize: '1.375rem', fontWeight: 700, color: scoreColor(result.score), fontVariantNumeric: 'tabular-nums' }}>
              {(result.score * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: '0.125rem' }}>
              {result.num_correct}/{result.num_examples}
              {result.num_errors > 0 && <span style={{ color: 'var(--error)' }}> · {result.num_errors} err</span>}
              {' · '}{result.time_seconds.toFixed(0)}s
            </div>
          </div>
        ))}
      </div>

      {/* Trajectory list */}
      {selectedBenchmark && (
        <>
          <div className="filters-bar">
            <div className="filter-group">
              <span className="filter-label">Show</span>
              <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
                <option value="all">All ({searched.length})</option>
                <option value="correct">Correct ({correctCount})</option>
                <option value="incorrect">Incorrect</option>
                <option value="errors">Errors ({errorCount})</option>
              </select>
            </div>
            <input
              type="text"
              placeholder="Search by example ID, error..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                padding: '0.3125rem 0.625rem',
                borderRadius: '6px',
                border: '1px solid var(--border)',
                background: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                fontSize: '0.8125rem',
                width: '220px',
              }}
            />
          </div>

          <div className="card" style={{ padding: 0, overflow: 'auto' }}>
            <SortableTable
              columns={trajColumns}
              data={searched}
              rowKey={(t) => String(t.idx)}
              onRowClick={(t) => navigate(`/eval/${evalRunId}/${selectedBenchmark}/${t.idx}`)}
            />
          </div>
        </>
      )}
    </div>
  );
}
