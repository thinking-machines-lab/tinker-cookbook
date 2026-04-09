import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { ConversationRenderer } from '../components/ConversationRenderer';
import { MetaField } from '../utils/shared';
import type { EvalTrajectoryDetail } from '../api/types';

export function EvalTrajectoryPage() {
  const { evalRunId, benchmark, idx } = useParams<{
    evalRunId: string;
    benchmark: string;
    idx: string;
  }>();
  const navigate = useNavigate();
  const [traj, setTraj] = useState<EvalTrajectoryDetail | null>(null);
  const [totalTrajs, setTotalTrajs] = useState(0);
  const [loading, setLoading] = useState(true);

  const idxNum = Number(idx);

  useEffect(() => {
    if (!evalRunId || !benchmark || !idx) return;
    setLoading(true);
    Promise.all([
      api.getEvalTrajectoryDetail(evalRunId, benchmark, idxNum),
      api.getEvalTrajectories(evalRunId, benchmark, {}).then((r) => r.total).catch(() => 0),
    ])
      .then(([trajData, total]) => {
        setTraj(trajData);
        setTotalTrajs(total);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [evalRunId, benchmark, idx]);

  if (loading) return <div className="loading">Loading trajectory...</div>;
  if (!traj) return <div className="empty-state">Trajectory not found</div>;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="breadcrumb" style={{ marginBottom: 0 }}>
          <Link to="/">Dashboard</Link>
          <span>/</span>
          <Link to={`/eval/${evalRunId}`}>{evalRunId}</Link>
          <span>/</span>
          <Link to={`/eval/${evalRunId}`}>{benchmark}</Link>
          <span>/</span>
          <span>#{idx}</span>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
          {totalTrajs > 0 && (
            <span className="text-muted" style={{ fontSize: '0.6875rem', marginRight: '0.25rem' }}>
              {idxNum + 1} of {totalTrajs}
            </span>
          )}
          <button
            className="tab"
            onClick={() => idxNum > 0 && navigate(`/eval/${evalRunId}/${benchmark}/${idxNum - 1}`)}
            disabled={idxNum <= 0}
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', opacity: idxNum > 0 ? 1 : 0.3, borderBottom: 'none' }}
          >
            Prev
          </button>
          <button
            className="tab"
            onClick={() => idxNum < totalTrajs - 1 && navigate(`/eval/${evalRunId}/${benchmark}/${idxNum + 1}`)}
            disabled={idxNum >= totalTrajs - 1}
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', opacity: idxNum < totalTrajs - 1 ? 1 : 0.3, borderBottom: 'none' }}
          >
            Next
          </button>
        </div>
      </div>

      {/* Header */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <MetaField label="Benchmark" value={traj.benchmark} />
          <MetaField label="Index" value={String(traj.idx)} />
          {traj.example_id && (
            <MetaField label="Example ID" value={traj.example_id.slice(0, 16)} />
          )}
          <MetaField
            label="Reward"
            value={traj.reward.toFixed(2)}
            color={traj.reward > 0 ? 'var(--success)' : 'var(--error)'}
          />
          <MetaField label="Turns" value={String(traj.turns.length)} />
          <MetaField label="Time" value={`${traj.time_seconds.toFixed(1)}s`} />
          {traj.error && <MetaField label="Error" value={traj.error} color="var(--error)" />}
        </div>

        {/* Logs */}
        {Object.keys(traj.logs).length > 0 && (
          <div style={{ marginTop: '12px', padding: '8px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
              LOGS
            </div>
            <div style={{ fontSize: '0.8rem' }}>
              {Object.entries(traj.logs).map(([k, v]) => (
                <div key={k} style={{ marginBottom: '2px' }}>
                  <span className="text-muted">{k}:</span>{' '}
                  <span className="mono">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Conversation */}
      <div className="card">
        <div className="card-header">Conversation ({traj.turns.length} turns)</div>
        <ConversationRenderer messages={traj.turns} showTokenCounts />
      </div>
    </div>
  );
}

