import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { CheckpointRecord } from '../api/types';

export function CheckpointsTabPanel({ runId }: { runId: string }) {
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getCheckpoints(runId)
      .then(setCheckpoints)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading checkpoints...</div>;
  if (checkpoints.length === 0) return <div className="empty-state">No checkpoints saved</div>;

  return (
    <div className="card" style={{ overflow: 'auto' }}>
      <div className="card-title" style={{ marginBottom: '0.5rem' }}>
        Checkpoints ({checkpoints.length})
      </div>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Step</th>
            <th>Final</th>
            <th>State Path</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map((ckpt, i) => (
            <tr key={i} style={{ cursor: 'default' }}>
              <td className="mono" style={{ fontWeight: 600 }}>{ckpt.name}</td>
              <td className="mono">{ckpt.batch ?? ckpt.loop_state?.batch ?? '-'}</td>
              <td>{ckpt.final ? 'Yes' : ''}</td>
              <td className="mono" style={{ fontSize: '0.6875rem', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {ckpt.state_path ?? '-'}
              </td>
              <td>
                {ckpt.sampler_path && (
                  <Link
                    to={`/runs/${runId}/chat?checkpoint=${ckpt.name}`}
                    onClick={(e) => e.stopPropagation()}
                    style={{ fontSize: '0.75rem', fontWeight: 600 }}
                  >
                    Chat →
                  </Link>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
