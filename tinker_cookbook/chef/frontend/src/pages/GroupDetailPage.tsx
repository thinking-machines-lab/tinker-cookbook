/** Group comparison view — side-by-side trajectories within a GRPO group. */

import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import { ConversationRenderer } from '../components/ConversationRenderer';
import { TOOLTIP_CONTENT_STYLE } from '../components/charts';
import { SERIES_COLORS } from '../theme/colors';
import { MetaField, ResultTag, rewardColor } from '../utils/shared';
import { extractConversation } from '../utils/conversation';
import type { RolloutDetail } from '../api/types';

export function GroupDetailPage() {
  const { runId, iteration, groupIdx } = useParams<{
    runId: string;
    iteration: string;
    groupIdx: string;
  }>();
  const [rollouts, setRollouts] = useState<RolloutDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !iteration || !groupIdx) return;
    const iter = Number(iteration);
    const gIdx = Number(groupIdx);

    setLoading(true);
    api.getGroupRollouts(runId, iter, gIdx)
      .then((resp) => {
        setRollouts(resp.rollouts.sort((a, b) => a.traj_idx - b.traj_idx));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId, iteration, groupIdx]);

  if (loading) return <div className="loading">Loading group...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (rollouts.length === 0) return <div className="empty-state">No trajectories found for group {groupIdx}</div>;

  // Reward comparison data
  const rewardData = rollouts.map((r) => ({
    name: `Traj ${r.traj_idx}`,
    reward: r.total_reward,
    traj_idx: r.traj_idx,
  }));

  const minReward = Math.min(...rewardData.map((d) => d.reward));
  const maxReward = Math.max(...rewardData.map((d) => d.reward));
  const rewardSpread = maxReward - minReward;

  return (
    <div>
      {/* Breadcrumb */}
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span>/</span>
        <Link to={`/runs/${runId}?tab=rollouts`}>{runId}</Link>
        <span>/</span>
        <Link to={`/runs/${runId}?tab=rollouts`}>Iter {iteration}</Link>
        <span>/</span>
        <span>Group {groupIdx}</span>
      </div>

      {/* Group summary */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <MetaField label="Trajectories" value={String(rollouts.length)} />
          <MetaField
            label="Reward Range"
            value={`${minReward.toFixed(3)} — ${maxReward.toFixed(3)}`}
          />
          <MetaField label="Spread" value={rewardSpread.toFixed(3)} />
          {rollouts[0].tags.length > 0 && (
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', marginBottom: '2px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Tags</div>
              <div>{rollouts[0].tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Reward comparison bar chart */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Reward Comparison</div>
        <ResponsiveContainer width="100%" height={Math.max(120, rollouts.length * 30 + 40)}>
          <BarChart data={rewardData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
            <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="name" stroke="var(--text-muted)" tick={{ fontSize: 10 }} width={55} />
            <Tooltip contentStyle={TOOLTIP_CONTENT_STYLE} formatter={(value: number) => [value.toFixed(3), 'Reward']} />
            <Bar dataKey="reward" radius={[0, 4, 4, 0]}>
              {rewardData.map((d, i) => (
                <Cell key={i} fill={SERIES_COLORS[i % SERIES_COLORS.length]} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Side-by-side trajectories */}
      <div className="card-title" style={{ marginBottom: '0.5rem' }}>Trajectories</div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: rollouts.length <= 2 ? `repeat(${rollouts.length}, 1fr)` : 'repeat(2, 1fr)',
        gap: '0.5rem',
      }}>
        {rollouts.map((rollout) => (
          <TrajectoryColumn key={rollout.traj_idx} rollout={rollout} runId={runId!} iteration={iteration!} />
        ))}
      </div>
    </div>
  );
}

function TrajectoryColumn({ rollout, runId, iteration }: { rollout: RolloutDetail; runId: string; iteration: string }) {
  const messages = extractConversation(rollout);

  return (
    <div className="card" style={{ padding: '0.5rem', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <Link
          to={`/runs/${runId}/iterations/${iteration}/rollouts/${rollout.group_idx}/${rollout.traj_idx}`}
          style={{ fontWeight: 700, fontSize: '0.875rem' }}
        >
          Traj {rollout.traj_idx}
        </Link>
        <span className="mono" style={{ fontWeight: 600, color: rewardColor(rollout.total_reward) }}>
          r={rollout.total_reward.toFixed(3)}
        </span>
      </div>

      {/* Quick stats */}
      <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
        <span>{rollout.steps.length} step{rollout.steps.length !== 1 ? 's' : ''}</span>
        <span>{rollout.steps.reduce((s, st) => s + st.ac_len, 0)} tok</span>
        {rollout.status && rollout.status !== 'ok' && (
          <ResultTag variant={rollout.status}>{rollout.status}</ResultTag>
        )}
      </div>

      {/* Conversation */}
      {messages.length > 0 ? (
        <ConversationRenderer messages={messages} />
      ) : (
        <div className="text-muted" style={{ fontSize: '0.75rem', fontStyle: 'italic' }}>
          No conversation data
        </div>
      )}
    </div>
  );
}
