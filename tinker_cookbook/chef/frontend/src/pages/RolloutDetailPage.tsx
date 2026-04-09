import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { ConversationRenderer } from '../components/ConversationRenderer';
import { MetaField, rewardColor } from '../utils/shared';
import type { RolloutDetail, RolloutSummary, LogtreeNode, LogtreeResponse, ConversationMessage } from '../api/types';

export function RolloutDetailPage() {
  const { runId, iteration, groupIdx, trajIdx } = useParams<{
    runId: string;
    iteration: string;
    groupIdx: string;
    trajIdx: string;
  }>();
  const navigate = useNavigate();
  const [rollout, setRollout] = useState<RolloutDetail | null>(null);
  const [logtree, setLogtree] = useState<LogtreeResponse | null>(null);
  const [siblings, setSiblings] = useState<RolloutSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !iteration || !groupIdx || !trajIdx) return;
    const iter = Number(iteration);
    const gIdx = Number(groupIdx);
    const tIdx = Number(trajIdx);

    setLoading(true);
    Promise.all([
      api.getRolloutDetail(runId, iter, gIdx, tIdx),
      api.getLogtree(runId, iter).catch(() => null),
      api.getRollouts(runId, iter),
    ])
      .then(([rolloutData, logtreeData, rolloutsResp]) => {
        setRollout(rolloutData);
        setLogtree(logtreeData);
        setSiblings(rolloutsResp.rollouts);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId, iteration, groupIdx, trajIdx]);

  if (loading) return <div className="loading">Loading rollout...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (!rollout || !runId) return <div className="empty-state">Rollout not found</div>;

  // Find current position among siblings for prev/next
  const currentIdx = siblings.findIndex(
    (s) => s.group_idx === rollout.group_idx && s.traj_idx === rollout.traj_idx
  );
  const prevSibling = currentIdx > 0 ? siblings[currentIdx - 1] : null;
  const nextSibling = currentIdx >= 0 && currentIdx < siblings.length - 1 ? siblings[currentIdx + 1] : null;

  const navTo = (s: RolloutSummary) =>
    `/runs/${runId}/iterations/${iteration}/rollouts/${s.group_idx}/${s.traj_idx}`;

  return (
    <div>
      {/* Breadcrumb + prev/next */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="breadcrumb" style={{ marginBottom: 0 }}>
          <Link to="/">Dashboard</Link>
          <span>/</span>
          <Link to={`/runs/${runId}?tab=rollouts`}>{runId}</Link>
          <span>/</span>
          <Link to={`/runs/${runId}?tab=rollouts`}>Iter {iteration}</Link>
          <span>/</span>
          <span>({groupIdx}, {trajIdx})</span>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
          {currentIdx >= 0 && (
            <span className="text-muted" style={{ fontSize: '0.6875rem', marginRight: '0.25rem' }}>
              {currentIdx + 1} of {siblings.length}
            </span>
          )}
          <button
            className="tab"
            onClick={() => prevSibling && navigate(navTo(prevSibling))}
            disabled={!prevSibling}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              opacity: prevSibling ? 1 : 0.3,
              borderBottom: 'none',
            }}
          >
            Prev
          </button>
          <button
            className="tab"
            onClick={() => nextSibling && navigate(navTo(nextSibling))}
            disabled={!nextSibling}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              opacity: nextSibling ? 1 : 0.3,
              borderBottom: 'none',
            }}
          >
            Next
          </button>
        </div>
      </div>

      {/* Header with rollout metadata */}
      {/* Error/timeout banner */}
      {rollout.status === 'error' && (
        <div style={{ padding: '0.75rem 1rem', marginBottom: '0.75rem', borderRadius: '8px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <div style={{ fontWeight: 600, color: 'var(--error)', marginBottom: '0.25rem' }}>
            Error: {rollout.error_type ?? 'Unknown'}
          </div>
          {rollout.error_message && (
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>{rollout.error_message}</div>
          )}
        </div>
      )}
      {rollout.status === 'timeout' && (
        <div style={{ padding: '0.75rem 1rem', marginBottom: '0.75rem', borderRadius: '8px', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)' }}>
          <div style={{ fontWeight: 600, color: 'var(--warning)' }}>
            Truncated — model hit max_tokens
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <MetaField label="Iteration" value={String(rollout.iteration)} />
          <MetaField label="Group" value={String(rollout.group_idx)} />
          <MetaField label="Trajectory" value={String(rollout.traj_idx)} />
          <MetaField
            label="Total Reward"
            value={rollout.total_reward.toFixed(3)}
            color={rewardColor(rollout.total_reward)}
          />
          <MetaField
            label="Final Reward"
            value={rollout.final_reward.toFixed(3)}
            color={rewardColor(rollout.final_reward)}
          />
          <MetaField label="Steps" value={String(rollout.steps.length)} />
          <MetaField label="Tokens Generated" value={`${rollout.steps.reduce((sum, s) => sum + s.ac_len, 0)} tok`} />
          <MetaField label="Context" value={`${rollout.final_ob_len} tok`} />
          {rollout.sampling_client_step != null && (
            <MetaField label="Sampled At" value={`step ${rollout.sampling_client_step}`} />
          )}
          {rollout.tags.length > 0 && (
            <div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Tags</div>
              <div>{rollout.tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Step timeline */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Step Timeline</div>
        <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
          {rollout.steps.map((step) => (
            <div
              key={step.step_idx}
              title={`Step ${step.step_idx}: reward=${step.reward}, ob=${step.ob_len}, ac=${step.ac_len}`}
              style={{
                flex: '0 0 auto',
                padding: '3px 6px',
                borderRadius: '4px',
                background: step.episode_done ? 'rgba(34, 197, 94, 0.15)' : 'var(--bg-tertiary)',
                border: `1px solid ${step.episode_done ? 'var(--success)' : 'var(--border)'}`,
                fontSize: '0.625rem',
                textAlign: 'center',
                minWidth: '42px',
              }}
            >
              <div className="mono" style={{ fontWeight: 600 }}>{step.step_idx}</div>
              <div className="mono" style={{ color: rewardColor(step.reward) }}>
                r={step.reward.toFixed(2)}
              </div>
              <div className="text-muted" style={{ fontSize: '0.5625rem' }}>
                {step.ob_len}+{step.ac_len}t
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Conversation — extract only this trajectory's messages from the logtree */}
      {(() => {
        const messages = logtree
          ? extractTrajectoryMessages(logtree.root, rollout.group_idx, rollout.traj_idx)
          : [];
        if (messages.length > 0) {
          return (
            <div className="card" style={{ marginBottom: '0.75rem' }}>
              <div className="card-title" style={{ marginBottom: '0.5rem' }}>
                Conversation ({messages.length} messages)
              </div>
              {messages.length > 1 && (
                <>
                  <div style={{ fontSize: '0.625rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '0.25rem 0' }}>
                    Prompt
                  </div>
                  <ConversationRenderer messages={messages.slice(0, -1)} showTokenCounts />
                  <div style={{ borderTop: '1px dashed var(--border)', margin: '0.25rem 0', position: 'relative' }}>
                    <span style={{ position: 'absolute', top: '-0.5rem', left: '0.5rem', background: 'var(--bg-surface)', padding: '0 0.5rem', fontSize: '0.625rem', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Model Generation
                    </span>
                  </div>
                </>
              )}
              <ConversationRenderer messages={messages.slice(-1)} showTokenCounts />
              {/* Conversation summary */}
              <div style={{ marginTop: '0.75rem', padding: '0.5rem 0.75rem', background: 'var(--bg-tertiary)', borderRadius: '6px', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem' }}>
                <span>{rollout.steps.length} turn{rollout.steps.length !== 1 ? 's' : ''}</span>
                <span>{rollout.steps.reduce((s, st) => s + st.ac_len, 0)} tokens generated</span>
                <span>{rollout.final_ob_len} tok final context</span>
              </div>
            </div>
          );
        }
        return (
          <div className="card" style={{ marginBottom: '0.75rem', padding: '1rem' }}>
            <div className="card-title" style={{ marginBottom: '0.25rem' }}>Conversation</div>
            <div className="text-muted" style={{ fontSize: '0.8rem' }}>
              Conversation not available for this trajectory. Only the first few groups
              per iteration have their conversations logged (controlled by{' '}
              <code className="mono" style={{ fontSize: '0.75rem' }}>num_groups_to_log</code>
              {' '}in the training config). Try browsing groups 0-1 which are typically logged.
            </div>
          </div>
        );
      })()}

      {/* Step details table */}
      <div className="card">
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Step Details</div>
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Obs</th>
              <th>Action</th>
              <th>Reward</th>
              <th>Done</th>
              <th>Metrics</th>
            </tr>
          </thead>
          <tbody>
            {rollout.steps.map((step) => (
              <tr key={step.step_idx} style={{ cursor: 'default' }}>
                <td className="mono">{step.step_idx}</td>
                <td className="mono">{step.ob_len}</td>
                <td className="mono">{step.ac_len}</td>
                <td>
                  <span style={{ color: rewardColor(step.reward), fontWeight: 600 }}>
                    {step.reward.toFixed(3)}
                  </span>
                </td>
                <td>{step.episode_done ? 'Yes' : ''}</td>
                <td style={{ fontSize: '0.6875rem' }}>
                  {Object.entries(step.metrics)
                    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
                    .join(', ') || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type ConvMessage = ConversationMessage;

/** Type guard for logtree nodes (children can be strings or nodes). */
function isNode(child: string | LogtreeNode): child is LogtreeNode {
  return typeof child !== 'string';
}

/** Type guard for logtree conversation data. */
function isConversationData(data: Record<string, unknown> | undefined): data is { type: 'conversation'; messages: ConvMessage[] } {
  return data != null && data.type === 'conversation' && Array.isArray(data.messages);
}

/** Extract conversation messages for a specific trajectory from the logtree.
 *
 * The logtree has "Group Rollout" sections (one per group), each containing
 * trajectory data. We navigate to the Nth group section, then extract
 * messages from the correct trajectory within that group.
 * Falls back to extracting all messages if the structure doesn't match.
 */
function extractTrajectoryMessages(
  root: LogtreeNode,
  groupIdx: number,
  trajIdx: number,
): ConvMessage[] {
  const groupSections = (root.children ?? []).filter(isNode).filter((node) => {
    if (node.tag !== 'section') return false;
    return (node.children ?? []).filter(isNode).some((c) =>
      c.tag === 'h2' && (c.children ?? []).some((t) => typeof t === 'string' && t.includes('Group Rollout'))
    );
  });

  if (groupIdx < groupSections.length) {
    const groupNode = groupSections[groupIdx];
    // Extract only "Policy Response" conversations (not "Prompt" which duplicates the user message).
    // Each trajectory has sections: Prompt, Policy Response, Reward, Trajectory N Episode.
    // We want the Policy Response conversation which has the complete user+assistant exchange.
    const responseConvs = extractResponseConversations(groupNode);

    // Each trajectory produces one response conversation.
    // trajIdx maps directly to the Nth response conversation.
    if (trajIdx < responseConvs.length) {
      return responseConvs[trajIdx];
    }
  }

  return [];
}

/** Extract per-trajectory conversations from a group's logtree.
 *
 * The logtree has alternating Prompt/Response/Reward sections per trajectory:
 *   Prompt → conversation with few-shot + user question
 *   Policy Response → conversation with just the model's answer
 *   Reward → not a conversation
 *
 * We pair Prompt + Response conversations for each trajectory.
 */
function extractResponseConversations(groupNode: LogtreeNode): ConvMessage[][] {
  const prompts: ConvMessage[][] = [];
  const responses: ConvMessage[][] = [];

  function walk(node: LogtreeNode) {
    if (node.tag === 'section') {
      const headerText = (node.children ?? []).filter(isNode)
        .filter((c) => c.tag === 'h3')
        .flatMap((c) => (c.children ?? []).filter((t): t is string => typeof t === 'string'))
        .join('');

      if (headerText.includes('Prompt')) {
        const convs = extractAllConversations(node);
        if (convs.length > 0) prompts.push(convs.flat());
        return;
      }
      if (headerText.includes('Policy Response')) {
        const convs = extractAllConversations(node);
        if (convs.length > 0) responses.push(convs.flat());
        return;
      }
    }
    for (const child of (node.children ?? []).filter(isNode)) {
      walk(child);
    }
  }
  walk(groupNode);

  // Pair prompts with responses — each trajectory has one prompt + one response
  const results: ConvMessage[][] = [];
  for (let i = 0; i < Math.max(prompts.length, responses.length); i++) {
    const combined: ConvMessage[] = [];
    if (i < prompts.length) combined.push(...prompts[i]);
    if (i < responses.length) combined.push(...responses[i]);
    if (combined.length > 0) results.push(combined);
  }
  return results;
}

/** Extract all conversation messages from a subtree. */
function extractAllConversations(node: LogtreeNode): ConvMessage[][] {
  const convs: ConvMessage[][] = [];
  if (isConversationData(node.data)) {
    if (node.data.messages.length > 0) convs.push(node.data.messages);
  }
  for (const child of (node.children ?? []).filter(isNode)) {
    convs.push(...extractAllConversations(child));
  }
  return convs;
}

