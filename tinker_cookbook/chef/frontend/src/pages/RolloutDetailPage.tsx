import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { ConversationRenderer } from '../components/ConversationRenderer';
import { MetaField, rewardColor } from '../utils/shared';
import type { RolloutDetail, RolloutSummary, RolloutStep, LogtreeNode, LogtreeResponse, ConversationMessage } from '../api/types';

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
    // Always fetch rollout detail + siblings; only fetch logtree if needed (v1/v2 data)
    Promise.all([
      api.getRolloutDetail(runId, iter, gIdx, tIdx),
      api.getRollouts(runId, iter),
    ])
      .then(([rolloutData, rolloutsResp]) => {
        setRollout(rolloutData);
        setSiblings(rolloutsResp.rollouts);
        // Only fetch logtree for legacy data without embedded conversation
        if (!rolloutData.conversation || rolloutData.conversation.length === 0) {
          api.getLogtree(runId, iter).then(setLogtree).catch(() => null);
        } else {
          setLogtree(null);
        }
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

  // Build per-step conversation messages
  const stepMessages = getStepMessages(rollout, logtree);

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
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', opacity: prevSibling ? 1 : 0.3, borderBottom: 'none' }}
          >
            Prev
          </button>
          <button
            className="tab"
            onClick={() => nextSibling && navigate(navTo(nextSibling))}
            disabled={!nextSibling}
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', opacity: nextSibling ? 1 : 0.3, borderBottom: 'none' }}
          >
            Next
          </button>
        </div>
      </div>

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

      {/* Header with rollout metadata */}
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
          {rollout.model_name && <MetaField label="Model" value={rollout.model_name} />}
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

      {/* Step timeline overview */}
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

      {/* Interleaved step + conversation timeline */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {rollout.steps.map((step) => (
          <StepCard
            key={step.step_idx}
            step={step}
            messages={stepMessages.get(step.step_idx) ?? []}
            isLast={step.step_idx === rollout.steps.length - 1}
          />
        ))}
      </div>

      {/* Conversation not available hint */}
      {stepMessages.size === 0 && (
        <div className="card" style={{ marginTop: '0.5rem', padding: '1rem' }}>
          <div className="card-title" style={{ marginBottom: '0.25rem' }}>Conversation</div>
          <div className="text-muted" style={{ fontSize: '0.8rem' }}>
            Conversation not available for this trajectory. Only the first few groups
            per iteration have their conversations logged (controlled by{' '}
            <code className="mono" style={{ fontSize: '0.75rem' }}>num_groups_to_log</code>
            {' '}in the training config). Try browsing groups 0-1 which are typically logged.
          </div>
        </div>
      )}
    </div>
  );
}

// ─── StepCard ────────────────────────────────────────────────────────────────

function StepCard({ step, messages, isLast }: { step: RolloutStep; messages: ConversationMessage[]; isLast: boolean }) {
  const [expanded, setExpanded] = useState(true);
  const hasMessages = messages.length > 0;
  const userLogs = getFilteredLogs(step.logs);
  const hasMetrics = Object.keys(step.metrics).length > 0;
  const hasLogs = userLogs.length > 0;

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      {/* Step header bar */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap',
          padding: '0.5rem 0.75rem',
          background: step.episode_done ? 'rgba(34, 197, 94, 0.06)' : 'var(--bg-tertiary)',
          borderBottom: expanded ? '1px solid var(--border)' : 'none',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>
          {expanded ? '\u25bc' : '\u25b6'}
        </span>
        <span className="mono" style={{ fontWeight: 700, fontSize: '0.8125rem' }}>
          Step {step.step_idx}
        </span>
        <span className="mono" style={{ fontWeight: 600, color: rewardColor(step.reward) }}>
          r={step.reward.toFixed(3)}
        </span>
        <span className="mono text-muted" style={{ fontSize: '0.6875rem' }}>
          {step.ob_len} + {step.ac_len} tok
        </span>
        {step.episode_done && (
          <span className="badge" style={{ background: 'rgba(34, 197, 94, 0.15)', color: 'var(--success)', fontSize: '0.5625rem' }}>
            DONE
          </span>
        )}
        {/* Inline metrics summary */}
        {hasMetrics && (
          <span className="text-muted" style={{ fontSize: '0.625rem', marginLeft: 'auto' }}>
            {Object.entries(step.metrics).slice(0, 4).map(([k, v]) =>
              `${k}=${typeof v === 'number' ? v.toFixed(2) : v}`
            ).join(', ')}
            {Object.keys(step.metrics).length > 4 && ` +${Object.keys(step.metrics).length - 4} more`}
          </span>
        )}
      </div>

      {expanded && (
        <div style={{ padding: '0.5rem 0.75rem' }}>
          {/* Conversation messages for this step */}
          {hasMessages && (
            <div style={{ marginBottom: hasMetrics || hasLogs ? '0.5rem' : 0 }}>
              {isLast && messages.length > 1 && (
                <div style={{ fontSize: '0.5625rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', paddingBottom: '0.25rem' }}>
                  Prompt
                </div>
              )}
              {isLast && messages.length > 1 ? (
                <>
                  <ConversationRenderer messages={messages.slice(0, -1)} showTokenCounts />
                  <div style={{ borderTop: '1px dashed var(--border)', margin: '0.25rem 0', position: 'relative' }}>
                    <span style={{ position: 'absolute', top: '-0.5rem', left: '0.5rem', background: 'var(--bg-surface)', padding: '0 0.5rem', fontSize: '0.5625rem', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Model Generation
                    </span>
                  </div>
                  <ConversationRenderer messages={messages.slice(-1)} showTokenCounts />
                </>
              ) : (
                <ConversationRenderer messages={messages} showTokenCounts />
              )}
            </div>
          )}

          {/* Step metrics */}
          {hasMetrics && (
            <div style={{ padding: '0.375rem 0.5rem', background: 'var(--bg-tertiary)', borderRadius: '6px', marginBottom: hasLogs ? '0.375rem' : 0 }}>
              <div style={{ fontSize: '0.5625rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.125rem' }}>
                Metrics
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.75rem' }}>
                {Object.entries(step.metrics).map(([k, v]) => (
                  <span key={k} style={{ fontSize: '0.6875rem' }}>
                    <span className="text-muted">{k}:</span>{' '}
                    <span className="mono" style={{ fontWeight: 600 }}>{typeof v === 'number' ? v.toFixed(3) : v}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Step logs (user-facing, excluding framework _-prefixed keys) */}
          {hasLogs && (
            <div style={{ padding: '0.375rem 0.5rem', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.5625rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.125rem' }}>
                Logs
              </div>
              {userLogs.map(([key, value]) => (
                <div key={key} style={{ fontSize: '0.6875rem', marginBottom: '2px' }}>
                  <span className="text-muted">{key}:</span>{' '}
                  {typeof value === 'string' && value.length > 200 ? (
                    <details style={{ display: 'inline' }}>
                      <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>{value.slice(0, 100)}...</summary>
                      <pre className="mono" style={{ fontSize: '0.625rem', whiteSpace: 'pre-wrap', marginTop: '0.25rem', padding: '0.25rem', background: 'var(--bg-primary)', borderRadius: '4px' }}>{value}</pre>
                    </details>
                  ) : (
                    <span className="mono">{String(value)}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Filter step.logs to only user-facing keys (exclude _-prefixed framework keys). */
function getFilteredLogs(logs: Record<string, unknown>): [string, unknown][] {
  return Object.entries(logs).filter(([k]) => !k.startsWith('_'));
}

/** Build a map of step_idx → ConversationMessage[] from available data.
 *
 * Priority:
 * 1. step.logs._conversation (schema v3 — per-step messages)
 * 2. rollout.conversation (schema v3 — flat list, assigned to last step for single-turn)
 * 3. Legacy logtree extraction (schema v1/v2)
 */
function getStepMessages(
  rollout: RolloutDetail,
  logtree: LogtreeResponse | null,
): Map<number, ConversationMessage[]> {
  const map = new Map<number, ConversationMessage[]>();

  // Priority 1: per-step _conversation from logs
  let hasPerStepConv = false;
  for (const step of rollout.steps) {
    const msgs = (step.logs as Record<string, unknown>)?._conversation;
    if (Array.isArray(msgs) && msgs.length > 0) {
      map.set(step.step_idx, msgs as ConversationMessage[]);
      hasPerStepConv = true;
    }
  }
  if (hasPerStepConv) return map;

  // Priority 2: top-level conversation field (flat list)
  if (rollout.conversation && rollout.conversation.length > 0) {
    if (rollout.steps.length === 1) {
      // Single-turn: all messages belong to step 0
      map.set(0, rollout.conversation);
    } else {
      // Multi-turn with flat conversation but no per-step breakdown:
      // Assign all to step 0 (best effort)
      map.set(0, rollout.conversation);
    }
    return map;
  }

  // Priority 3: legacy logtree extraction
  if (logtree) {
    const legacyMsgs = extractTrajectoryMessages(logtree.root, rollout.group_idx, rollout.traj_idx);
    if (legacyMsgs.length > 0) {
      // Legacy: assign all messages to step 0
      map.set(0, legacyMsgs);
    }
  }

  return map;
}

// ─── Legacy logtree extraction (for schema v1/v2 backward compat) ────────────

function isNode(child: string | LogtreeNode): child is LogtreeNode {
  return typeof child !== 'string';
}

function isConversationData(data: Record<string, unknown> | undefined): data is { type: 'conversation'; messages: ConversationMessage[] } {
  return data != null && data.type === 'conversation' && Array.isArray(data.messages);
}

function extractTrajectoryMessages(
  root: LogtreeNode,
  groupIdx: number,
  trajIdx: number,
): ConversationMessage[] {
  const groupSections = (root.children ?? []).filter(isNode).filter((node) => {
    if (node.tag !== 'section') return false;
    return (node.children ?? []).filter(isNode).some((c) =>
      c.tag === 'h2' && (c.children ?? []).some((t) => typeof t === 'string' && t.includes('Group Rollout'))
    );
  });

  if (groupIdx < groupSections.length) {
    const groupNode = groupSections[groupIdx];
    const responseConvs = extractResponseConversations(groupNode);
    if (trajIdx < responseConvs.length) {
      return responseConvs[trajIdx];
    }
  }

  return [];
}

function extractResponseConversations(groupNode: LogtreeNode): ConversationMessage[][] {
  const prompts: ConversationMessage[][] = [];
  const responses: ConversationMessage[][] = [];

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

  const results: ConversationMessage[][] = [];
  for (let i = 0; i < Math.max(prompts.length, responses.length); i++) {
    const combined: ConversationMessage[] = [];
    if (i < prompts.length) combined.push(...prompts[i]);
    if (i < responses.length) combined.push(...responses[i]);
    if (combined.length > 0) results.push(combined);
  }
  return results;
}

function extractAllConversations(node: LogtreeNode): ConversationMessage[][] {
  const convs: ConversationMessage[][] = [];
  if (isConversationData(node.data)) {
    if (node.data.messages.length > 0) convs.push(node.data.messages);
  }
  for (const child of (node.children ?? []).filter(isNode)) {
    convs.push(...extractAllConversations(child));
  }
  return convs;
}
