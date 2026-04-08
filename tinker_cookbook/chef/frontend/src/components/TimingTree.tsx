import { useState } from 'react';
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
  do_group_rollout_and_filter_constant_reward: '#a78bfa',
  trajectory_group_worker: '#a78bfa',
  policy_sample: '#e5a11c',
  env_step: '#6aad7a',
  env_initial_observation: '#6aad7a',
  compute_group_rewards: '#64748b',
  train_step: '#e85850',
  do_train_step_and_get_sampling_client: '#e85850',
  assemble_training_data: '#ec4899',
  prepare_minibatch: '#ec4899',
  compute_kl_sample_train: '#06b6d4',
  save_checkpoint: '#f97316',
  save_checkpoint_and_get_sampling_client: '#f97316',
  compute_full_batch_metrics_and_get_sampling_client: '#14b8a6',
};

function getColor(name: string): string {
  return SPAN_COLORS[name] ?? '#64748b';
}

function shortName(name: string): string {
  // Shorten long span names for display
  return name
    .replace('do_group_rollout_and_filter_constant_reward', 'group_rollout')
    .replace('do_train_step_and_get_sampling_client', 'train_step')
    .replace('save_checkpoint_and_get_sampling_client', 'save_checkpoint')
    .replace('compute_full_batch_metrics_and_get_sampling_client', 'compute_metrics')
    .replace('trajectory_group_worker_task', 'worker')
    .replace('trajectory_group_worker', 'worker');
}

export function TimingTree({ root, totalDuration, runId, step }: Props) {
  if (!root) return <div className="empty-state">No timing data for this step</div>;

  return (
    <div>
      {/* Top-level summary bar */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        {root.children.map((child, i) => {
          const pct = totalDuration > 0 ? (child.duration / totalDuration) * 100 : 0;
          return (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontFamily: 'var(--font-mono)', fontSize: '0.6875rem' }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: getColor(child.name), flexShrink: 0 }} />
              <span style={{ color: 'var(--text-secondary)' }}>{shortName(child.name)}</span>
              <span style={{ color: 'var(--text-muted)' }}>{pct.toFixed(0)}%</span>
            </div>
          );
        })}
      </div>

      {/* Tree */}
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
        {root.children.map((child, i) => (
          <TreeNode key={i} node={child} totalDuration={totalDuration} depth={0} runId={runId} step={step} />
        ))}
      </div>
    </div>
  );
}

function TreeNode({ node, totalDuration, depth, runId, step }: {
  node: SpanNode;
  totalDuration: number;
  depth: number;
  runId: string;
  step: number;
}) {
  const [expanded, setExpanded] = useState(depth < 2); // Auto-expand first 2 levels
  const hasChildren = node.children.length > 0;
  const pct = totalDuration > 0 ? (node.duration / totalDuration) * 100 : 0;
  const barWidth = Math.max(pct, 0.5);
  const color = getColor(node.name);
  const groupIdx = node.attributes?.group_idx;

  return (
    <div style={{ marginLeft: depth * 16 }}>
      {/* Row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.375rem',
          padding: '0.1875rem 0',
          cursor: hasChildren ? 'pointer' : 'default',
        }}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {/* Expand/collapse icon */}
        <span style={{ width: 12, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.625rem', flexShrink: 0 }}>
          {hasChildren ? (expanded ? '\u25bc' : '\u25b6') : '\u2500'}
        </span>

        {/* Duration bar */}
        <div style={{ width: 80, height: 12, background: 'var(--bg-elevated)', borderRadius: 2, flexShrink: 0, overflow: 'hidden' }}>
          <div style={{ width: `${barWidth}%`, height: '100%', background: color, borderRadius: 2, minWidth: 1 }} />
        </div>

        {/* Duration text */}
        <span style={{ width: 55, textAlign: 'right', color: 'var(--text-secondary)', flexShrink: 0 }}>
          {node.duration >= 1 ? `${node.duration.toFixed(2)}s` : `${(node.duration * 1000).toFixed(0)}ms`}
        </span>

        {/* Percentage */}
        <span style={{ width: 35, textAlign: 'right', color: 'var(--text-muted)', flexShrink: 0 }}>
          {pct.toFixed(0)}%
        </span>

        {/* Name */}
        <span style={{ color }}>
          {shortName(node.name)}
          {node._concurrent && (
            <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', marginLeft: '0.25rem' }}>
              x{node._concurrent} concurrent
            </span>
          )}
        </span>

        {/* Count indicator if children have same name */}
        {hasChildren && (() => {
          const childCounts = new Map<string, number>();
          for (const c of node.children) {
            childCounts.set(c.name, (childCounts.get(c.name) ?? 0) + 1);
          }
          const uniqueNames = childCounts.size;
          if (uniqueNames < node.children.length) {
            return (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.625rem' }}>
                ({node.children.length} children)
              </span>
            );
          }
          return null;
        })()}

        {/* Group link */}
        {groupIdx != null && (
          <Link
            to={`/runs/${runId}/iterations/${step}/rollouts/${groupIdx}/0`}
            onClick={(e) => e.stopPropagation()}
            style={{ fontSize: '0.625rem', marginLeft: '0.25rem' }}
          >
            group {String(groupIdx)} →
          </Link>
        )}
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div>
          {node.children.map((child, i) => (
            <TreeNode key={i} node={child} totalDuration={totalDuration} depth={depth + 1} runId={runId} step={step} />
          ))}
        </div>
      )}
    </div>
  );
}
