import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { StatusBadge, TypeBadge, scoreColor, timeAgo } from '../utils/shared';
import type { DataSource, RunInfo } from '../api/types';

export function DashboardPage() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedForCompare, setSelectedForCompare] = useState<Set<string>>(new Set());
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [sources, setSources] = useState<DataSource[]>([]);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [newSourceUri, setNewSourceUri] = useState('');
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const navigate = useNavigate();

  const toggleCompare = (runId: string) => {
    setSelectedForCompare((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  useEffect(() => {
    api.listRuns()
      .then(setRuns)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    api.listSources().then(setSources).catch(() => {});
  }, []);

  const refreshRuns = () => {
    api.listRuns().then(setRuns).catch(() => {});
  };

  const handleAddSource = async () => {
    if (!newSourceUri.trim()) return;
    setSourceLoading(true);
    setSourceError(null);
    try {
      const result = await api.addSource(newSourceUri.trim());
      setSources(result.sources);
      setNewSourceUri('');
      refreshRuns();
    } catch (e: any) {
      setSourceError(e.message || 'Failed to add source');
    } finally {
      setSourceLoading(false);
    }
  };

  const handleRemoveSource = async (url: string) => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      const result = await api.removeSource(url);
      setSources(result.sources);
      refreshRuns();
    } catch (e: any) {
      setSourceError(e.message || 'Failed to remove source');
    } finally {
      setSourceLoading(false);
    }
  };

  const handleRefreshSources = async () => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      await api.refreshSources();
      const updatedSources = await api.listSources();
      setSources(updatedSources);
      refreshRuns();
    } catch (e: any) {
      setSourceError(e.message || 'Failed to refresh');
    } finally {
      setSourceLoading(false);
    }
  };

  // Extract all unique model names from both training runs and eval runs
  const allModels = useMemo(() => {
    const models = new Set<string>();
    for (const run of runs) {
      const name = run.config_summary?.model_name;
      if (typeof name === 'string' && name) models.add(name);
    }
    return [...models].sort();
  }, [runs]);

  const filteredRuns = useMemo(() => {
    if (selectedModels.size === 0) return runs;
    return runs.filter((r) => {
      const name = r.config_summary?.model_name;
      return typeof name === 'string' && selectedModels.has(name);
    });
  }, [runs, selectedModels]);

  const toggleModel = (model: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  };

  if (loading) return <div className="loading">Loading dashboard...</div>;
  if (error) return (
    <div className="error-msg">
      <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Could not connect to backend</div>
      <div>{error}</div>
      <div style={{ marginTop: '0.75rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
        Make sure <code className="mono">tinker-chef serve</code> is running and pointing at your log directory.
      </div>
    </div>
  );

  const activeRuns = filteredRuns.filter((r) => r.status === 'running');
  const displayedRuns = showAllRuns ? filteredRuns : filteredRuns.slice(0, 10);

  return (
    <div>
      <h2 className="page-title">Dashboard</h2>
      <div className="page-subtitle">
        {activeRuns.length > 0 && <span style={{ color: 'var(--success)' }}>{activeRuns.length} running</span>}
        {activeRuns.length > 0 && filteredRuns.length > activeRuns.length && ' · '}
        {filteredRuns.length - activeRuns.length > 0 && `${filteredRuns.length - activeRuns.length} completed`}
        {filteredRuns.length === 0 && 'No training runs found'}
      </div>

      {/* Data Sources */}
      <div className="card" style={{ marginBottom: '1rem', padding: 0 }}>
        <div
          className="card-header"
          style={{ padding: '0.5rem 1rem', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          onClick={() => setSourcesOpen(!sourcesOpen)}
        >
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Data Sources ({sources.length})
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {sourcesOpen ? '\u25B2' : '\u25BC'}
          </span>
        </div>
        {sourcesOpen && (
          <div style={{ padding: '0.75rem 1rem' }}>
            {sourceError && (
              <div style={{ color: 'var(--error, #e53e3e)', fontSize: '0.8125rem', marginBottom: '0.5rem' }}>
                {sourceError}
              </div>
            )}
            {sources.length > 0 && (
              <div style={{ marginBottom: '0.75rem' }}>
                {sources.map((src) => (
                  <div
                    key={src.url}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '0.5rem',
                      padding: '0.375rem 0', borderBottom: '1px solid var(--border)',
                      fontSize: '0.8125rem',
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-block', padding: '0.0625rem 0.375rem',
                        borderRadius: '8px', fontSize: '0.625rem', fontWeight: 600,
                        textTransform: 'uppercase', letterSpacing: '0.04em',
                        background: src.type === 'local' ? 'var(--accent-dim)' : 'rgba(139, 92, 246, 0.1)',
                        color: src.type === 'local' ? 'var(--accent)' : '#8b5cf6',
                      }}
                    >
                      {src.type}
                    </span>
                    <span className="mono" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {src.url}
                    </span>
                    <button
                      onClick={() => handleRemoveSource(src.url)}
                      disabled={sourceLoading}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--text-muted)', fontSize: '1rem', lineHeight: 1,
                        padding: '0 0.25rem',
                      }}
                      title="Remove source"
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <input
                type="text"
                value={newSourceUri}
                onChange={(e) => setNewSourceUri(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleAddSource(); }}
                placeholder="Local path or s3://... or gs://..."
                style={{
                  flex: 1, padding: '0.375rem 0.625rem',
                  border: '1px solid var(--border)', borderRadius: '6px',
                  background: 'var(--bg-secondary, var(--surface))',
                  color: 'var(--text-primary)', fontSize: '0.8125rem',
                  fontFamily: 'var(--font-mono)',
                }}
              />
              <button
                onClick={handleAddSource}
                disabled={sourceLoading || !newSourceUri.trim()}
                style={{
                  padding: '0.375rem 0.75rem', borderRadius: '6px', border: 'none',
                  background: 'var(--accent)', color: 'white', cursor: 'pointer',
                  fontWeight: 600, fontSize: '0.8125rem',
                  opacity: sourceLoading || !newSourceUri.trim() ? 0.5 : 1,
                }}
              >
                Add
              </button>
              <button
                onClick={handleRefreshSources}
                disabled={sourceLoading}
                style={{
                  padding: '0.375rem 0.75rem', borderRadius: '6px',
                  border: '1px solid var(--border)', background: 'transparent',
                  color: 'var(--text-secondary)', cursor: 'pointer',
                  fontSize: '0.8125rem', opacity: sourceLoading ? 0.5 : 1,
                }}
              >
                Refresh
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Model filter */}
      {allModels.length > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Models
          </span>
          <button
            onClick={() => setSelectedModels(new Set())}
            style={{
              padding: '0.1875rem 0.5rem',
              border: '1px solid var(--border)',
              borderRadius: '12px',
              fontSize: '0.6875rem',
              fontFamily: 'var(--font-mono)',
              cursor: 'pointer',
              background: selectedModels.size === 0 ? 'var(--accent-dim)' : 'transparent',
              color: selectedModels.size === 0 ? 'var(--accent)' : 'var(--text-muted)',
              fontWeight: selectedModels.size === 0 ? 600 : 400,
            }}
          >
            All
          </button>
          {allModels.map((model) => {
            const active = selectedModels.has(model);
            // Short display name: take last component after /
            const shortName = model.includes('/') ? model.split('/').pop()! : model;
            return (
              <button
                key={model}
                onClick={() => toggleModel(model)}
                title={model}
                style={{
                  padding: '0.1875rem 0.5rem',
                  border: '1px solid var(--border)',
                  borderRadius: '12px',
                  fontSize: '0.6875rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  background: active ? 'var(--accent-dim)' : 'transparent',
                  color: active ? 'var(--accent)' : 'var(--text-secondary)',
                  fontWeight: active ? 600 : 400,
                }}
              >
                {shortName}
              </button>
            );
          })}
        </div>
      )}

      {/* Active Runs */}
      {activeRuns.length > 0 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>
            Active Runs
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '0.625rem' }}>
            {activeRuns.map((run) => (
              <RunCard key={run.run_id} run={run} onClick={() => navigate(`/runs/${run.run_id}`)} />
            ))}
          </div>
        </div>
      )}

      {/* Eval progression moved to per-run Eval tab — scores shown inline in run table */}

      {/* All Runs */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        <div className="card-header" style={{ padding: '0.75rem 1rem', cursor: 'pointer' }} onClick={() => setShowAllRuns(!showAllRuns)}>
          <span className="card-title">
            All Runs ({filteredRuns.length}{selectedModels.size > 0 ? ` of ${runs.length}` : ''})
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {!showAllRuns && filteredRuns.length > 10 ? 'Click to expand' : ''}
          </span>
        </div>
        {selectedForCompare.size >= 2 && (
          <div style={{ padding: '0.5rem 1rem', background: 'var(--accent-dim)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.8125rem', fontWeight: 600 }}>
              {selectedForCompare.size} runs selected
            </span>
            <button
              onClick={() => navigate(`/compare?runs=${[...selectedForCompare].join(',')}`)}
              style={{
                padding: '0.375rem 0.75rem', borderRadius: '6px', border: 'none',
                background: 'var(--accent)', color: 'white', cursor: 'pointer',
                fontWeight: 600, fontSize: '0.8125rem',
              }}
            >
              Compare
            </button>
            <button
              onClick={() => setSelectedForCompare(new Set())}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.75rem' }}
            >
              Clear
            </button>
          </div>
        )}
        <table>
          <thead>
            <tr>
              <th style={{ width: '32px' }}></th>
              <th>Run</th>
              <th>Type</th>
              <th>Model</th>
              <th>Status</th>
              <th>Steps</th>
              <th>Eval</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {displayedRuns.map((run) => (
              <tr key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <td onClick={(e) => { e.stopPropagation(); toggleCompare(run.run_id); }} style={{ cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selectedForCompare.has(run.run_id)}
                    onChange={() => {}}
                    style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
                  />
                </td>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{run.run_id}</td>
                <td>
                  <TypeBadge type={run.training_type} />
                </td>
                <td style={{ fontSize: '0.75rem' }}>
                  {(() => {
                    const name = run.config_summary?.model_name;
                    if (typeof name !== 'string') return '-';
                    return name.includes('/') ? name.split('/').pop() : name;
                  })()}
                </td>
                <td>
                  <StatusBadge status={run.status} />
                </td>
                <td className="mono">{run.latest_step ?? '-'}</td>
                <td style={{ fontSize: '0.6875rem' }}>
                  {run.eval_scores ? (
                    <span className="mono">
                      {Object.entries(run.eval_scores as Record<string, number>).map(([bench, score], i) => (
                        <span key={bench}>
                          {i > 0 && ' · '}
                          <span className="text-muted">{bench}:</span>{' '}
                          <span style={{ color: scoreColor(score) }}>{(score * 100).toFixed(0)}%</span>
                        </span>
                      ))}
                    </span>
                  ) : <span className="text-muted">-</span>}
                </td>
                <td className="text-muted" style={{ fontSize: '0.75rem' }}>
                  {run.last_updated ? timeAgo(run.last_updated) : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {runs.length === 0 && (
        <div className="empty-state" style={{ marginTop: '2rem' }}>
          <p>Point <code className="mono">tinker-chef serve</code> at a directory containing training run outputs.</p>
        </div>
      )}
    </div>
  );
}

function RunCard({ run, onClick }: { run: RunInfo; onClick: () => void }) {
  return (
    <div className="card" onClick={onClick} style={{ cursor: 'pointer', padding: '0.875rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.375rem' }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: '0.9375rem' }}>{run.run_id}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>
            {run.config_summary?.model_name as string ?? 'Unknown model'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
          <TypeBadge type={run.training_type} />
          <StatusBadge status={run.status} />
        </div>
      </div>
      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
        {run.latest_step != null && <span>Step {run.latest_step}</span>}
        {run.iteration_count > 0 && <span>{run.iteration_count} iterations</span>}
        {run.last_updated && <span>{timeAgo(run.last_updated)}</span>}
      </div>
    </div>
  );
}
