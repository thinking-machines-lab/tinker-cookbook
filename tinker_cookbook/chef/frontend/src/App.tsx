import { useCallback, useEffect, useRef, useState } from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import { api } from './api/client';
import type { DataSource } from './api/types';
import { DashboardPage } from './pages/DashboardPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RolloutDetailPage } from './pages/RolloutDetailPage';
import { CompareRunsPage } from './pages/CompareRunsPage';
import { EvalRunDetailPage } from './pages/EvalRunDetailPage';
import { EvalTrajectoryPage } from './pages/EvalTrajectoryPage';
import { ChatPage } from './pages/ChatPage';
import './App.css';

function useTheme() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('tinker-chef-theme') as 'dark' | 'light') ?? 'dark';
    }
    return 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tinker-chef-theme', theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggle };
}

/** Truncate a path/URL to at most `max` characters, keeping the tail. */
function truncateSource(url: string, max = 40): string {
  if (url.length <= max) return url;
  return '\u2026' + url.slice(url.length - max + 1);
}

function SourceDropdown({
  sources,
  sourceLoading,
  sourceError,
  onAdd,
  onRemove,
  onRefresh,
}: {
  sources: DataSource[];
  sourceLoading: boolean;
  sourceError: string | null;
  onAdd: (uri: string) => void;
  onRemove: (url: string) => void;
  onRefresh: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [newUri, setNewUri] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleAdd = () => {
    if (!newUri.trim()) return;
    onAdd(newUri.trim());
    setNewUri('');
  };

  const label = sources.length === 0
    ? 'No sources'
    : sources.length === 1
      ? truncateSource(sources[0].url)
      : `${sources.length} sources`;

  return (
    <div className="source-dropdown" ref={dropdownRef}>
      <button
        className="source-trigger"
        onClick={() => setOpen(!open)}
        title={sources.map((s) => s.url).join('\n') || 'Manage data sources'}
      >
        <span className="source-trigger-label">{label}</span>
        <span className="source-trigger-caret">{open ? '\u25B2' : '\u25BC'}</span>
      </button>
      {open && (
        <div className="source-panel">
          {sourceError && (
            <div style={{ color: 'var(--error)', fontSize: '0.75rem', padding: '0.375rem 0.5rem' }}>
              {sourceError}
            </div>
          )}
          {sources.length > 0 && (
            <div className="source-list">
              {sources.map((src) => (
                <div key={src.url} className="source-item">
                  <span
                    className="source-type-badge"
                    style={{
                      background: src.type === 'local' ? 'var(--accent-dim)' : 'rgba(139, 92, 246, 0.1)',
                      color: src.type === 'local' ? 'var(--accent)' : '#8b5cf6',
                    }}
                  >
                    {src.type}
                  </span>
                  <span className="mono source-url" title={src.url}>{src.url}</span>
                  <button
                    className="source-remove"
                    onClick={() => onRemove(src.url)}
                    disabled={sourceLoading}
                    title="Remove source"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
          {sources.length === 0 && (
            <div style={{ padding: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}>
              No data sources configured
            </div>
          )}
          <div className="source-add-row">
            <input
              type="text"
              value={newUri}
              onChange={(e) => setNewUri(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleAdd(); }}
              placeholder="Path or s3://... or gs://..."
              className="source-input"
            />
            <button
              onClick={handleAdd}
              disabled={sourceLoading || !newUri.trim()}
              className="source-add-btn"
            >
              Add
            </button>
            <button
              onClick={onRefresh}
              disabled={sourceLoading}
              className="source-refresh-btn"
              title="Refresh all sources"
            >
              {sourceLoading ? '\u21BB' : '\u21BB'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const { theme, toggle } = useTheme();
  const [sources, setSources] = useState<DataSource[]>([]);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [sourceVersion, setSourceVersion] = useState(0);

  useEffect(() => {
    api.listSources().then(setSources).catch(() => {});
  }, []);

  const handleAddSource = useCallback(async (uri: string) => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      const result = await api.addSource(uri);
      setSources(result.sources);
      setSourceVersion((v) => v + 1);
    } catch (e: any) {
      setSourceError(e.message || 'Failed to add source');
    } finally {
      setSourceLoading(false);
    }
  }, []);

  const handleRemoveSource = useCallback(async (url: string) => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      const result = await api.removeSource(url);
      setSources(result.sources);
      setSourceVersion((v) => v + 1);
    } catch (e: any) {
      setSourceError(e.message || 'Failed to remove source');
    } finally {
      setSourceLoading(false);
    }
  }, []);

  const handleRefreshSources = useCallback(async () => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      await api.refreshSources();
      const updatedSources = await api.listSources();
      setSources(updatedSources);
      setSourceVersion((v) => v + 1);
    } catch (e: any) {
      setSourceError(e.message || 'Failed to refresh');
    } finally {
      setSourceLoading(false);
    }
  }, []);

  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <Link to="/" className="app-logo">tinker-chef</Link>
          <span className="app-tagline">Training Dashboard</span>
          <SourceDropdown
            sources={sources}
            sourceLoading={sourceLoading}
            sourceError={sourceError}
            onAdd={handleAddSource}
            onRemove={handleRemoveSource}
            onRefresh={handleRefreshSources}
          />
          <button className="theme-toggle" onClick={toggle} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage key={sourceVersion} />} />
            <Route path="/compare" element={<CompareRunsPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/runs/:runId/iterations/:iteration/rollouts/:groupIdx/:trajIdx" element={<RolloutDetailPage />} />
            <Route path="/runs/:runId/chat" element={<ChatPage />} />
            <Route path="/eval/:evalRunId" element={<EvalRunDetailPage />} />
            <Route path="/eval/:evalRunId/:benchmark/:idx" element={<EvalTrajectoryPage />} />
            <Route path="*" element={
              <div className="empty-state">
                <div style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Page not found</div>
                <Link to="/">Back to Dashboard</Link>
              </div>
            } />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
