import { useCallback, useEffect, useRef, useState } from 'react';
import { BrowserRouter, Link, Route, Routes, useSearchParams } from 'react-router-dom';
import { api, setApiSources } from './api/client';
import { DashboardPage } from './pages/DashboardPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RolloutDetailPage } from './pages/RolloutDetailPage';
import { CompareRunsPage } from './pages/CompareRunsPage';
import { EvalRunDetailPage } from './pages/EvalRunDetailPage';
import { EvalTrajectoryPage } from './pages/EvalTrajectoryPage';
import { ChatPage } from './pages/ChatPage';
import './App.css';

const HISTORY_KEY = 'tinker-chef-source-history';
const SOURCES_KEY = 'tinker-chef-sources';
const MAX_HISTORY = 20;

function getHistory(): string[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

function addToHistory(source: string) {
  const history = getHistory().filter(s => s !== source);
  history.unshift(source);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

function saveSourcesToStorage(sources: string[]) {
  localStorage.setItem(SOURCES_KEY, JSON.stringify(sources));
}

function loadSourcesFromStorage(): string[] {
  try {
    return JSON.parse(localStorage.getItem(SOURCES_KEY) || '[]');
  } catch {
    return [];
  }
}

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
  sources: string[];
  sourceLoading: boolean;
  sourceError: string | null;
  onAdd: (uri: string) => void;
  onRemove: (uri: string) => void;
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

  const history = getHistory();
  // Show history entries that are not already active sources
  const recentEntries = history.filter(h => !sources.includes(h));

  const label = sources.length === 0
    ? 'No sources'
    : sources.length === 1
      ? truncateSource(sources[0])
      : `${sources.length} sources`;

  return (
    <div className="source-dropdown" ref={dropdownRef}>
      <button
        className="source-trigger"
        onClick={() => setOpen(!open)}
        title={sources.join('\n') || 'Manage data sources'}
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
                <div key={src} className="source-item">
                  <span className="mono source-url" title={src}>{src}</span>
                  <button
                    className="source-remove"
                    onClick={() => onRemove(src)}
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
          {recentEntries.length > 0 && (
            <>
              <div style={{ padding: '0.375rem 0.5rem', fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', borderTop: '1px solid var(--border)' }}>
                Recent
              </div>
              <div className="source-list">
                {recentEntries.slice(0, 5).map((entry) => (
                  <div key={entry} className="source-item" style={{ cursor: 'pointer' }} onClick={() => onAdd(entry)}>
                    <span className="mono source-url" title={entry} style={{ opacity: 0.7 }}>{truncateSource(entry, 50)}</span>
                  </div>
                ))}
              </div>
            </>
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

function AppContent() {
  const { theme, toggle } = useTheme();
  const [searchParams, setSearchParams] = useSearchParams();
  const [sources, setSources] = useState<string[]>([]);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [sourceVersion, setSourceVersion] = useState(0);
  const [initialized, setInitialized] = useState(false);

  // On mount: resolve sources from URL > localStorage > server defaults
  useEffect(() => {
    const urlSources = searchParams.getAll('source');
    if (urlSources.length > 0) {
      // URL has sources — use them as the source of truth
      setSources(urlSources);
      setApiSources(urlSources);
      saveSourcesToStorage(urlSources);
      urlSources.forEach(addToHistory);
      setInitialized(true);
    } else {
      // No URL sources — try localStorage, then server defaults
      const stored = loadSourcesFromStorage();
      if (stored.length > 0) {
        setSources(stored);
        setApiSources(stored);
        setInitialized(true);
      } else {
        // Fetch defaults from server
        api.getDefaultSources().then((defaults) => {
          setSources(defaults);
          setApiSources(defaults);
          saveSourcesToStorage(defaults);
          defaults.forEach(addToHistory);
          setInitialized(true);
        }).catch(() => {
          // No defaults available — proceed with empty sources
          setInitialized(true);
        });
      }
    }
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep API client in sync whenever sources change
  useEffect(() => {
    setApiSources(sources);
  }, [sources]);

  const updateSources = useCallback((newSources: string[]) => {
    setSources(newSources);
    setApiSources(newSources);
    saveSourcesToStorage(newSources);
    // Update URL params on the dashboard (current page)
    const params = new URLSearchParams(searchParams);
    params.delete('source');
    newSources.forEach(s => params.append('source', s));
    setSearchParams(params, { replace: true });
    setSourceVersion(v => v + 1);
  }, [searchParams, setSearchParams]);

  const handleAddSource = useCallback((uri: string) => {
    if (sources.includes(uri)) return;
    const newSources = [...sources, uri];
    addToHistory(uri);
    updateSources(newSources);
  }, [sources, updateSources]);

  const handleRemoveSource = useCallback((uri: string) => {
    const newSources = sources.filter(s => s !== uri);
    updateSources(newSources);
  }, [sources, updateSources]);

  const handleRefreshSources = useCallback(async () => {
    setSourceLoading(true);
    setSourceError(null);
    try {
      await api.refreshSources(sources);
      setSourceVersion(v => v + 1);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to refresh';
      setSourceError(msg);
    } finally {
      setSourceLoading(false);
    }
  }, [sources]);

  if (!initialized) {
    return (
      <div className="app">
        <header className="app-header">
          <span className="app-logo">tinker-chef</span>
          <span className="app-tagline">Training Dashboard</span>
        </header>
        <main className="app-main">
          <div className="empty-state">Loading...</div>
        </main>
      </div>
    );
  }

  return (
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
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
