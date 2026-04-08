import { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
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

function App() {
  const { theme, toggle } = useTheme();

  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <Link to="/" className="app-logo">tinker-chef</Link>
          <span className="app-tagline">Training Dashboard</span>
          <button className="theme-toggle" onClick={toggle} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
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
