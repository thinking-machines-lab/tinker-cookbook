import { HashRouter, Link, Navigate, Outlet, Route, Routes, useParams } from "react-router-dom";
import { apiBase, getJSON, type Mode, type RunInfo } from "./api";
import { useApi } from "./hooks/useApi";
import { Dashboard } from "./screens/Dashboard";
import { Detail } from "./screens/Detail";
import { Feed } from "./screens/Feed";
import { Search } from "./screens/Search";

/** Header + outlet for the per-run screens (feed, detail, search). */
function RunLayout({ mode }: { mode: Mode }) {
  const { runId } = useParams();
  const run = useApi(() => getJSON<RunInfo>(`${apiBase(runId)}/run`), [runId]);
  const model = run.data?.context?.model_name ?? "unknown model";
  return (
    <>
      <header className="app-header">
        <span className="brand">Token DB</span>
        <nav>
          {mode === "registry" && <Link to="/">Dashboard</Link>}
          <Link to=".">Feed</Link>
          <Link to="search">Search / SQL</Link>
        </nav>
        <span className="muted small mono">
          {run.data
            ? `${model} · run ${run.data.run_id} · attempt ${run.data.run_attempt}`
            : run.error
              ? "run.json not found"
              : ""}
        </span>
      </header>
      <main>
        <Outlet />
      </main>
    </>
  );
}

function runRoutes() {
  return (
    <>
      <Route index element={<Feed />} />
      <Route path="rollout/:split/:iteration/:group/:traj" element={<Detail />} />
      <Route path="search" element={<Search />} />
    </>
  );
}

export function App({ mode }: { mode: Mode }) {
  return (
    <HashRouter>
      <Routes>
        <Route
          path="/"
          element={mode === "registry" ? <Dashboard /> : <Navigate to="/run" replace />}
        />
        <Route path="/run" element={<RunLayout mode={mode} />}>
          {runRoutes()}
        </Route>
        <Route path="/runs/:runId" element={<RunLayout mode={mode} />}>
          {runRoutes()}
        </Route>
        <Route path="*" element={<p className="error">unknown route</p>} />
      </Routes>
    </HashRouter>
  );
}
