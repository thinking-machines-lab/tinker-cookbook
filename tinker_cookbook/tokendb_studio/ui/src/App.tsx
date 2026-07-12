// Route table and per-scope layouts. Navigation is two levels:
//   Level 1 (AppHeader): the app-wide destinations — Dashboard and All-runs
//     chat in registry mode, just the product name in single-run mode.
//   Level 2 (ContextBar): only inside a run — breadcrumb, run identity chips,
//     and the run-scoped tabs (Chat | Visuals). Rollout detail pages extend
//     the breadcrumb instead of adding a tab.

import { HashRouter, Navigate, Outlet, Route, Routes, useMatch, useParams } from "react-router-dom";
import { apiBase, getJSON, type Mode, type RegistryRun, type RunInfo } from "./api";
import { SettingsButton } from "./components/AgentSettings";
import { AppHeader } from "./components/AppHeader";
import { ContextBar, type RunSection } from "./components/ContextBar";
import { useApi } from "./hooks/useApi";
import { Chat } from "./screens/Chat";
import { Dashboard } from "./screens/Dashboard";
import { Detail } from "./screens/Detail";
import { Visuals } from "./screens/Visuals";

/** App header + run context bar + outlet for the per-run screens. */
function RunLayout({ mode }: { mode: Mode }) {
  const { runId } = useParams();
  const run = useApi(() => getJSON<RunInfo>(`${apiBase(runId)}/run`), [runId]);
  // Registry mode: the run's registry record supplies recipe/model fallbacks
  // and the LIVE/stale probe for the identity chips.
  const registry = useApi<{ runs: RegistryRun[] } | null>(
    () =>
      mode === "registry" ? getJSON<{ runs: RegistryRun[] }>("/api/runs") : Promise.resolve(null),
    [mode],
  );
  const record = registry.data?.runs.find((r) => r.run_id === runId) ?? null;

  // Which child route is showing decides the active tab; a rollout detail
  // page is a breadcrumb extension, not a tab.
  // (`??` would skip the second useMatch when the first hits — call all four
  // unconditionally to keep the hook order stable.)
  const detailRegistry = useMatch("/runs/:runId/rollout/:split/:iteration/:group/:traj");
  const detailSingle = useMatch("/run/rollout/:split/:iteration/:group/:traj");
  const visualsRegistry = useMatch("/runs/:runId/visuals");
  const visualsSingle = useMatch("/run/visuals");
  const detailMatch = detailRegistry !== null ? detailRegistry : detailSingle;
  const visualsMatch = visualsRegistry !== null ? visualsRegistry : visualsSingle;
  const section: RunSection =
    detailMatch !== null ? "rollout" : visualsMatch !== null ? "visuals" : "chat";
  const p = detailMatch?.params;
  const rolloutCrumb = p !== undefined ? `${p.split}/${p.iteration}/${p.group}/${p.traj}` : null;

  return (
    <>
      <AppHeader mode={mode} right={<SettingsButton />} />
      <ContextBar
        mode={mode}
        runId={runId}
        run={run.data}
        runError={run.error}
        record={record}
        section={section}
        rolloutCrumb={rolloutCrumb}
      />
      <main>
        <Outlet />
      </main>
    </>
  );
}

/** The registry-level cross-run chat: app header + a scope heading (no run
 * context bar — the absence of run identity is what marks this as global). */
function GlobalChatLayout() {
  return (
    <>
      <AppHeader mode="registry" right={<SettingsButton />} />
      <main>
        <div className="page-heading">
          <h1>All-runs chat</h1>
          <span className="muted small">queries across every registered run</span>
        </div>
        <Chat scope="global" />
      </main>
    </>
  );
}

function runRoutes() {
  return (
    <>
      <Route index element={<Navigate to="chat" replace />} />
      <Route path="chat" element={<Chat scope="run" />} />
      <Route path="visuals" element={<Visuals />} />
      <Route path="rollout/:split/:iteration/:group/:traj" element={<Detail />} />
    </>
  );
}

export function App({ mode }: { mode: Mode }) {
  return (
    <HashRouter>
      <Routes>
        <Route
          path="/"
          element={mode === "registry" ? <Dashboard /> : <Navigate to="/run/chat" replace />}
        />
        <Route
          path="/chat"
          element={mode === "registry" ? <GlobalChatLayout /> : <Navigate to="/run/chat" replace />}
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
