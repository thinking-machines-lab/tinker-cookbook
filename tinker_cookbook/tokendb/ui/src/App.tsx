import { HashRouter, Link, Navigate, Outlet, Route, Routes, useParams } from "react-router-dom";
import { apiBase, getJSON, type Mode, type RunInfo } from "./api";
import { SettingsButton } from "./components/AgentSettings";
import { AppHeader } from "./components/AppHeader";
import { useApi } from "./hooks/useApi";
import { Chat } from "./screens/Chat";
import { Dashboard } from "./screens/Dashboard";
import { Detail } from "./screens/Detail";

/** Header + outlet for the per-run screens (chat, rollout detail). */
function RunLayout({ mode }: { mode: Mode }) {
  const { runId } = useParams();
  const run = useApi(() => getJSON<RunInfo>(`${apiBase(runId)}/run`), [runId]);
  const model = run.data?.context?.model_name ?? "unknown model";
  return (
    <>
      <AppHeader
        nav={
          <>
            {mode === "registry" && <Link to="/">Dashboard</Link>}
            <Link to="chat">Chat</Link>
          </>
        }
        right={
          <>
            <span className="muted small mono">
              {run.data
                ? `${model} · run ${run.data.run_id} · attempt ${run.data.run_attempt}`
                : run.error
                  ? "run.json not found"
                  : ""}
            </span>
            <SettingsButton />
          </>
        }
      />
      <main>
        <Outlet />
      </main>
    </>
  );
}

/** Header + the registry-level cross-run chat (registry mode only). */
function GlobalChatLayout() {
  return (
    <>
      <AppHeader
        nav={
          <>
            <Link to="/">Dashboard</Link>
            <span className="nav-current">Chat across runs</span>
          </>
        }
        right={<SettingsButton />}
      />
      <main>
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
        <Route path="/chat" element={<GlobalChatLayout />} />
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
