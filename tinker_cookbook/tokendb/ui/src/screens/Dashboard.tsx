// Dashboard screen (registry mode): every registered run in one table,
// live-updated over the dashboard websocket.

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getJSON, type DashboardRun, type RecentChat } from "../api";
import { Badge } from "../components/Badge";
import { CellValue, DataTable } from "../components/DataTable";
import { Sparkline } from "../components/Sparkline";
import { AppHeader } from "../components/AppHeader";
import { StatCard } from "../components/StatCard";
import { useApi } from "../hooks/useApi";
import { useWebSocket } from "../hooks/useWebSocket";
import { fmtCount, fmtRelative, fmtReward } from "../util";

/** Live runs first, then newest started. */
function sortRuns(runs: DashboardRun[]): DashboardRun[] {
  return [...runs].sort((a, b) => {
    if (a.live !== b.live) return a.live ? -1 : 1;
    return (b.started_at ?? "").localeCompare(a.started_at ?? "");
  });
}

function RunRow({ run }: { run: DashboardRun }) {
  const navigate = useNavigate();
  return (
    <tr
      className="clickable"
      onClick={() => navigate(`/runs/${encodeURIComponent(run.run_id)}/chat`)}
    >
      <td>
        {run.live ? <Badge variant="success">LIVE</Badge> : <Badge variant="neutral">stale</Badge>}{" "}
        {run.error && (
          <Badge variant="danger" title={run.error}>
            error
          </Badge>
        )}
      </td>
      <td className="mono">{run.run_id}</td>
      <td>
        <CellValue value={run.model_name} />
      </td>
      <td>
        <CellValue value={run.recipe_name} />
      </td>
      <td>
        <CellValue value={fmtRelative(run.started_at)} />
      </td>
      <td>
        <CellValue value={fmtRelative(run.last_activity_ts)} />
      </td>
      <td className="num">
        <CellValue value={run.latest_iteration} />
      </td>
      <td className="num">
        <CellValue value={fmtCount(run.n_rows)} />
      </td>
      <td className="num">
        <CellValue value={fmtCount(run.n_filtered_rows)} />
      </td>
      <td className="num">
        <CellValue value={run.mean_recent_reward !== null ? fmtReward(run.mean_recent_reward) : null} />
      </td>
      <td>
        <Sparkline
          values={run.reward_series
            .map((p) => p.mean_total_reward)
            .filter((v): v is number => v !== null)}
        />
      </td>
    </tr>
  );
}

/** How often the recent-chats card refreshes (activity + in_flight). */
const RECENT_CHATS_POLL_MS = 10000;

/** The 5 most recent conversations across every run (and the cross-run
 * chat); clicking one opens that chat with the conversation loaded. An
 * in-flight turn shows a spinner and keeps ticking while you're here. */
function RecentChats() {
  const navigate = useNavigate();
  const recent = useApi(
    () => getJSON<{ conversations: RecentChat[] }>("/api/chats/recent", { limit: "5" }),
    [],
  );
  useEffect(() => {
    const timer = setInterval(() => recent.reload(), RECENT_CHATS_POLL_MS);
    return () => clearInterval(timer);
  }, [recent.reload]);

  const chats = recent.data?.conversations ?? [];
  if (recent.data !== null && chats.length === 0) return null;

  const open = (chat: RecentChat) => {
    const search = `?c=${encodeURIComponent(chat.conversation_id)}`;
    navigate({
      pathname: chat.run_id !== null ? `/runs/${encodeURIComponent(chat.run_id)}/chat` : "/chat",
      search,
    });
  };

  return (
    <div className="recent-chats">
      <div className="recent-chats-label turn-label">Recent chats</div>
      {recent.data === null && !recent.error && (
        <div className="recent-chats-list">
          {[0, 1, 2].map((i) => (
            <div key={i} aria-hidden="true" className="recent-chat">
              <span className="skeleton skeleton-line" style={{ width: `${64 - i * 12}%` }} />
            </div>
          ))}
        </div>
      )}
      <div className="recent-chats-list">
        {chats.map((chat) => (
          <div
            key={`${chat.run_id ?? "global"}/${chat.conversation_id}`}
            className="recent-chat clickable"
            onClick={() => open(chat)}
          >
            {chat.in_flight === true ? (
              <span className="spinner" aria-label="turn running" />
            ) : (
              <span className="recent-chat-dot" aria-hidden="true" />
            )}
            <span className="recent-chat-title">{chat.title || "(empty chat)"}</span>
            <span className="recent-chat-run mono muted">{chat.run_id ?? "all runs"}</span>
            <span className="recent-chat-time muted">
              {chat.in_flight === true ? "running…" : fmtRelative(chat.mtime)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

const TABLE_HEAD = [
  "status",
  "run",
  "model",
  "recipe",
  "started",
  "last activity",
  "iteration",
  "rows",
  "filtered",
  "recent reward",
  "reward (last 50 iterations)",
];

/** Placeholder row matching the height of a real run row while loading. */
function SkeletonRow() {
  return (
    <tr aria-hidden="true" className="skeleton-row">
      {TABLE_HEAD.map((_, i) => (
        <td key={i}>
          <span className="skeleton skeleton-line" />
        </td>
      ))}
    </tr>
  );
}

export function Dashboard() {
  const [runs, setRuns] = useState<DashboardRun[] | null>(null);

  // Initial load over HTTP; the websocket then pushes fresh rows on an interval.
  const initial = useApi(async () => {
    const payload = await getJSON<{ runs: DashboardRun[] }>("/api/dashboard");
    setRuns((current) => current ?? payload.runs);
    return payload;
  }, []);

  const onMessage = useCallback((msg: Record<string, unknown>) => {
    if (msg.type === "dashboard") setRuns(msg.runs as DashboardRun[]);
  }, []);
  const { status: wsStatus } = useWebSocket("/ws/dashboard?poll_interval_s=5", { onMessage });

  const sorted = sortRuns(runs ?? []);
  const nLive = sorted.filter((run) => run.live).length;
  const totalRows = sorted.reduce((sum, run) => sum + (run.n_rows ?? 0), 0);
  const totalFiltered = sorted.reduce((sum, run) => sum + (run.n_filtered_rows ?? 0), 0);

  // While loading, the stat tiles and table keep their loaded dimensions
  // (skeleton values/rows) so data arriving doesn't shift the layout.
  const isLoading = runs === null && !initial.error;
  const stat = (value: string) =>
    isLoading ? <span className="skeleton skeleton-line" style={{ width: "2em" }} /> : value;

  return (
    <>
      <AppHeader
        mode="registry"
        right={
          wsStatus === "live" ? (
            <Badge variant="success">live</Badge>
          ) : (
            <Badge variant="neutral">offline</Badge>
          )
        }
      />
      <main>
        <div className="stat-grid">
          <StatCard title="Runs" value={stat(fmtCount(sorted.length))} description="registered runs" />
          <StatCard title="Live" value={stat(fmtCount(nLive))} description="active in the last 2 minutes" />
          <StatCard title="Rows" value={stat(fmtCount(totalRows))} description="rollout rows across runs" />
          <StatCard
            title="Filtered"
            value={stat(fmtCount(totalFiltered))}
            description="rows dropped by filters"
          />
        </div>
        <RecentChats />
        <div className="dashboard-actions">
          <Link to="/chat">
            <button className="primary" tabIndex={-1}>
              Open all-runs chat
            </button>
          </Link>
          <span className="muted small">
            or click a run to chat about it: ask about reward trends, failure modes, specific
            rollouts
          </span>
        </div>
        <DataTable head={TABLE_HEAD}>
          {isLoading && [0, 1, 2].map((i) => <SkeletonRow key={i} />)}
          {initial.error && runs === null && (
            <tr>
              <td colSpan={TABLE_HEAD.length} className="error">
                {initial.error}
              </td>
            </tr>
          )}
          {runs !== null && sorted.length === 0 && (
            <tr>
              <td colSpan={TABLE_HEAD.length} className="muted">
                no registered runs yet: start a training run with token DB capture enabled and it
                will appear here
              </td>
            </tr>
          )}
          {sorted.map((run) => (
            <RunRow key={`${run.run_id}/${run.run_attempt}`} run={run} />
          ))}
        </DataTable>
      </main>
    </>
  );
}
