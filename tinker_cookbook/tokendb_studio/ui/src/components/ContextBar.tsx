// Level-2 (run-level) context bar, rendered only when a run is open. Three
// parts, left to right:
//   1. a breadcrumb locating the page ("Dashboard / <run>", extended with
//      "/ split/iter/group/traj" on a rollout detail page) — ancestors are
//      links, the current node is plain text with aria-current;
//   2. the run's identity chips (model, LIVE/stale, attempt), so a run-scoped
//      page is never mistaken for a global one;
//   3. the run-scoped tabs (Chat | Visuals) — the active tab is a non-link
//      with aria-current, matching the breadcrumb's "everywhere but here"
//      rule. A rollout detail page is breadcrumb-only: neither tab is active.

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Mode, RegistryRun, RunInfo } from "../api";
import { shortRunId } from "../util";
import { Badge } from "./Badge";

export type RunSection = "chat" | "visuals" | "rollout";

function RunTab({ to, current, children }: { to: string; current: boolean; children: ReactNode }) {
  if (current) {
    return (
      <span className="tab tab-current" aria-current="page">
        {children}
      </span>
    );
  }
  return (
    <Link className="tab" to={to}>
      {children}
    </Link>
  );
}

export function ContextBar({
  mode,
  runId,
  run,
  runError,
  record,
  section,
  rolloutCrumb,
}: {
  mode: Mode;
  runId: string | undefined;
  run: RunInfo | null;
  runError: string | null;
  record: RegistryRun | null;
  section: RunSection;
  /** "split/iteration/group/traj" when a rollout detail page is open. */
  rolloutCrumb: string | null;
}) {
  const runRoot = runId !== undefined ? `/runs/${encodeURIComponent(runId)}` : "/run";
  const recipe = run?.context?.recipe_name ?? record?.recipe_name ?? null;
  const model = run?.context?.model_name ?? record?.model_name ?? null;
  // Run label: what a human calls the run (recipe, falling back to model),
  // plus enough of the run id to disambiguate.
  const displayId = run?.run_id ?? runId ?? null;
  const label =
    displayId !== null
      ? recipe !== null || model !== null
        ? `${recipe ?? model} · ${shortRunId(displayId)}`
        : shortRunId(displayId)
      : null;
  const loading = run === null && runError === null;

  return (
    <div className="context-bar">
      <nav className="crumbs" aria-label="breadcrumb">
        {mode === "registry" && (
          <>
            <Link to="/">Dashboard</Link>
            <span className="crumb-sep" aria-hidden="true">
              /
            </span>
          </>
        )}
        {loading && label === null ? (
          <span className="skeleton skeleton-line" style={{ width: "14em" }} />
        ) : rolloutCrumb !== null ? (
          <Link to={`${runRoot}/chat`}>{label ?? "run"}</Link>
        ) : (
          <span className="crumb-current" aria-current="page">
            {label ?? "run"}
          </span>
        )}
        {rolloutCrumb !== null && (
          <>
            <span className="crumb-sep" aria-hidden="true">
              /
            </span>
            <span className="crumb-current mono" aria-current="page">
              {rolloutCrumb}
            </span>
          </>
        )}
      </nav>
      <div className="run-chips">
        {/* The model chip is redundant when the label already falls back to it. */}
        {recipe !== null && model !== null && <span className="chip">{model}</span>}
        {mode === "registry" &&
          record !== null &&
          (record.status?.live ? (
            <Badge variant="success">LIVE</Badge>
          ) : (
            <Badge variant="neutral">stale</Badge>
          ))}
        {run !== null && <span className="chip">attempt {run.run_attempt}</span>}
        {runError !== null && <span className="muted small">run.json not found</span>}
      </div>
      <nav className="run-tabs" aria-label="run sections">
        <RunTab to={`${runRoot}/chat`} current={section === "chat"}>
          Chat
        </RunTab>
        <RunTab to={`${runRoot}/visuals`} current={section === "visuals"}>
          Visuals
        </RunTab>
      </nav>
    </div>
  );
}
