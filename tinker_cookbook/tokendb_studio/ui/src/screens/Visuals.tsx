// Visuals screen: the run's published visuals as a proper run-level tab
// (#/runs/:id/visuals). Each card embeds the live HTML visual itself (they
// poll the read-only SQL endpoint), so the page doubles as a wallboard for
// the run.

import { Link, useParams } from "react-router-dom";
import { apiBase, getJSON, type VisualInfo } from "../api";
import { VisualFrame } from "../components/VisualFrame";
import { useApi } from "../hooks/useApi";
import { visualTitle } from "../util";

/** Placeholder cards sized like real visual frames, shown while loading. */
function VisualsSkeleton() {
  return (
    <div className="visual-grid" aria-hidden="true">
      {[0, 1].map((i) => (
        <div key={i} className="visual-frame">
          <div className="visual-frame-bar">
            <span className="skeleton skeleton-line" style={{ width: `${11 - i * 3}em` }} />
          </div>
          <div className="visual-frame-skeleton-body" />
        </div>
      ))}
    </div>
  );
}

export function Visuals() {
  const { runId } = useParams();
  const base = apiBase(runId);
  const visuals = useApi(() => getJSON<{ visuals: VisualInfo[] }>(`${base}/visuals`), [base]);

  if (visuals.error !== null) return <p className="error">{visuals.error}</p>;
  if (visuals.data === null) return <VisualsSkeleton />;

  const list = visuals.data.visuals;
  if (list.length === 0) {
    return (
      <p className="muted">
        no visuals yet: ask the{" "}
        <Link to="../chat" relative="route">
          chat
        </Link>{" "}
        for a chart and it will appear here
      </p>
    );
  }
  return (
    <div className="visual-grid">
      {list.map((visual) => (
        <VisualFrame key={visual.name} url={visual.url} title={visualTitle(visual)} />
      ))}
    </div>
  );
}
