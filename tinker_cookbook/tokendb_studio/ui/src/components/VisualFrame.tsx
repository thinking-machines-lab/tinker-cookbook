// Sandboxed iframe for agent-published HTML visuals, inline in the chat
// thread or maximized as a full-screen overlay. allow-same-origin is
// required so live visuals can poll the read-only /api/sql endpoint.

import { useState } from "react";

const SANDBOX = "allow-scripts allow-same-origin";

export function VisualOverlay({
  url,
  title,
  onClose,
}: {
  url: string;
  title: string;
  onClose: () => void;
}) {
  return (
    <div className="visual-overlay" onClick={onClose}>
      <div className="visual-overlay-inner" onClick={(event) => event.stopPropagation()}>
        <div className="visual-frame-bar">
          <span className="visual-title">{title}</span>
          <span className="visual-actions">
            <a href={url} target="_blank" rel="noreferrer">
              open in new tab
            </a>
            <button className="linkish" onClick={onClose}>
              close
            </button>
          </span>
        </div>
        <iframe src={url} title={title} sandbox={SANDBOX} />
      </div>
    </div>
  );
}

export function VisualFrame({
  url,
  title,
  description,
}: {
  url: string;
  title: string;
  description?: string;
}) {
  const [maximized, setMaximized] = useState(false);
  return (
    <div className="visual-frame">
      <div className="visual-frame-bar">
        <span className="visual-title">{title}</span>
        {description && <span className="muted small visual-description">{description}</span>}
        <span className="visual-actions">
          <button className="linkish" onClick={() => setMaximized(true)}>
            maximize
          </button>
          <a href={url} target="_blank" rel="noreferrer">
            open in new tab
          </a>
        </span>
      </div>
      <iframe src={url} title={title} sandbox={SANDBOX} />
      {maximized && <VisualOverlay url={url} title={title} onClose={() => setMaximized(false)} />}
    </div>
  );
}
