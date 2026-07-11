// Level-1 (app-level) navigation bar, identical on every screen: the product
// name as the home link, then the top-level destinations. This bar answers
// "where can I go in the app"; the run context bar below it (ContextBar.tsx)
// answers "what am I looking at". The current destination renders as plain
// text with aria-current="page" (never as a link), so the page you are on is
// never a click target.

import type { ReactNode } from "react";
import { Link, useMatch } from "react-router-dom";
import type { Mode } from "../api";

/** One top-level destination: a link everywhere except on its own page. */
function TopNavItem({ to, children }: { to: string; children: ReactNode }) {
  const current = useMatch({ path: to, end: true }) !== null;
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

export function AppHeader({ mode, right }: { mode: Mode; right?: ReactNode }) {
  return (
    <header className="app-header">
      <Link to="/" className="brand">
        Token DB
      </Link>
      {mode === "registry" && (
        <nav className="app-nav" aria-label="primary">
          <TopNavItem to="/">Dashboard</TopNavItem>
          <TopNavItem to="/chat">All-runs chat</TopNavItem>
        </nav>
      )}
      <div className="app-header-right">{right}</div>
    </header>
  );
}
