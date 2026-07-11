import type { ReactNode } from "react";

/**
 * Shared app header so every screen renders the identical bar: brand on the
 * left, nav links next to it, and a right-aligned slot for per-screen content
 * (status text, badges, the settings gear). The height is fixed by CSS
 * (`.app-header { min-height }`), so switching tabs never shifts the layout.
 */
export function AppHeader({ nav, right }: { nav: ReactNode; right?: ReactNode }) {
  return (
    <header className="app-header">
      <span className="brand">Token DB</span>
      <nav>{nav}</nav>
      <div className="app-header-right">{right}</div>
    </header>
  );
}
