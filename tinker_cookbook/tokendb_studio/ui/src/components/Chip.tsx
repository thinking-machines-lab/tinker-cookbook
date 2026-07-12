import type { ReactNode } from "react";

/** Metadata pill (labels, hit counts). Quieter than a Badge. */
export function Chip({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <span className="chip" title={title}>
      {children}
    </span>
  );
}
