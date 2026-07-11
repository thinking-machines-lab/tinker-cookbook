import type { ReactNode } from "react";

export type BadgeVariant = "success" | "danger" | "warning" | "neutral";

/** Small solid status pill (LIVE, superseded, filtered reasons, errors). */
export function Badge({
  variant = "neutral",
  title,
  children,
}: {
  variant?: BadgeVariant;
  title?: string;
  children: ReactNode;
}) {
  return (
    <span className={`badge badge-${variant}`} title={title}>
      {children}
    </span>
  );
}
