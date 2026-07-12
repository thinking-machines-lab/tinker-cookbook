import type { ReactNode } from "react";

/** Aggregate tile for the dashboard header. */
export function StatCard({
  title,
  value,
  description,
}: {
  title: string;
  value: ReactNode;
  description?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-card-title">{title}</div>
      <div className="stat-card-value">{value}</div>
      {description && <div className="stat-card-description">{description}</div>}
    </div>
  );
}
