import type { ReactNode } from "react";

/** Table shell with the shared header/body styling; rows come in as children. */
export function DataTable({ head, children }: { head: ReactNode[]; children: ReactNode }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** Render a value, or a muted em dash for empty cells. */
export function CellValue({ value }: { value: ReactNode }) {
  if (value === null || value === undefined || value === "") {
    return <span className="empty-dash">—</span>;
  }
  return <>{value}</>;
}
