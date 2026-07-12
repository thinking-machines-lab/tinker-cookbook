/** Inline SVG sparkline: a stroked line with a soft area fill underneath. */
export function Sparkline({
  values,
  width = 120,
  height = 28,
  stroke = "#2563eb",
}: {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
}) {
  if (values.length < 2) {
    return <span className="empty-dash">—</span>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const x = (i: number) => (i / (values.length - 1)) * (width - 3) + 1.5;
  const y = (v: number) => height - 2 - ((v - min) / span) * (height - 4);
  const line = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const area = `${line}L${x(values.length - 1).toFixed(1)},${height - 1}L${x(0).toFixed(1)},${height - 1}Z`;
  return (
    <svg className="sparkline" width={width} height={height} aria-hidden="true">
      <path d={area} fill={stroke} opacity={0.12} stroke="none" />
      <path d={line} fill="none" stroke={stroke} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  );
}
