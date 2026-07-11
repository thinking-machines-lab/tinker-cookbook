// Formatting helpers shared by the screens.

export function fmtReward(value: number | null | undefined): string {
  if (value === null || value === undefined) return "";
  return Number(value).toFixed(3).replace(/\.?0+$/, "") || "0";
}

export function fmtCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "";
  return value.toLocaleString("en-US");
}

/** Route to a rollout detail page, relative to the current run's root. */
export function detailPath(row: {
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  run_attempt?: number;
}): string {
  const base = `rollout/${encodeURIComponent(row.split)}/${row.iteration}/${row.group_idx}/${row.traj_idx}`;
  return row.run_attempt !== undefined ? `${base}?run_attempt=${row.run_attempt}` : base;
}

/**
 * Map a logprob to a red/yellow/green span color (prob 1 -> green, prob 0 -> red),
 * interpolating within the semantic palette.
 */
export function logprobColor(logprob: number): string {
  const prob = Math.exp(Math.max(-20, logprob));
  // Anchors in HSL: red #ff383c, yellow #ffcc00, green #34c759.
  const red: [number, number, number] = [359, 100, 61];
  const yellow: [number, number, number] = [48, 100, 50];
  const green: [number, number, number] = [134, 57, 49];
  const mix = (a: [number, number, number], b: [number, number, number], t: number) =>
    a.map((v, i) => v + (b[i] - v) * t) as [number, number, number];
  const [h, s, l] = prob < 0.5 ? mix(red, yellow, prob * 2) : mix(yellow, green, (prob - 0.5) * 2);
  return `hsl(${((h % 360) + 360) % 360}, ${s}%, ${l}%)`;
}

/** Printable repr of a decoded token (make whitespace/control chars visible). */
export function tokenRepr(text: string): string {
  return JSON.stringify(text);
}

export function prettyJSON(raw: string): string {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && Object.keys(parsed).length === 0) return "";
    return JSON.stringify(parsed, null, 2);
  } catch {
    return raw;
  }
}

/** "3m ago" style relative time from epoch seconds or an ISO string. */
export function fmtRelative(ts: number | string | null | undefined): string {
  if (ts === null || ts === undefined || ts === "") return "";
  const ms = typeof ts === "number" ? ts * 1000 : Date.parse(ts);
  if (Number.isNaN(ms)) return "";
  const delta = Math.max(0, Date.now() - ms) / 1000;
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}
