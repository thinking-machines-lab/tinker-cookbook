// Small DOM/formatting helpers shared by the screens.

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

export function fmtReward(value: number | null | undefined): string {
  if (value === null || value === undefined) return "";
  return Number(value).toFixed(3).replace(/\.?0+$/, "") || "0";
}

export function detailHash(row: {
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  run_attempt?: number;
}): string {
  const base = `#/rollout/${encodeURIComponent(row.split)}/${row.iteration}/${row.group_idx}/${row.traj_idx}`;
  return row.run_attempt !== undefined ? `${base}?run_attempt=${row.run_attempt}` : base;
}

/** Map a logprob to a green/yellow/red span color (prob 1 -> green, prob 0 -> red). */
export function logprobColor(logprob: number): string {
  const prob = Math.exp(Math.max(-20, logprob));
  const hue = Math.round(120 * prob); // 120 = green, 60 = yellow, 0 = red
  return `hsl(${hue}, 70%, 62%)`;
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

/** Tiny inline sparkline: draws `values` as a filled line on a canvas. */
export function drawSparkline(canvas: HTMLCanvasElement, values: number[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  if (values.length < 2) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const x = (i: number) => (i / (values.length - 1)) * (w - 2) + 1;
  const y = (v: number) => h - 2 - ((v - min) / span) * (h - 4);
  ctx.beginPath();
  values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.strokeStyle = "#7aa2f7";
  ctx.lineWidth = 1.25;
  ctx.stroke();
}
