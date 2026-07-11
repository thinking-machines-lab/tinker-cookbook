// Feed screen: reverse-chronological trajectory table with live updates.

import { getJSON, subscribe, type Label, type StepRow, type TrajectoryRow } from "./api";
import { detailHash, drawSparkline, el, fmtReward } from "./util";

interface FeedFilters {
  split: string;
  min_iteration: string;
  max_iteration: string;
  tag: string;
  min_reward: string;
  max_reward: string;
  stop_reason: string;
  source: string; // "" = all, "rollout", "filtered", "sample"
  text_regex: string;
}

function trajKey(row: {
  run_attempt: number;
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
}): string {
  return `${row.run_attempt}/${row.split}/${row.iteration}/${row.group_idx}/${row.traj_idx}`;
}

function labelKey(label: Label): string | null {
  if (label.split === null || label.iteration === null) return null;
  return `${label.split}/${label.iteration}/${label.group_idx}/${label.traj_idx}`;
}

/** Fold a live step row into the trajectory-grain map. */
function foldRow(map: Map<string, TrajectoryRow>, row: StepRow): TrajectoryRow {
  const key = trajKey(row);
  let traj = map.get(key);
  if (!traj) {
    traj = {
      ...row,
      n_steps: 0,
      n_ac_tokens: 0,
      ac_preview: null,
      superseded: false, // live rows come from the newest attempt
    };
    map.set(key, traj);
  }
  traj.n_steps += 1;
  traj.n_ac_tokens += row.ac_tokens.length;
  traj.total_reward = row.total_reward;
  traj.final_reward = row.final_reward;
  traj.stop_reason = row.stop_reason ?? traj.stop_reason;
  traj.filtered_reason = row.filtered_reason ?? traj.filtered_reason;
  traj.ac_preview = row.ac_text ?? traj.ac_preview;
  return traj;
}

export function renderFeed(root: HTMLElement): () => void {
  let showSuperseded = true;
  let follow = true;
  const trajMap = new Map<string, TrajectoryRow>();
  let order: string[] = []; // newest first
  let labelsByKey = new Map<string, Label[]>();

  // --- Filter bar ---
  const inputs: Partial<Record<keyof FeedFilters, HTMLInputElement | HTMLSelectElement>> = {};
  const textInput = (name: keyof FeedFilters, placeholder: string, width = "7em") => {
    const input = el("input", { placeholder, style: `width:${width}` });
    inputs[name] = input;
    return input;
  };
  const sourceSelect = el("select", {}, [
    el("option", { value: "" }, ["all sources"]),
    el("option", { value: "rollout" }, ["rollouts"]),
    el("option", { value: "filtered" }, ["filtered"]),
    el("option", { value: "sample" }, ["samples"]),
  ]);
  inputs.source = sourceSelect;
  const supersededToggle = el("input", { type: "checkbox" }) as HTMLInputElement;
  supersededToggle.checked = true;
  const followToggle = el("input", { type: "checkbox" }) as HTMLInputElement;
  followToggle.checked = true;
  const applyButton = el("button", {}, ["Apply"]);
  const statusDot = el("span", { class: "status-dot" }, ["○"]);

  const filterBar = el("div", { class: "filter-bar" }, [
    textInput("split", "split", "5em"),
    textInput("min_iteration", "iter ≥", "4.5em"),
    textInput("max_iteration", "iter ≤", "4.5em"),
    textInput("tag", "tag", "6em"),
    textInput("min_reward", "reward ≥", "5em"),
    textInput("max_reward", "reward ≤", "5em"),
    textInput("stop_reason", "stop_reason", "7em"),
    sourceSelect,
    textInput("text_regex", "regex over ob/ac text", "14em"),
    applyButton,
    el("label", { class: "toggle" }, [supersededToggle, " show superseded"]),
    el("label", { class: "toggle" }, [followToggle, " follow"]),
    statusDot,
  ]);

  const sparkCanvas = el("canvas", { width: "560", height: "36", class: "sparkline" });
  const sparkLabel = el("span", { class: "muted small" }, ["total_reward by trajectory (old → new)"]);
  const tbody = el("tbody");
  const table = el("table", { class: "feed-table" }, [
    el("thead", {}, [
      el("tr", {}, [
        "iter",
        "grp/traj",
        "attempt",
        "tags",
        "env_row_id",
        "total",
        "final",
        "steps",
        "ac toks",
        "stop",
        "ac preview",
        "labels",
      ].map((h) => el("th", {}, [h]))),
    ]),
    tbody,
  ]);
  root.replaceChildren(filterBar, el("div", { class: "spark-row" }, [sparkCanvas, sparkLabel]), table);

  const currentFilters = (): Record<string, string> => {
    const out: Record<string, string> = {};
    for (const [name, input] of Object.entries(inputs)) {
      const value = (input as HTMLInputElement).value.trim();
      if (value) out[name] = value;
    }
    return out;
  };

  const renderRow = (traj: TrajectoryRow): HTMLTableRowElement => {
    const badges: Node[] = [];
    if (traj.filtered_reason) {
      badges.push(el("span", { class: "badge badge-filtered" }, [traj.filtered_reason]));
    }
    const attemptCell: (Node | string)[] = [String(traj.run_attempt)];
    if (traj.superseded) {
      attemptCell.push(" ", el("span", { class: "badge badge-superseded" }, ["superseded"]));
    }
    const labels = labelsByKey.get(
      `${traj.split}/${traj.iteration}/${traj.group_idx}/${traj.traj_idx}`,
    );
    const chips = (labels ?? []).map((label) =>
      el("span", { class: "chip", title: label.note ?? "" }, [
        `${label.label_key}: ${JSON.stringify(label.label_value)}`,
      ]),
    );
    const tr = el("tr", { class: traj.superseded ? "superseded" : "" }, [
      el("td", {}, [String(traj.iteration)]),
      el("td", {}, [
        el("a", { href: detailHash(traj) }, [`${traj.group_idx}/${traj.traj_idx}`]),
      ]),
      el("td", {}, attemptCell),
      el("td", {}, [traj.tags?.join(", ") ?? ""]),
      el("td", { class: "muted" }, [traj.env_row_id ?? ""]),
      el("td", { class: "num" }, [fmtReward(traj.total_reward)]),
      el("td", { class: "num" }, [fmtReward(traj.final_reward)]),
      el("td", { class: "num" }, [String(traj.n_steps)]),
      el("td", { class: "num" }, [String(traj.n_ac_tokens)]),
      el("td", {}, [traj.stop_reason ?? "", ...badges]),
      el("td", { class: "preview" }, [traj.ac_preview ?? ""]),
      el("td", {}, chips),
    ]);
    return tr;
  };

  const rerender = () => {
    const rows = order
      .map((key) => trajMap.get(key)!)
      .filter((traj) => showSuperseded || !traj.superseded);
    tbody.replaceChildren(...rows.map(renderRow));
    const chronological = [...rows].reverse().map((traj) => traj.total_reward);
    drawSparkline(sparkCanvas as HTMLCanvasElement, chronological);
  };

  const loadLabels = async () => {
    try {
      const payload = await getJSON<{ labels: Label[] }>("/api/labels");
      labelsByKey = new Map();
      for (const label of payload.labels) {
        const key = labelKey(label);
        if (key === null) continue;
        if (!labelsByKey.has(key)) labelsByKey.set(key, []);
        labelsByKey.get(key)!.push(label);
      }
    } catch {
      /* labels are decorative in the feed */
    }
  };

  const load = async () => {
    await loadLabels();
    const payload = await getJSON<{ rows: TrajectoryRow[] }>("/api/rollouts", {
      grain: "trajectories",
      limit: "500",
      ...currentFilters(),
    });
    trajMap.clear();
    order = [];
    // Server returns newest iterations first already.
    for (const traj of payload.rows) {
      const key = trajKey(traj);
      trajMap.set(key, traj);
      order.push(key);
    }
    rerender();
  };

  // --- Live updates ---
  const wsHandlers = {
    onRow: (row: StepRow) => {
      if (!follow) return;
      const key = trajKey(row);
      const isNew = !trajMap.has(key);
      foldRow(trajMap, row);
      if (isNew) order.unshift(key);
      rerender();
    },
    onLabelsChanged: () => void loadLabels().then(rerender),
    onStatus: (status: string) => {
      statusDot.textContent = status === "live" ? "● live" : "○ offline";
      statusDot.className = `status-dot ${status === "live" ? "live" : ""}`;
    },
  };
  let closeWs = subscribe(currentFilters(), wsHandlers);
  const resubscribe = () => {
    closeWs();
    closeWs = subscribe(currentFilters(), wsHandlers);
  };

  applyButton.onclick = () => {
    void load();
    resubscribe();
  };
  supersededToggle.onchange = () => {
    showSuperseded = supersededToggle.checked;
    rerender();
  };
  followToggle.onchange = () => {
    follow = followToggle.checked;
  };
  filterBar.addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key === "Enter") applyButton.click();
  });

  void load();
  return () => closeWs();
}
