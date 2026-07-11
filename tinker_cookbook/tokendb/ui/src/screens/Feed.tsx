// Feed screen: reverse-chronological trajectory table with live updates.

import { useCallback, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  apiBase,
  getJSON,
  wsPath,
  type Label,
  type StepRow,
  type TrajectoryRow,
} from "../api";
import { Badge } from "../components/Badge";
import { Chip } from "../components/Chip";
import { CellValue, DataTable } from "../components/DataTable";
import { FilterBar } from "../components/FilterBar";
import { Sparkline } from "../components/Sparkline";
import { useApi } from "../hooks/useApi";
import { useDebounce } from "../hooks/useDebounce";
import { useWebSocket } from "../hooks/useWebSocket";
import { detailPath, fmtReward } from "../util";

const FILTER_FIELDS = [
  { name: "split", placeholder: "split", width: "5.5em" },
  { name: "min_iteration", placeholder: "iteration ≥", width: "5.5em" },
  { name: "max_iteration", placeholder: "iteration ≤", width: "5.5em" },
  { name: "tag", placeholder: "tag", width: "6em" },
  { name: "min_reward", placeholder: "reward ≥", width: "5.5em" },
  { name: "max_reward", placeholder: "reward ≤", width: "5.5em" },
  { name: "stop_reason", placeholder: "stop reason", width: "7em" },
  { name: "text_regex", placeholder: "regex over observation/action text", width: "16em" },
];

function trajKey(row: {
  run_attempt: number;
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
}): string {
  return `${row.run_attempt}/${row.split}/${row.iteration}/${row.group_idx}/${row.traj_idx}`;
}

function labelGroupKey(label: Label): string | null {
  if (label.split === null || label.iteration === null) return null;
  return `${label.split}/${label.iteration}/${label.group_idx}/${label.traj_idx}`;
}

/** Fold a live step row into the trajectory-grain list (newest first). */
function foldRow(trajs: TrajectoryRow[], row: StepRow): TrajectoryRow[] {
  const key = trajKey(row);
  const index = trajs.findIndex((t) => trajKey(t) === key);
  const previous: TrajectoryRow =
    index >= 0
      ? trajs[index]
      : {
          ...row,
          n_steps: 0,
          n_ac_tokens: 0,
          ac_preview: null,
          superseded: false, // live rows come from the newest attempt
        };
  const updated: TrajectoryRow = {
    ...previous,
    n_steps: previous.n_steps + 1,
    n_ac_tokens: previous.n_ac_tokens + row.ac_tokens.length,
    total_reward: row.total_reward,
    final_reward: row.final_reward,
    stop_reason: row.stop_reason ?? previous.stop_reason,
    filtered_reason: row.filtered_reason ?? previous.filtered_reason,
    ac_preview: row.ac_text ?? previous.ac_preview,
  };
  if (index >= 0) {
    const next = [...trajs];
    next[index] = updated;
    return next;
  }
  return [updated, ...trajs];
}

function useFeedLabels(base: string) {
  const labels = useApi(() => getJSON<{ labels: Label[] }>(`${base}/labels`), [base]);
  const byKey = useMemo(() => {
    const map = new Map<string, Label[]>();
    for (const label of labels.data?.labels ?? []) {
      const key = labelGroupKey(label);
      if (key === null) continue;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(label);
    }
    return map;
  }, [labels.data]);
  return { byKey, reload: labels.reload };
}

function FeedRow({ traj, labels }: { traj: TrajectoryRow; labels: Label[] }) {
  return (
    <tr className={traj.superseded ? "superseded" : undefined}>
      <td className="num">{traj.iteration}</td>
      <td className="mono">
        <Link to={detailPath(traj)}>
          {traj.group_idx}/{traj.traj_idx}
        </Link>
      </td>
      <td className="num">
        {traj.run_attempt}
        {traj.superseded && (
          <>
            {" "}
            <Badge variant="danger">superseded</Badge>
          </>
        )}
      </td>
      <td>
        <CellValue value={traj.tags?.join(", ")} />
      </td>
      <td className="mono muted">
        <CellValue value={traj.env_row_id} />
      </td>
      <td className="num">{fmtReward(traj.total_reward)}</td>
      <td className="num">{fmtReward(traj.final_reward)}</td>
      <td className="num">{traj.n_steps}</td>
      <td className="num">{traj.n_ac_tokens}</td>
      <td>
        <CellValue value={traj.stop_reason} />{" "}
        {traj.filtered_reason && <Badge variant="warning">{traj.filtered_reason}</Badge>}
      </td>
      <td className="preview">
        <CellValue value={traj.ac_preview} />
      </td>
      <td>
        {labels.map((label, i) => (
          <Chip key={i} title={label.note ?? ""}>
            {label.label_key}: {JSON.stringify(label.label_value)}
          </Chip>
        ))}
      </td>
    </tr>
  );
}

export function Feed() {
  const { runId } = useParams();
  const base = apiBase(runId);

  const [draft, setDraft] = useState<Record<string, string>>({ source: "" });
  const [showSuperseded, setShowSuperseded] = useState(true);
  const [follow, setFollow] = useState(true);
  const followRef = useRef(follow);
  followRef.current = follow;

  // Filters auto-apply once the inputs settle (Enter applies immediately).
  const debouncedDraft = useDebounce(draft);
  const [applied, setApplied] = useState(debouncedDraft);
  const appliedJson = JSON.stringify(applied);
  if (JSON.stringify(debouncedDraft) !== appliedJson) {
    setApplied(debouncedDraft); // derived state: adopt the settled inputs
  }
  const filters = useMemo(() => {
    const out: Record<string, string> = {};
    for (const [name, value] of Object.entries(applied)) {
      if (value.trim()) out[name] = value.trim();
    }
    return out;
  }, [applied]);

  const [trajs, setTrajs] = useState<TrajectoryRow[]>([]);
  const { byKey: labelsByKey, reload: reloadLabels } = useFeedLabels(base);

  const rollouts = useApi(async () => {
    const payload = await getJSON<{ rows: TrajectoryRow[] }>(`${base}/rollouts`, {
      grain: "trajectories",
      limit: "500",
      ...filters,
    });
    setTrajs(payload.rows); // server returns newest iterations first already
    return payload;
  }, [base, appliedJson]);

  const onMessage = useCallback(
    (msg: Record<string, unknown>) => {
      if (msg.type === "row") {
        if (!followRef.current) return;
        setTrajs((current) => foldRow(current, msg.row as StepRow));
      } else if (msg.type === "labels_changed") {
        reloadLabels();
      }
    },
    [reloadLabels],
  );
  const wsStatus = useWebSocket(wsPath(runId), {
    onMessage,
    sendOnOpen: { type: "subscribe", filters, poll_interval_s: 2 },
  });

  const visible = trajs.filter((traj) => showSuperseded || !traj.superseded);
  const sparkValues = [...visible].reverse().map((traj) => traj.total_reward);

  return (
    <>
      <FilterBar
        fields={FILTER_FIELDS}
        values={draft}
        onChange={(name, value) => setDraft((d) => ({ ...d, [name]: value }))}
        onSubmit={() => setApplied(draft)}
      >
        <select
          value={draft.source}
          onChange={(event) => setDraft((d) => ({ ...d, source: event.target.value }))}
        >
          <option value="">all sources</option>
          <option value="rollout">rollouts</option>
          <option value="filtered">filtered</option>
          <option value="sample">samples</option>
        </select>
        <button onClick={() => setApplied(draft)}>Apply</button>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showSuperseded}
            onChange={(event) => setShowSuperseded(event.target.checked)}
          />{" "}
          show superseded
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
          />{" "}
          follow
        </label>
        {wsStatus === "live" ? (
          <Badge variant="success">live</Badge>
        ) : (
          <Badge variant="neutral">offline</Badge>
        )}
      </FilterBar>
      <div className="spark-row">
        <Sparkline values={sparkValues} width={560} height={36} />
        <span className="muted small">total reward by trajectory (old → new)</span>
      </div>
      {rollouts.error && <p className="error">{rollouts.error}</p>}
      <DataTable
        head={[
          "iteration",
          "group/traj",
          "attempt",
          "tags",
          "env row id",
          "total",
          "final",
          "steps",
          "action tokens",
          "stop",
          "action preview",
          "labels",
        ]}
      >
        {visible.map((traj) => (
          <FeedRow
            key={trajKey(traj)}
            traj={traj}
            labels={
              labelsByKey.get(
                `${traj.split}/${traj.iteration}/${traj.group_idx}/${traj.traj_idx}`,
              ) ?? []
            }
          />
        ))}
      </DataTable>
      {rollouts.loading && trajs.length === 0 && <p className="muted">loading…</p>}
    </>
  );
}
