// Search / SQL screen: regex + token-subsequence search with per-iteration
// hit counts, and a read-only SQL console.

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { apiBase, postJSON, type StepRow } from "../api";
import { Chip } from "../components/Chip";
import { CellValue, DataTable } from "../components/DataTable";
import { detailPath, fmtReward } from "../util";

const SEARCH_FIELDS = [
  { field: "ac_text", label: "action text" },
  { field: "ob_text", label: "observation text" },
  { field: "logs", label: "logs" },
];

const DEFAULT_SQL =
  "SELECT iteration, count(*) AS n, avg(total_reward) AS avg_reward\nFROM rollouts GROUP BY iteration ORDER BY iteration";

function SearchResults({ rows }: { rows: StepRow[] }) {
  if (rows.length === 0) return <p className="muted">no matches</p>;
  return (
    <DataTable head={["iteration", "group/traj", "step", "attempt", "total", "stop", "action text"]}>
      {rows.map((row, i) => (
        <tr key={i} className={row.superseded ? "superseded" : undefined}>
          <td className="num">{row.iteration}</td>
          <td className="mono">
            <Link to={`../${detailPath(row)}`}>
              {row.group_idx}/{row.traj_idx}
            </Link>
          </td>
          <td className="num">{row.step_idx}</td>
          <td className="num">{row.run_attempt}</td>
          <td className="num">{fmtReward(row.total_reward)}</td>
          <td>
            <CellValue value={row.stop_reason} />
          </td>
          <td className="preview">
            <CellValue value={row.ac_text} />
          </td>
        </tr>
      ))}
    </DataTable>
  );
}

function SqlResults({ rows }: { rows: Record<string, unknown>[] }) {
  if (rows.length === 0) return <p className="muted">0 rows</p>;
  const columns = Object.keys(rows[0]);
  const cell = (value: unknown) =>
    typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
  return (
    <DataTable head={columns}>
      {rows.map((row, i) => (
        <tr key={i}>
          {columns.map((column) => (
            <td key={column} className="preview">
              <CellValue value={cell(row[column])} />
            </td>
          ))}
        </tr>
      ))}
    </DataTable>
  );
}

export function Search() {
  const { runId } = useParams();
  const base = apiBase(runId);

  // --- Regex / token-subsequence search ---
  const [regex, setRegex] = useState("");
  const [fields, setFields] = useState<Record<string, boolean>>({
    ac_text: true,
    ob_text: true,
    logs: false,
  });
  const [tokens, setTokens] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [results, setResults] = useState<StepRow[] | null>(null);
  const [hitCounts, setHitCounts] = useState<[string, number][]>([]);

  const runSearch = async () => {
    setSearchError("");
    setHitCounts([]);
    setResults(null);
    setSearching(true);
    const body: Record<string, unknown> = {};
    if (regex.trim()) {
      body.regex = regex.trim();
      body.fields = SEARCH_FIELDS.filter(({ field }) => fields[field]).map(({ field }) => field);
    }
    if (tokens.trim()) {
      body.token_subsequence = tokens.trim().split(/[\s,]+/).map(Number);
    }
    try {
      const payload = await postJSON<{ rows: StepRow[]; hit_counts: Record<string, number> }>(
        `${base}/search`,
        body,
      );
      setHitCounts(Object.entries(payload.hit_counts));
      setResults(payload.rows);
    } catch (error) {
      setSearchError((error as Error).message);
    } finally {
      setSearching(false);
    }
  };

  // --- SQL console ---
  const [sql, setSql] = useState(DEFAULT_SQL);
  const [running, setRunning] = useState(false);
  const [sqlError, setSqlError] = useState("");
  const [sqlRows, setSqlRows] = useState<Record<string, unknown>[] | null>(null);

  const runSql = async () => {
    setSqlError("");
    setSqlRows(null);
    setRunning(true);
    try {
      const payload = await postJSON<{ rows: Record<string, unknown>[] }>(`${base}/sql`, {
        query: sql,
      });
      setSqlRows(payload.rows);
    } catch (error) {
      setSqlError((error as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <>
      <h2>Search</h2>
      <div className="filter-bar">
        <input
          placeholder="regex"
          style={{ width: "20em" }}
          value={regex}
          onChange={(event) => setRegex(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void runSearch();
          }}
        />
        {SEARCH_FIELDS.map(({ field, label }) => (
          <label key={field} className="toggle">
            <input
              type="checkbox"
              checked={fields[field]}
              onChange={(event) => setFields((f) => ({ ...f, [field]: event.target.checked }))}
            />{" "}
            {label}
          </label>
        ))}
        <input
          placeholder="token ID subsequence, e.g. 128000 882"
          style={{ width: "18em" }}
          value={tokens}
          onChange={(event) => setTokens(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void runSearch();
          }}
        />
        <button className="primary" onClick={() => void runSearch()}>
          Search
        </button>
      </div>
      {searchError && <div className="error">{searchError}</div>}
      {hitCounts.length > 0 && (
        <div className="hit-counts">
          <span className="muted">hits by iteration: </span>
          {hitCounts.map(([iteration, count]) => (
            <Chip key={iteration}>
              iteration {iteration}: {count}
            </Chip>
          ))}
        </div>
      )}
      {searching && <p className="muted">searching…</p>}
      {results !== null && <SearchResults rows={results} />}

      <h2>SQL console</h2>
      <p className="muted small">
        SELECT-only, over views: rollouts, rollouts_latest, trajectories, labels, segment_rows.
      </p>
      <textarea
        rows={5}
        spellCheck={false}
        value={sql}
        onChange={(event) => setSql(event.target.value)}
      />
      <div>
        <button className="primary" onClick={() => void runSql()}>
          Run
        </button>
      </div>
      {sqlError && <div className="error">{sqlError}</div>}
      {running && <p className="muted">running…</p>}
      {sqlRows !== null && <SqlResults rows={sqlRows} />}
    </>
  );
}
