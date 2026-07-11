// Search / SQL screen: regex + token-subsequence search with per-iteration
// hit counts, and a read-only SQL console.

import { postJSON, type StepRow } from "./api";
import { detailHash, el, fmtReward } from "./util";

function resultsTable(rows: StepRow[]): HTMLElement {
  if (rows.length === 0) return el("p", { class: "muted" }, ["no matches"]);
  const header = ["iter", "grp/traj", "step", "attempt", "total", "stop", "action text"];
  return el("table", { class: "feed-table" }, [
    el("thead", {}, [el("tr", {}, header.map((h) => el("th", {}, [h])))]),
    el("tbody", {},
      rows.map((row) =>
        el("tr", { class: row.superseded ? "superseded" : "" }, [
          el("td", {}, [String(row.iteration)]),
          el("td", {}, [el("a", { href: detailHash(row) }, [`${row.group_idx}/${row.traj_idx}`])]),
          el("td", {}, [String(row.step_idx)]),
          el("td", {}, [String(row.run_attempt)]),
          el("td", { class: "num" }, [fmtReward(row.total_reward)]),
          el("td", {}, [row.stop_reason ?? ""]),
          el("td", { class: "preview" }, [row.ac_text ?? ""]),
        ]),
      ),
    ),
  ]);
}

function genericTable(rows: Record<string, unknown>[]): HTMLElement {
  if (rows.length === 0) return el("p", { class: "muted" }, ["0 rows"]);
  const columns = Object.keys(rows[0]);
  const cell = (value: unknown) =>
    typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
  return el("table", { class: "feed-table" }, [
    el("thead", {}, [el("tr", {}, columns.map((c) => el("th", {}, [c])))]),
    el("tbody", {},
      rows.map((row) => el("tr", {}, columns.map((c) => el("td", { class: "preview" }, [cell(row[c])])))),
    ),
  ]);
}

export function renderSearch(root: HTMLElement): () => void {
  // --- Search form ---
  const regexInput = el("input", { placeholder: "regex", style: "width:20em" }) as HTMLInputElement;
  const searchFields = [
    { field: "ac_text", label: "action text" },
    { field: "ob_text", label: "observation text" },
    { field: "logs", label: "logs" },
  ];
  const fieldChecks = searchFields.map(({ field, label }) => {
    const check = el("input", { type: "checkbox" }) as HTMLInputElement;
    check.checked = field !== "logs";
    return { field, label, check };
  });
  const tokensInput = el("input", {
    placeholder: "token ID subsequence, e.g. 128000 882",
    style: "width:18em",
  }) as HTMLInputElement;
  const searchButton = el("button", {}, ["Search"]);
  const searchError = el("div", { class: "error" });
  const hitCounts = el("div", { class: "hit-counts" });
  const searchResults = el("div");

  searchButton.onclick = async () => {
    searchError.textContent = "";
    hitCounts.replaceChildren();
    searchResults.replaceChildren(el("p", { class: "muted" }, ["searching…"]));
    const body: Record<string, unknown> = {};
    if (regexInput.value.trim()) {
      body.regex = regexInput.value.trim();
      body.fields = fieldChecks.filter(({ check }) => check.checked).map(({ field }) => field);
    }
    const tokenText = tokensInput.value.trim();
    if (tokenText) {
      body.token_subsequence = tokenText.split(/[\s,]+/).map(Number);
    }
    try {
      const payload = await postJSON<{ rows: StepRow[]; hit_counts: Record<string, number> }>(
        "/api/search",
        body,
      );
      const counts = Object.entries(payload.hit_counts);
      hitCounts.replaceChildren(
        el("span", { class: "muted" }, ["hits by iteration: "]),
        ...counts.map(([iteration, count]) =>
          el("span", { class: "chip" }, [`iter ${iteration}: ${count}`]),
        ),
      );
      searchResults.replaceChildren(resultsTable(payload.rows));
    } catch (error) {
      searchResults.replaceChildren();
      searchError.textContent = (error as Error).message;
    }
  };

  // --- SQL console ---
  const sqlArea = el("textarea", { rows: "5", spellcheck: "false" }) as HTMLTextAreaElement;
  sqlArea.value =
    "SELECT iteration, count(*) AS n, avg(total_reward) AS avg_reward\nFROM rollouts GROUP BY iteration ORDER BY iteration";
  const runButton = el("button", {}, ["Run"]);
  const sqlError = el("div", { class: "error" });
  const sqlResults = el("div");

  runButton.onclick = async () => {
    sqlError.textContent = "";
    sqlResults.replaceChildren(el("p", { class: "muted" }, ["running…"]));
    try {
      const payload = await postJSON<{ rows: Record<string, unknown>[] }>("/api/sql", {
        query: sqlArea.value,
      });
      sqlResults.replaceChildren(genericTable(payload.rows));
    } catch (error) {
      sqlResults.replaceChildren();
      sqlError.textContent = (error as Error).message;
    }
  };

  root.replaceChildren(
    el("h2", {}, ["Search"]),
    el("div", { class: "filter-bar" }, [
      regexInput,
      ...fieldChecks.map(({ label, check }) => el("label", { class: "toggle" }, [check, ` ${label}`])),
      tokensInput,
      searchButton,
    ]),
    searchError,
    hitCounts,
    searchResults,
    el("h2", {}, ["SQL console"]),
    el("p", { class: "muted small" }, [
      "SELECT-only, over views: rollouts, rollouts_latest, trajectories, labels, segment_rows.",
    ]),
    sqlArea,
    el("div", {}, [runButton]),
    sqlError,
    sqlResults,
  );
  return () => {};
}
