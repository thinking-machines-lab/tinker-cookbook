// Detail screen: full transcript per turn, per-token logprob spans,
// text/raw-token-ID toggle, labels editor, group navigation.

import { getJSON, postJSON, type RolloutDetail, type StepRow } from "./api";
import { detailHash, el, fmtReward, logprobColor, prettyJSON, tokenRepr } from "./util";

export interface DetailParams {
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  run_attempt?: number;
}

type ViewMode = "text" | "ids";

function tokenSpan(tokenId: number, text: string | null, logprob: number | null, mode: ViewMode): HTMLElement {
  const shown = mode === "ids" ? `${tokenId} ` : (text ?? String(tokenId));
  const span = el("span", { class: "tok" }, [shown]);
  if (logprob !== null) span.style.color = logprobColor(logprob);
  const parts = [`id ${tokenId}`];
  if (logprob !== null) parts.push(`logprob ${logprob.toFixed(4)}`);
  if (text !== null) parts.push(`repr ${tokenRepr(text)}`);
  span.title = parts.join("\n");
  return span;
}

function acTokenSpans(step: StepRow, mode: ViewMode): HTMLElement {
  const container = el("div", { class: "tok-run" });
  const strs = step.ac_token_strs ?? null;
  if (mode === "text" && strs === null) {
    // Tokenizer unavailable: fall back to the stored whole-turn text.
    container.append(
      step.ac_text ?? "(no ac_text stored; switch to raw token IDs)",
    );
    return container;
  }
  step.ac_tokens.forEach((tokenId, i) => {
    const logprob = step.ac_logprobs ? step.ac_logprobs[i] : null;
    container.append(tokenSpan(tokenId, strs ? strs[i] : null, logprob, mode));
  });
  return container;
}

function obSection(step: StepRow, mode: ViewMode): HTMLElement {
  const section = el("div", { class: "ob" });
  const heading = step.ob_is_delta ? "ob (delta: new tokens since last turn)" : "ob";
  section.append(el("div", { class: "turn-label" }, [heading]));
  const body = el("div", { class: "ob-body" });
  if (mode === "ids") {
    body.textContent = step.ob_tokens.join(" ");
  } else {
    body.textContent = step.ob_text ?? `(no ob_text stored: ${step.ob_tokens.length} tokens)`;
  }
  section.append(body);

  if (step.ob_is_delta && step.ob_full_tokens) {
    const expand = el("button", { class: "linkish" }, ["expand full context"]);
    const full = el("div", { class: "ob-body full-context", hidden: "" });
    expand.onclick = async () => {
      if (!full.hidden) {
        full.hidden = true;
        expand.textContent = "expand full context";
        return;
      }
      if (!full.textContent) {
        try {
          const payload = await postJSON<{ strs: string[] }>("/api/tokens/decode", {
            tokens: step.ob_full_tokens,
          });
          full.textContent = payload.strs.join("");
        } catch (error) {
          // Tokenizer unavailable: raw IDs are canonical and always stored.
          full.textContent = `(decode unavailable: ${(error as Error).message})\n${step.ob_full_tokens!.join(" ")}`;
        }
      }
      full.hidden = false;
      expand.textContent = "collapse full context";
    };
    section.append(expand, full);
  }
  return section;
}

function stepCard(step: StepRow, mode: ViewMode): HTMLElement {
  const meta = [
    `step ${step.step_idx}`,
    `reward ${fmtReward(step.reward)}`,
    step.stop_reason ? `stop ${step.stop_reason}` : "",
    step.episode_done ? "done" : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const card = el("div", { class: "step-card" }, [
    el("div", { class: "step-meta" }, [meta]),
    obSection(step, mode),
    el("div", { class: "turn-label" }, ["ac (colored by logprob)"]),
    acTokenSpans(step, mode),
  ]);
  const logs = prettyJSON(step.logs);
  const metrics = prettyJSON(step.metrics);
  if (metrics || logs) {
    card.append(
      el("details", { class: "step-json" }, [
        el("summary", {}, ["metrics / logs"]),
        el("pre", {}, [[metrics, logs].filter(Boolean).join("\n")]),
      ]),
    );
  }
  return card;
}

function reproducingSQL(params: DetailParams, attempt: number): string {
  return (
    `SELECT * FROM rollouts WHERE split = '${params.split}' AND iteration = ${params.iteration}` +
    ` AND group_idx = ${params.group_idx} AND traj_idx = ${params.traj_idx}` +
    ` AND run_attempt = ${attempt} ORDER BY step_idx`
  );
}

function sidebar(detail: RolloutDetail, params: DetailParams, refresh: () => void): HTMLElement {
  const first = detail.steps[0];
  const side = el("div", { class: "sidebar" });

  if (first.superseded) {
    side.append(
      el("div", { class: "banner-superseded" }, [
        `attempt ${first.run_attempt}, superseded by a later run attempt (crash/resume)`,
      ]),
    );
  }

  const facts: [string, string][] = [
    ["split", first.split],
    ["iteration", String(first.iteration)],
    ["group / traj", `${first.group_idx} / ${first.traj_idx}`],
    ["run_attempt", String(first.run_attempt)],
    ["sampling_client_step", first.sampling_client_step === null ? "—" : String(first.sampling_client_step)],
    ["source", first.source],
    ["tags", first.tags.join(", ") || "—"],
    ["env_row_id", first.env_row_id ?? "—"],
    ["total_reward", fmtReward(first.total_reward)],
    ["final_reward", fmtReward(first.final_reward)],
    ["filtered_reason", first.filtered_reason ?? "—"],
  ];
  side.append(
    el("table", { class: "facts" },
      facts.map(([k, v]) => el("tr", {}, [el("th", {}, [k]), el("td", {}, [v])])),
    ),
  );

  // Group siblings.
  const siblingLinks = detail.group_traj_idxs.map((idx) =>
    idx === params.traj_idx
      ? el("strong", {}, [String(idx)])
      : el("a", { href: detailHash({ ...params, traj_idx: idx, run_attempt: undefined }) }, [String(idx)]),
  );
  const siblings = el("div", { class: "side-block" }, [el("span", { class: "muted" }, ["group siblings: "])]);
  siblingLinks.forEach((link, i) => {
    if (i > 0) siblings.append(" ");
    siblings.append(link);
  });
  side.append(siblings);

  // Copy reproducing SQL.
  const copySQL = el("button", {}, ["copy SQL"]);
  copySQL.onclick = () => {
    void navigator.clipboard.writeText(reproducingSQL(params, first.run_attempt));
    copySQL.textContent = "copied";
    setTimeout(() => (copySQL.textContent = "copy SQL"), 1200);
  };
  side.append(el("div", { class: "side-block" }, [copySQL]));

  // Labels.
  const labelList = el("div", { class: "side-block" });
  labelList.append(el("div", { class: "turn-label" }, ["labels"]));
  for (const label of detail.labels) {
    labelList.append(
      el("div", { class: "chip", title: label.note ?? "" }, [
        `${label.label_key}: ${JSON.stringify(label.label_value)} (${label.author})`,
      ]),
    );
  }
  const keyInput = el("input", { placeholder: "key", style: "width:6em" }) as HTMLInputElement;
  const valueInput = el("input", { placeholder: "value (JSON or text)", style: "width:10em" }) as HTMLInputElement;
  const authorInput = el("input", { placeholder: "author", style: "width:6em" }) as HTMLInputElement;
  authorInput.value = localStorage.getItem("tokendb.author") ?? "";
  const addButton = el("button", {}, ["add label"]);
  const labelError = el("div", { class: "error small" });
  addButton.onclick = async () => {
    labelError.textContent = "";
    let value: unknown = valueInput.value;
    try {
      value = JSON.parse(valueInput.value);
    } catch {
      /* keep as plain string */
    }
    try {
      localStorage.setItem("tokendb.author", authorInput.value);
      await postJSON("/api/labels", {
        key: {
          run_id: first.run_id,
          split: params.split,
          iteration: params.iteration,
          group_idx: params.group_idx,
          traj_idx: params.traj_idx,
        },
        label_key: keyInput.value,
        label_value: value,
        author: authorInput.value,
      });
      refresh();
    } catch (error) {
      labelError.textContent = (error as Error).message;
    }
  };
  labelList.append(
    el("div", { class: "label-editor" }, [keyInput, valueInput, authorInput, addButton]),
    labelError,
  );
  side.append(labelList);
  return side;
}

export function renderDetail(root: HTMLElement, params: DetailParams): () => void {
  let mode: ViewMode = "text";

  const load = async () => {
    root.replaceChildren(el("p", { class: "muted" }, ["loading…"]));
    let detail: RolloutDetail;
    try {
      const query: Record<string, string> = {};
      if (params.run_attempt !== undefined) query.run_attempt = String(params.run_attempt);
      detail = await getJSON<RolloutDetail>(
        `/api/rollout/${encodeURIComponent(params.split)}/${params.iteration}/${params.group_idx}/${params.traj_idx}`,
        query,
      );
    } catch (error) {
      root.replaceChildren(el("p", { class: "error" }, [(error as Error).message]));
      return;
    }

    const modeToggle = el("button", {}, [mode === "text" ? "view raw token IDs" : "view text"]);
    modeToggle.onclick = () => {
      mode = mode === "text" ? "ids" : "text";
      void load();
    };
    const transcript = el("div", { class: "transcript" }, [
      el("div", { class: "detail-header" }, [
        el("a", { href: "#/" }, ["← feed"]),
        el("h2", {}, [
          `${params.split} · iter ${params.iteration} · group ${params.group_idx} · traj ${params.traj_idx}`,
        ]),
        modeToggle,
      ]),
      ...detail.steps.map((step) => stepCard(step, mode)),
    ]);
    root.replaceChildren(
      el("div", { class: "detail-layout" }, [transcript, sidebar(detail, params, () => void load())]),
    );
  };

  void load();
  return () => {};
}
