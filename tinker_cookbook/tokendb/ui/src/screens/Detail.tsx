// Detail screen: full transcript per turn, per-token logprob spans,
// text/raw-token-ID toggle, labels editor, group navigation.

import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { apiBase, getJSON, postJSON, type RolloutDetail, type StepRow } from "../api";
import { Chip } from "../components/Chip";
import { TokenSpan, type TokenViewMode } from "../components/TokenSpan";
import { useApi } from "../hooks/useApi";
import { detailPath, fmtReward, prettyJSON } from "../util";

interface DetailParams {
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  run_attempt?: number;
}

function ActionTokens({ step, mode }: { step: StepRow; mode: TokenViewMode }) {
  const strs = step.ac_token_strs ?? null;
  if (mode === "text" && strs === null) {
    // Tokenizer unavailable: fall back to the stored whole-turn text.
    return (
      <div className="tok-run">
        {step.ac_text ?? "(no action text stored; switch to raw token IDs)"}
      </div>
    );
  }
  return (
    <div className="tok-run">
      {step.ac_tokens.map((tokenId, i) => (
        <TokenSpan
          key={i}
          tokenId={tokenId}
          text={strs ? strs[i] : null}
          logprob={step.ac_logprobs ? step.ac_logprobs[i] : null}
          mode={mode}
        />
      ))}
    </div>
  );
}

function FullContext({ base, tokens }: { base: string; tokens: number[] }) {
  const [expanded, setExpanded] = useState(false);
  const [content, setContent] = useState<string | null>(null);

  const toggle = async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    if (content === null) {
      try {
        const payload = await postJSON<{ strs: string[] }>(`${base}/tokens/decode`, { tokens });
        setContent(payload.strs.join(""));
      } catch (error) {
        // Tokenizer unavailable: raw IDs are canonical and always stored.
        setContent(`(decode unavailable: ${(error as Error).message})\n${tokens.join(" ")}`);
      }
    }
    setExpanded(true);
  };

  return (
    <>
      <button className="linkish" onClick={() => void toggle()}>
        {expanded ? "collapse full context" : "expand full context"}
      </button>
      {expanded && content !== null && <div className="ob-body full-context">{content}</div>}
    </>
  );
}

function ObservationSection({
  step,
  mode,
  base,
}: {
  step: StepRow;
  mode: TokenViewMode;
  base: string;
}) {
  const heading = step.ob_is_delta
    ? "observation (delta: new tokens since last turn)"
    : "observation";
  const body =
    mode === "ids"
      ? step.ob_tokens.join(" ")
      : (step.ob_text ?? `(no observation text stored: ${step.ob_tokens.length} tokens)`);
  return (
    <div className="ob">
      <div className="turn-label">{heading}</div>
      <div className="ob-body">{body}</div>
      {step.ob_is_delta && step.ob_full_tokens && (
        <FullContext base={base} tokens={step.ob_full_tokens} />
      )}
    </div>
  );
}

function StepCard({ step, mode, base }: { step: StepRow; mode: TokenViewMode; base: string }) {
  const meta = [
    `step ${step.step_idx}`,
    `reward ${fmtReward(step.reward)}`,
    step.stop_reason ? `stop ${step.stop_reason}` : "",
    step.episode_done ? "done" : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const logs = prettyJSON(step.logs);
  const metrics = prettyJSON(step.metrics);
  return (
    <div className="step-card">
      <div className="step-meta">{meta}</div>
      <ObservationSection step={step} mode={mode} base={base} />
      <div className="turn-label">action (colored by logprob)</div>
      <ActionTokens step={step} mode={mode} />
      {(metrics || logs) && (
        <details className="step-json">
          <summary>metrics / logs</summary>
          <pre>{[metrics, logs].filter(Boolean).join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

function reproducingSQL(params: DetailParams, attempt: number): string {
  return (
    `SELECT * FROM rollouts WHERE split = '${params.split}' AND iteration = ${params.iteration}` +
    ` AND group_idx = ${params.group_idx} AND traj_idx = ${params.traj_idx}` +
    ` AND run_attempt = ${attempt} ORDER BY step_idx`
  );
}

function LabelEditor({
  base,
  first,
  params,
  onSaved,
}: {
  base: string;
  first: StepRow;
  params: DetailParams;
  onSaved: () => void;
}) {
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const [author, setAuthor] = useState(() => localStorage.getItem("tokendb.author") ?? "");
  const [error, setError] = useState("");

  const add = async () => {
    setError("");
    let parsed: unknown = value;
    try {
      parsed = JSON.parse(value);
    } catch {
      /* keep as plain string */
    }
    try {
      localStorage.setItem("tokendb.author", author);
      await postJSON(`${base}/labels`, {
        key: {
          run_id: first.run_id,
          split: params.split,
          iteration: params.iteration,
          group_idx: params.group_idx,
          traj_idx: params.traj_idx,
        },
        label_key: key,
        label_value: parsed,
        author,
      });
      onSaved();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <>
      <div className="label-editor">
        <input
          placeholder="key"
          style={{ width: "6em" }}
          value={key}
          onChange={(event) => setKey(event.target.value)}
        />
        <input
          placeholder="value (JSON or text)"
          style={{ width: "10em" }}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <input
          placeholder="author"
          style={{ width: "6em" }}
          value={author}
          onChange={(event) => setAuthor(event.target.value)}
        />
        <button onClick={() => void add()}>add label</button>
      </div>
      {error && <div className="error small">{error}</div>}
    </>
  );
}

function CopySQLButton({ sql }: { sql: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        void navigator.clipboard.writeText(sql);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
    >
      {copied ? "copied" : "copy SQL"}
    </button>
  );
}

function Sidebar({
  detail,
  params,
  base,
  refresh,
}: {
  detail: RolloutDetail;
  params: DetailParams;
  base: string;
  refresh: () => void;
}) {
  const first = detail.steps[0];
  const facts: [string, string][] = [
    ["split", first.split],
    ["iteration", String(first.iteration)],
    ["group / traj", `${first.group_idx} / ${first.traj_idx}`],
    ["run attempt", String(first.run_attempt)],
    [
      "sampling client step",
      first.sampling_client_step === null ? "—" : String(first.sampling_client_step),
    ],
    ["source", first.source],
    ["tags", first.tags.join(", ") || "—"],
    ["env row id", first.env_row_id ?? "—"],
    ["total reward", fmtReward(first.total_reward)],
    ["final reward", fmtReward(first.final_reward)],
    ["filtered reason", first.filtered_reason ?? "—"],
  ];
  return (
    <div className="sidebar">
      {first.superseded && (
        <div className="banner-superseded">
          attempt {first.run_attempt}, superseded by a later run attempt (crash/resume)
        </div>
      )}
      <table className="facts">
        <tbody>
          {facts.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="side-block">
        <span className="muted">group siblings: </span>
        {detail.group_traj_idxs.map((idx, i) => (
          <span key={idx}>
            {i > 0 && " "}
            {idx === params.traj_idx ? (
              <strong>{idx}</strong>
            ) : (
              // ".." resolves to the run root (the parent route), so this
              // navigates within the current run in both server modes.
              <Link to={`../${detailPath({ ...params, traj_idx: idx, run_attempt: undefined })}`}>
                {idx}
              </Link>
            )}
          </span>
        ))}
      </div>
      <div className="side-block">
        <CopySQLButton sql={reproducingSQL(params, first.run_attempt)} />
      </div>
      <div className="side-block">
        <div className="turn-label">labels</div>
        {detail.labels.map((label, i) => (
          <div key={i}>
            <Chip title={label.note ?? ""}>
              {label.label_key}: {JSON.stringify(label.label_value)} ({label.author})
            </Chip>
          </div>
        ))}
        <LabelEditor base={base} first={first} params={params} onSaved={refresh} />
      </div>
    </div>
  );
}

function useRolloutDetail(base: string, params: DetailParams, key: string) {
  return useApi(() => {
    const query: Record<string, string> = {};
    if (params.run_attempt !== undefined) query.run_attempt = String(params.run_attempt);
    return getJSON<RolloutDetail>(
      `${base}/rollout/${encodeURIComponent(params.split)}/${params.iteration}/${params.group_idx}/${params.traj_idx}`,
      query,
    );
  }, [key]);
}

export function Detail() {
  const { runId, split, iteration, group, traj } = useParams();
  const [searchParams] = useSearchParams();
  const base = apiBase(runId);
  const [mode, setMode] = useState<TokenViewMode>("text");

  const attemptRaw = searchParams.get("run_attempt");
  const params: DetailParams = {
    split: split ?? "",
    iteration: Number(iteration),
    group_idx: Number(group),
    traj_idx: Number(traj),
    ...(attemptRaw !== null ? { run_attempt: Number(attemptRaw) } : {}),
  };
  const paramsKey = `${base}/${params.split}/${params.iteration}/${params.group_idx}/${params.traj_idx}/${attemptRaw ?? ""}`;

  const detail = useRolloutDetail(base, params, paramsKey);

  if (detail.error) return <p className="error">{detail.error}</p>;
  if (!detail.data) {
    // Skeleton mirroring the loaded layout (header, step cards, facts
    // sidebar) so the page doesn't jump when the rollout arrives.
    return (
      <div className="detail-layout" aria-hidden="true">
        <div className="transcript">
          <div className="detail-header">
            <h2>
              <span className="skeleton skeleton-line" style={{ width: "22em" }} />
            </h2>
          </div>
          {[0, 1].map((i) => (
            <div key={i} className="step-card">
              <div className="step-meta">
                <span className="skeleton skeleton-line" style={{ width: "12em" }} />
              </div>
              <div className="turn-label">
                <span className="skeleton skeleton-line" style={{ width: "8em" }} />
              </div>
              <div className="ob-body">
                <span className="skeleton skeleton-line" />
                <span className="skeleton skeleton-line" style={{ width: "85%" }} />
                <span className="skeleton skeleton-line" style={{ width: "60%" }} />
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar">
          <table className="facts">
            <tbody>
              {Array.from({ length: 11 }, (_, i) => (
                <tr key={i}>
                  <th>
                    <span className="skeleton skeleton-line" style={{ width: "5em" }} />
                  </th>
                  <td>
                    <span className="skeleton skeleton-line" style={{ width: "60%" }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-layout">
      <div className="transcript">
        <div className="detail-header">
          <Link to="../chat" relative="route">
            ← chat
          </Link>
          <h2>
            {params.split} · iteration {params.iteration} · group {params.group_idx} · traj{" "}
            {params.traj_idx}
          </h2>
          <button onClick={() => setMode(mode === "text" ? "ids" : "text")}>
            {mode === "text" ? "view raw token IDs" : "view text"}
          </button>
        </div>
        {detail.data.steps.map((step) => (
          <StepCard key={step.step_idx} step={step} mode={mode} base={base} />
        ))}
      </div>
      <Sidebar detail={detail.data} params={params} base={base} refresh={detail.reload} />
    </div>
  );
}
