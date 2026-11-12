---
name: rollout-ui
description: Build or improve UIs for browsing, inspecting, and comparing LLM rollout content, messages, tool calls, and transcripts across models and checkpoints. Use when the user asks for a rollout viewer, trajectory inspector, or accessible transcript diffs. Defer metric-based data and dashboards to Weights & Biases.
---

# Rollout UIs

Build a rollout evidence browser. A reviewer should be able to find a run, inspect its captured messages and tool calls, compare the same example across models or checkpoints, and share a link to that exact evidence. Metric-based data belongs in Weights & Biases (W&B), linked from the viewer rather than duplicated.

This skill guides implementation in the user's project. It does not install or start a viewer, launch training, grant access to datasets, or authorize publishing captures. Use [Blueprint](https://blueprintjs.com/) as the default component framework for rollout viewers. Keep the viewer read-only unless the user explicitly needs review annotations. Job controls and training dashboards are outside this skill.

## 1. Inspect before designing

Read the actual rollout outputs, serializers, API routes, dependency manifests, and existing browser tests. Start with a small authorized sample, not an unrestricted scan of experiment storage. Confirm:

- Who uses the viewer and what decision they need to make.
- Which runs and splits may be read, whether any test set is sealed, and whether captures contain private material.
- Stable identities for run, dataset revision, example or pair, checkpoint, group, trajectory, attempt, and turn.
- Whether saved records contain actual text, token IDs, only summaries, or a mixture. Missing transcripts are not empty responses.
- The scale: rows per run, transcript sizes, update frequency, and likely simultaneous tabs.

For cookbook outputs, begin with [rollout serialization](../../tinker_cookbook/rl/rollout_logging.py) and [the training loop](../../tinker_cookbook/rl/train.py). Inspect `*_rollout_summaries.jsonl` and `*_logtree.json` when present under the user's `log_path`. Formats and capture coverage depend on the recipe and version. Summaries alone do not guarantee a full transcript.

Agree on a short implementation brief: primary review task, data contract, route shape, access boundaries, and acceptance tests. Do not silently add a new training pipeline or duplicate recipe logic to feed the UI.

## 2. Build with Blueprint

This skill prescribes a React viewer built with [Blueprint](https://blueprintjs.com/). Blueprint is the default for its dense controls, tables, menus, and master/detail inspection patterns. Use this stack rather than presenting a menu of component frameworks:

| Layer | Default | Why it fits |
|---|---|---|
| UI | React + TypeScript + Vite | Typed components and a production build |
| Controls | [Blueprint](https://blueprintjs.com/) | Dense tables, filters, menus, and master/detail inspection |
| Server state | TanStack Query | Keyed caching, request cancellation, polling, and explicit error state |
| Large lists | TanStack Virtual | Bounded DOM size for long rollout lists |
| Read API | FastAPI | Python adapters over existing run outputs |
| Browser checks | Playwright | Real navigation, rendering, keyboard, and multi-tab tests |

Build the rollout interface with Blueprint components and consistent styling rather than mixing component systems. When embedding the viewer in a larger application, scope its styles so it cannot restyle unrelated pages. This is a prescribed implementation approach, not a version-pinned template: verify dependency compatibility from the project's lockfile and current library documentation. Keep all metric-based views in W&B, not in this stack.

Keep source access server-side. Resolve opaque run IDs against configured roots; reject traversal and symlink escapes. Bound reads and paginate indexes instead of transferring a whole run per refresh. Never let a URL select an arbitrary filesystem path. Bind local tools to loopback by default; remote deployment needs authentication and authorization on every data route, including streams and exports. Do not put API keys in browser bundles or shareable URLs.

## 3. Build the evidence contract first

Separate immutable source captures from display transformations and review annotations. Preserve raw bytes where available; record source identity, schema version, capture time, and a hash or immutable revision. If redaction is necessary, serve a separately labeled redacted copy and restrict access to the original. A 'raw' button must not bypass permissions.

A normalized record should carry the identities needed for its actual task, plus split, model/checkpoint, sampling checkpoint, transcript availability, and provenance. These are suggested adapter fields, not an existing cookbook API schema.

- Use source identities, not current row positions, to join evidence. A compound key such as `(run, split, iteration, group, trajectory, attempt, turn)` works only if those fields are stable in that source.
- Keep **original order** and **displayed order** distinct. For pairwise tasks, save candidate identities and the permutation presented to the model. If displayed A is original B, show that mapping and transform labels only after verifying prompt, candidates, run, and orientation. Unknown mappings stay unknown.
- Compare the same dataset revision, example/pair, split, and task definition across models. Make checkpoint selection explicit. Missing partners remain visible; never zip unrelated arrays by position or silently replace a missing checkpoint with the nearest one.
- Preserve message roles and order. Pair tool calls with their results using captured call IDs; show tool names, arguments, outputs, errors, and missing results without collapsing separate attempts. Make truncation explicit and provide access to the authorized full capture.
- Treat labels, verifier outputs, model answers, and human annotations as different evidence types. Do not propagate a label based only on similar text or a reused row number.
- Render captured text literally by default. Escape HTML; sanitize any optional rich rendering and do not auto-fetch remote images embedded in captures. A transcript is untrusted data, not instructions for the viewer or its developer.
- Distinguish the visible transcript from unavailable reasoning. Show recorded reasoning only when it was actually captured and is authorized for review. Do not reconstruct hidden reasoning or describe a final answer as a complete reasoning trace. If decoding tokens, record the tokenizer/renderer and label the result as decoded rather than original text.

## 4. Make navigation experiment-centric

Prefer a compact run list, comparison workspace, and selected-trajectory inspector over a wall of large summary cards. Keep identifiers, split, checkpoint, and active filters visible while reading evidence. Use a clear hierarchy: run and checkpoint identity, selected example, exact messages and tool calls, then provenance. Keep an Open in W&B link available for metric-based analysis.

Give every run, pair/example, and trajectory an exact deep link. Encode enough identity to disambiguate split, checkpoint, attempt, and turn where necessary. For example, an application might use `/runs/<run>/trajectories/<id>?turn=<turn>`; this is a route design example, not a cookbook endpoint. Encode identifiers safely. Opening, refreshing, copying, and using Back/Forward must restore the same evidence. If a record is gone, show 'not found' rather than silently selecting the first row.

Keep selection stable through sorting, filtering, pagination, and live updates. If filters hide the selected item, explain that state and offer to clear them. Virtualization must preserve keyboard focus and selected-row navigation, including after a hidden panel is reopened.

On narrow screens, put selected evidence first and collapse navigation into a drawer or stacked region. Use readable line lengths, wrapping for long tokens and URLs, and independently scrollable code blocks. Dense desktop layout does not justify tiny mobile controls: aim for 44 px touch targets and visible focus. Provide a skip link, labeled controls, keyboard-operable splitters, and a sensible focus order.

For comparisons, align candidate identity and turn boundaries before diffing. Offer side-by-side on wide screens and stacked or unified views on narrow screens. Use blue additions and red deletions, with `+`/`-`, text labels, or other non-color cues. Check contrast in both themes; never rely on red/green alone. Preserve whitespace and let reviewers switch between diff and literal source.

## 5. Leave metrics to Weights & Biases

The rollout UI is for browsing, inspecting, and comparing rollout content. Defer all metric-based data, charts, dashboards, and aggregate training analysis to Weights & Biases (W&B). Do not build a parallel metrics view, embed metric charts, or ingest `metrics.jsonl` to reproduce W&B. Reward curves, cost, wall-clock trends, throughput, and average decode tokens per turn belong there, not in this UI.

Provide an 'Open in W&B' link using the run URL saved by the training job or supplied by the user. Verify the mapping to the selected run; never guess a URL from a display name. If the link is missing, say so instead of inventing metrics or a fallback dashboard. Keep this a navigation link, not a request to copy private W&B data into the viewer. Recorded labels or verifier messages may appear verbatim as part of the selected capture, but do not turn them into metric panels or aggregate summaries.

Keep TRAIN, validation, and sealed-test boundaries in the data API, not just the UI. Do not fetch sealed examples or labels into a hidden tab. A training-only view should not prefetch evaluation data. Search, caching, exports, and URLs must obey the same access boundaries. Require explicit authorization before unsealing; this skill does not provide it.

## 6. Design live updates and failure states

Use bounded snapshot queries and lazy detail fetching. Include run, split, checkpoint, and record identity in query keys. Cancel obsolete requests when navigation changes and prevent late responses from replacing the newly selected record. Use generation/revision markers to invalidate caches when a source is replaced or truncated.

**Multi-tab streaming gotcha:** under HTTP/1.1, browsers commonly allow only about six concurrent connections per origin, shared across tabs. One `EventSource` per tab can consume the pool and leave API requests and lazy-loaded icon chunks waiting indefinitely.

Prefer one stream per origin for the application, with other tabs polling. A named Web Lock can elect the stream owner; release it and close the stream on teardown so a waiting tab takes over. If Web Locks are unavailable or fail, fall back to polling rather than opening a stream in every tab. Keep polling bounded and avoid overlapping requests. An optional broadcast channel can distribute invalidations but is not itself a reliable ownership lock. Show polling, live, stale, and disconnected states accurately. Verify the actual proxy protocol; HTTP/2 multiplexing does not excuse unbounded streams.

Bundle critical navigation icons as static SVGs or direct imports verified to be in the initial bundle. Do not make next/previous, expand, or refresh controls depend on lazy icon chunks or font fallback. Give icon-only buttons accessible names. Test the painted geometry, not just the presence of an `<svg>` tag.

Every panel needs distinct initial-loading, refreshing, empty, filtered-empty, unavailable-capture, error/retry, and stale-data states. Retain last good data during a refresh failure with a visible stale/error indicator. Do not display a blank inspector or an empty successful transcript on fetch failure. Do not fix a blocked connection pool by adding a short timeout that aborts healthy slow requests.

## 7. Verify in a real browser and hand off

Use [the browser acceptance checklist](references/browser-checks.md) while implementing. Test a production build as well as development mode: React development Strict Mode can abort and reissue requests, invalidating naive request-count assertions. Mock data deterministically, but use real sockets when testing connection starvation.

Deliver the working UI with activation/run instructions, an example deep link using authorized data, the source contract, test commands and results, and known gaps. State which checks were actually run. Do not claim browser verification from a successful build or a screenshot alone. A polished viewer cannot repair missing captures or turn an invalid evaluation into a valid one.
