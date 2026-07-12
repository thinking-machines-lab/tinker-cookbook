# Token DB studio

The interactive analysis app over the [token DB](../tokendb/README.md): a local web viewer with a chat-first workflow (an LLM agent queries your training data for you), per-rollout transcripts, live dashboards, and publishable HTML visuals. The persistence and query layer (capture, parquet segments, `TokenDB`, `RegistryBackend`) lives in `tinker_cookbook/tokendb/`; this package only reads it through its public API.

## Install

```bash
pip install 'tinker-cookbook[tokendb-studio]'
```

This adds `aiohttp` on top of the `tokendb` extra (`pyarrow`, `duckdb`). Training runs that capture token data only need `tinker-cookbook[tokendb]`.

## Viewer quickstart

Single-run mode (one run's store):

```bash
python -m tinker_cookbook.tokendb_studio.serve log_path=~/runs/my-run  # http://127.0.0.1:7423
```

The viewer is chat-first: ask questions about the run in plain language (see [Chat](#chat)) and the agent queries the token DB for you, linking the rollouts it cites. A per-rollout detail view shows the full transcript with tokens colored by logprob and a raw-token-ID toggle.

Registry mode (no `log_path`) shows every run registered in the local [run registry](../tokendb/README.md#run-registry), including concurrently running experiments, in one dashboard:

```bash
python -m tinker_cookbook.tokendb_studio.serve  # registry mode, http://127.0.0.1:7423
```

In registry mode the server exposes:

- `GET /api/runs`: registered runs with a cheap liveness probe (a run is live if any `manifest-*.jsonl` was modified within the last 120 seconds)
- `GET /api/dashboard`: per-run aggregates (row counts, filtered-row count, latest iteration, mean recent reward, a reward-per-iteration sparkline series), TTL-cached so the dashboard can poll cheaply
- All single-run endpoints per run under `/api/runs/{run_id}/...` (rollouts, rollout detail, search, sql, labels, decode, ws), with per-run readers constructed lazily and LRU-cached
- `POST /api/sql` at the registry root: cross-run read-only DuckDB over every registered run (backed by [`RegistryBackend`](../tokendb/README.md#cross-run-sql-registrybackend)), with `run_id` as an ordinary column
- `/ws` (subscribe messages carry a `run_id`) and `/ws/dashboard` (pushes dashboard rows on a poll interval)

The server binds 127.0.0.1 by default; it is a local, unauthenticated viewer. The cross-run reader's segment cache location can be overridden with `segcache_dir=...` (or `TINKER_TOKENDB_SEGCACHE`).

`python -m tinker_cookbook.tokendb.serve` still works as a deprecated alias of `tinker_cookbook.tokendb_studio.serve`.

## Chat

Ask questions about your training data in plain language and an LLM agent answers by querying the token DB for you. No SQL required. The agent can run read-only DuckDB queries (`sql`), search by regex or token-ID subsequence (`search`), pull whole trajectories (`get_rollout`), and publish self-contained HTML visuals (`publish_visual`) that render inline in the chat and in a gallery. In registry mode there is also a global cross-run chat (with `list_runs` and `dashboard` tools for comparing experiments) alongside the per-run chats.

### Providers

To enable chat, give the server an API key for one of the supported providers:

- Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `TINKER_API_KEY` in the server's environment, or
- Configure the provider, model, and key at runtime in the UI settings (backed by `POST /api/agent/config`; the key is held in server memory only and is never written to disk or returned by the API).

The `tinker` provider runs the agent on any model served by Tinker: the model dropdown is populated from `get_server_capabilities().supported_models` (fetched lazily and cached, so it needs `TINKER_API_KEY`), prompts are built with the model's recommended renderer, and tool calls use the renderer's native tool-call format when it has one (Qwen3, DeepSeek, Kimi, GPT-OSS, ...). For models whose renderer has no tool convention, the agent falls back to a documented JSON-in-text protocol (a fenced ```` ```json {"tool": ..., "arguments": ...}```` block).

### Visuals

Published visuals are single HTML files with inline JS/SVG (no external CDNs). For live views the visual polls the read-only SQL endpoint on an interval and re-renders in place, so a chart of, say, reward by iteration keeps updating while training runs. The files are standalone and shareable.

### Background turns

Turns are server-owned: a `user_message` starts an asyncio task that runs the agent loop to completion regardless of the websocket that started it, so switching tabs or navigating away never kills a turn mid-flight. Every streamed record (messages plus coalesced text deltas, tool calls/results, published visuals, and the terminal done/error/cancelled) is appended to the conversation transcript as it happens with a monotonically increasing `seq`. Websockets are subscribers: `{"type": "subscribe_conversation", "conversation_id": ..., "after_seq": N}` replays the persisted records past `N` and then tails the live turn, so a reconnecting (or second) client resumes exactly where it left off with no gaps or duplicates. `{"type": "cancel"}` stops the running turn from any subscribed client. `GET /api/chats` reports `in_flight` per conversation, and `GET /api/chats/recent?limit=5` returns the most recent conversations (across all runs in registry mode, each tagged with its `run_id`), which backs the dashboard's recent-chats card.

On-disk layout: conversations are appended as JSONL to `{log_path}/tokens/chats/{conversation_id}.jsonl` and visuals are written to `{log_path}/tokens/visuals/`. The registry-level chat stores both under the registry directory (`chats/` and `visuals/`) instead. Like everything else, this goes through the `Storage` protocol, so cloud `log_path`s work.

## Frontend

The UI is a React + TypeScript (Vite) app in `ui/`; the built bundle is committed under `static/` and served by the python CLI, so end users never need node. See [`ui/README.md`](ui/README.md) for the dev loop and build instructions.
