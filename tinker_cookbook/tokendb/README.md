# Token DB

Capture every raw token exchanged during RL rollouts (prompts, sampled tokens, per-token logprobs, rewards) into parquet segments under your run's `log_path`, then inspect them with a live web viewer, the Python API, or plain DuckDB SQL. Designed so that agents can query training data programmatically, not just humans clicking through HTML reports.

## Install

The write and read paths use optional dependencies (`pyarrow`, `duckdb`, `aiohttp`):

```bash
pip install 'tinker-cookbook[tokendb]'
```

A run with `token_db` disabled never imports any of these.

## Enabling capture on an RL run

Set `token_db` on the RL training `Config`:

```python
from tinker_cookbook.tokendb import TokenDbConfig

config = Config(
    ...,
    token_db=TokenDbConfig(),  # defaults: store_text=True, capture_filtered=True
)
```

Every rollout transition is then written as one row (one row per turn) under `{log_path}/tokens/`. Rows carry the same `(split, iteration, group_idx, traj_idx, step_idx)` keys as the rollout-summary JSONL files, so the two line up 1:1. Groups dropped by the rollout pipeline (constant reward, rollout errors) are captured too, as `source="filtered"` rows with a `filtered_reason`.

`TokenDbConfig` knobs: `store_text` (decode `ob_text` / `ac_text` alongside the canonical token IDs), `buffer_rows` and `flush_interval_s` (writer batching), `capture_filtered`.

## Live viewer

```bash
python -m tinker_cookbook.tokendb.serve log_path=~/runs/my-run  # http://127.0.0.1:7423
```

The viewer is chat-first: ask questions about the run in plain language (see [Chat](#chat)) and the agent queries the token DB for you, linking the rollouts it cites. A per-rollout detail view shows the full transcript with tokens colored by logprob and a raw-token-ID toggle.

## Multi-run dashboard and run registry

Start the server with no `log_path` to see every run, including concurrently running experiments, in one dashboard:

```bash
python -m tinker_cookbook.tokendb.serve  # registry mode, http://127.0.0.1:7423
```

This works because every coordinator `TokenDbWriter` registers its run in a local **run registry**: one JSON file per run (so concurrent jobs never share an append target) recording `run_id`, `log_path`, `model_name`, `recipe_name`, `started_at`, `pid`, and `hostname`.

- Default location: `~/.cache/tinker-cookbook/tokendb/runs/`
- Override with the `TINKER_TOKENDB_REGISTRY` environment variable, or per run via `TokenDbConfig(registry_dir=...)` / `TokenDbWriter(..., registry_dir=...)`
- Disable by setting either to an empty string (`TINKER_TOKENDB_REGISTRY=""` or `registry_dir=""`)

Registration is best-effort: a broken registry logs a warning and never breaks training. Worker-mode writers (explicit `run_id` / `run_attempt` in context) never register; only the coordinator does.

In registry mode the server exposes:

- `GET /api/runs`: registered runs with a cheap liveness probe (a run is live if any `manifest-*.jsonl` was modified within the last 120 seconds)
- `GET /api/dashboard`: per-run aggregates (row counts, filtered-row count, latest iteration, mean recent reward, a reward-per-iteration sparkline series), TTL-cached so the dashboard can poll cheaply
- All single-run endpoints per run under `/api/runs/{run_id}/...` (rollouts, rollout detail, search, sql, labels, decode, ws), with per-run readers constructed lazily and LRU-cached
- `/ws` (subscribe messages carry a `run_id`) and `/ws/dashboard` (pushes dashboard rows on a poll interval)

Pointing the server at a specific `log_path` still works exactly as before and does not need the registry.

## Chat

The viewer has a chat mode: ask questions about your training data in plain language and an LLM agent answers by querying the token DB for you. No SQL required. The agent can run read-only DuckDB queries (`sql`), search by regex or token-ID subsequence (`search`), pull whole trajectories (`get_rollout`), and publish self-contained HTML visuals (`publish_visual`) that render inline in the chat and in a gallery. In registry mode there is also a global cross-run chat (with `list_runs` and `dashboard` tools for comparing experiments) alongside the per-run chats.

To enable it, give the server an API key for one of the supported providers:

- Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the server's environment, or
- Configure the provider, model, and key at runtime in the UI settings (backed by `POST /api/agent/config`; the key is held in server memory only and is never written to disk or returned by the API).

Published visuals are single HTML files with inline JS/SVG (no external CDNs). For live views the visual polls the read-only SQL endpoint on an interval and re-renders in place, so a chart of, say, reward by iteration keeps updating while training runs. The files are standalone and shareable.

On-disk layout: conversations are appended as JSONL to `{log_path}/tokens/chats/{conversation_id}.jsonl` and visuals are written to `{log_path}/tokens/visuals/`. The registry-level chat stores both under the registry directory (`chats/` and `visuals/`) instead. Like everything else, this goes through the `Storage` protocol, so cloud `log_path`s work.

## Python API

No server needed. `TokenDB` reads the segment files directly:

```python
from tinker_cookbook.tokendb import TokenDB

db = TokenDB("~/runs/my-run")

# Structured filters: split, iteration ranges, reward ranges, tags, stop_reason, ...
rows = db.query(split="train", min_iteration=10, max_reward=0.0, limit=100)

# Regex over decoded text, or a contiguous token-ID subsequence
hits = db.search(regex=r"I give up", fields=("ac_text",))
hits = db.search(token_subsequence=[128000, 128006])

# All turns of one trajectory, with delta-encoded observations reconstructed
turns = db.get_rollout("train", iteration=12, group_idx=3, traj_idx=1)

# Annotations (stored in labels.jsonl, pushed live to the viewer)
db.add_label(
    {"split": "train", "iteration": 12, "group_idx": 3, "traj_idx": 1, "step_idx": 0},
    "verdict", "reward_hack", author="agent",
)
```

### Needle-in-haystack with DuckDB

`db.sql(...)` runs read-only (SELECT-only) DuckDB over the `rollouts` view. Find every rollout where a particular special token id appears, with hit counts by iteration:

```python
counts = db.sql(
    """
    SELECT iteration, count(*) AS hits
    FROM rollouts
    WHERE list_contains(ac_tokens, 128009)
    GROUP BY iteration ORDER BY iteration
    """
)
```

Or hunt by regex over the decoded action text:

```python
rows = db.sql(
    """
    SELECT iteration, group_idx, traj_idx, total_reward, ac_text
    FROM rollouts
    WHERE regexp_matches(ac_text, 'Final Answer: (?!en)')
    ORDER BY iteration
    """
)
```

Token IDs are canonical; text columns are a decoded convenience. Search for special tokens by ID (`list_contains`), since their decodings can be unstable.

## Capturing synthetic-data / sampling runs

Two opt-in paths for code outside the RL loop:

`capture_samples` observes every sample made through `TinkerTokenCompleter` / `TinkerMessageCompleter` inside the block, writing `source="sample"` rows (prompt as `ob_tokens`, sampled tokens/logprobs/stop reason as the action (`ac_*`) fields, keyword metadata into the `extra` column):

```python
from tinker_cookbook.tokendb import TokenDbWriter, capture_samples

with TokenDbWriter("~/runs/distill-data") as writer:
    with capture_samples(writer, dataset="multilingual"):
        results = await asyncio.gather(*[completer(mi, stop) for mi in prompts])
```

Code that calls `sample_async` directly can record explicitly:

```python
result = await sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
writer.record_sample(model_input, result.sequences[0], group_idx=i, tokenizer=tokenizer, prompt_id=pid)
```

See `recipes/prompt_distillation/create_data.py` (`token_db_path=...`) for a worked example.

## Storage layout

```
{log_path}/tokens/
  run.json                                    # run identity (run_id, run_attempt, model_name)
  segments/seg-{writer_id}-{seq:06d}.parquet  # immutable, one file per writer flush
  manifest-{writer_id}.jsonl                  # one line per segment
  labels.jsonl                                # annotations, appended by readers/agents
```

All I/O goes through the cookbook `Storage` protocol, so `log_path` can be local, `s3://`, or `gs://`.

Multi-writer and distributed notes:

- Segments are immutable and named by `writer_id` (hostname, pid, random suffix), so any number of processes or hosts can write to one store with no coordination. Each writer appends only to its own manifest; a segment file is fully written before its manifest line, so a crash leaves an orphan segment, never a dangling manifest entry.
- Readers treat the segments directory listing as the source of truth; manifests are a liveness hint for the viewer's push loop.
- `run.json` is coordinator-owned. `run_attempt` increments each time the coordinator restarts (a resumed run re-plays iterations), and every row is stamped with the attempt that produced it. Readers show all attempts by default, flag superseded ones, and offer a `rollouts_latest` view and `latest_only=True` for deduped queries.

## Compaction

Long runs accumulate many small segments (one per flush). Coalesce them:

```bash
python -m tinker_cookbook.tokendb.compact log_path=~/runs/my-run              # dry run (default)
python -m tinker_cookbook.tokendb.compact log_path=~/runs/my-run dry_run=False
```

Compaction rewrites all rows, sorted by `(split, iteration, group_idx, traj_idx, step_idx)`, into segments of `target_rows_per_segment` (default 65536) under a fresh `compact-...` writer ID. Rows are preserved exactly; only the file layout changes. New segments and their manifest are fully written before any old file is deleted, so a crash can leave temporary duplicates but never lose data (recovery: delete the `seg-compact-*` files from the interrupted attempt and re-run). It refuses to run if any manifest was modified within the last `min_quiet_s` seconds (default 60), a heuristic guard against compacting under a live writer; only compact runs whose training process has exited.

## Limitations

- VLM image chunks are flagged (`has_images=True`) but not stored; only text-chunk token IDs are captured.
- Eval rollout capture rides the rollout-summary export path, so it requires `rollout_json_export=True` (the default) and covers `RLTestSetEvaluator` evals; other sampling evaluators are not captured.
- With a cross-process rollout executor, filtered groups are dropped inside worker processes and do not reach the coordinator's `capture_filtered` sink (successful rollouts are unaffected).
