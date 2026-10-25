# capture store

Local daemon that owns queryable capture data behind a small HTTP API. It is
the service boundary of the capture pipeline (see `../README.md` for the
architecture): exporters POST records in, viewers query and stream them out,
all while the run is still going. It complements, not competes with,
`tinker_cookbook/stores`: `stores/` remains the byte/run layer (Storage
protocol, run-centric JSONL), while this daemon owns structured, streamable
capture data (SQLite WAL).

## Data model

The column specs live in `schema.py` as declarative `Column` tuples, the
single source of truth: the SQLite DDL the daemon runs, the ClickHouse DDL a
hosted backend would run, and an Arrow schema for analytics export are all
rendered from them (`render_sqlite_ddl`, `render_clickhouse_ddl`,
`render_arrow_schema`), so a column is added in exactly one place and drift
between representations is a test failure (`schema_test.py`).

Two tables share one monotone cursor sequence so `/stream` interleaves them
in exact insert order:

| Table | Columns | Dedup key |
| --- | --- | --- |
| `wire_rows` | `cursor`, `run_id`, `run_attempt`, `split`, `iteration`, `group_idx`, `traj_idx`, `purpose`, `sampling_session_id`, `seq_id`, `sample_idx`, `policy_version`, `created_at`, `prompt_tokens`, `sampled_tokens`, `logprobs`, `metadata` | `(sampling_session_id, seq_id, sample_idx)` when all non-null |
| `annotations` | `cursor`, `event_id`, `run_id`, `kind`, `payload`, `created_at` | `event_id` |

Reserved scope keys map to their dedicated `wire_rows` columns; non-reserved
scope pairs (e.g. `capture(phase="eval", worker=3)`) persist inside the row's
`metadata` JSON under `metadata.scope`, so they can never collide with
request-metadata keys.

Design points a first-time reader should know:

- **No registration.** A run exists once rows carrying its `run_id` arrive;
  `GET /runs` aggregates over rows (latest attempt = `MAX(run_attempt)`).
- **Idempotent ingest.** Duplicate rows (by the dedup keys above) are
  counted, not re-inserted, so clients can retry batches safely.
- **Atomic batches.** A malformed row rolls back its whole ingest batch
  (including the cursor sequence), so a failed request leaves nothing
  partial behind.
- **Single owner per data dir.** The daemon holds an exclusive `flock` on
  `DIR/daemon.lock`; a second daemon on the same dir exits immediately.
  Discovery is via `DIR/daemon.json` (written after bind, so `--port 0`
  works), and clients verify daemon identity via `/healthz`'s
  `instance_token` so a stale discovery file can never route rows to an
  unrelated daemon that reused the port.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `POST /ingest/wire` | `{"rows": [...]}`, idempotent |
| `POST /ingest/annotations` | `{"annotations": [...]}`, idempotent |
| `GET /runs` | run listing (aggregated, no registration) |
| `GET /runs/{run_id}/rows?split=&iteration=&group_idx=&traj_idx=&purpose=&limit=&cursor=` | filtered, cursor-paged wire rows |
| `GET /runs/{run_id}/annotations?kind=&limit=&cursor=` | kind-filtered, cursor-paged annotations (train ops land here) |
| `GET /stream?run_id=&cursor=` | SSE with `id:` cursors, heartbeats, exact resume (`?cursor=` or `Last-Event-ID`) |
| `GET /touch` | activity-counting no-op; resets the idle timer (used by `ensure_daemon` to claim a near-idle daemon) |
| `GET /healthz` | liveness + identity (does not count as activity for idle shutdown) |

## Lifecycle

```bash
python -m tinker_cookbook.capture.store.daemon --data-dir DIR [--port 0] \
    [--idle-shutdown-minutes N]
```

Most callers never run that by hand:

- `ensure_daemon(data_dir)` health-checks, claims the daemon via `/touch`
  (so a daemon about to idle-shutdown cannot be handed out and exit before
  the first request), and spawns a detached daemon if needed (race-safe: the
  flock picks the single winner; losers exit and both callers converge on
  it).
- `capture_to_store(run_id, data_dir=...)` is the one-liner: it wires
  `instrument_tinker` + `CaptureExporter` + `StoreSink`, enables thread
  scope propagation (`instrument_threads=True` by default, so thread-pool
  rollout code gets attributed rows without any manual
  `propagate.instrument_threads()` call), and enters the run's capture
  scope. It is nesting-aware (on exit it restores the previously active
  exporter, and only unpatches threading if it was the one to patch it) and
  never disturbs training: if the store goes down, exports fail into
  exporter counters and the run continues.
