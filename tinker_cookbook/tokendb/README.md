# Token DB

Capture every raw token exchanged during RL rollouts (prompts, sampled tokens, per-token logprobs, rewards) into parquet segments under your run's `log_path`, then query them with the Python API or plain DuckDB SQL. Designed so that agents can query training data programmatically, not just humans clicking through HTML reports.

This package is the persistence and query layer. The interactive analysis app on top of it (web viewer, chat agent, visuals) lives in [`tinker_cookbook/tokendb_studio/`](../tokendb_studio/README.md).

## Install

The write and read paths use optional dependencies (`pyarrow`, `duckdb`):

```bash
pip install 'tinker-cookbook[tokendb]'
```

A run with `token_db` disabled never imports any of these. The viewer app needs the `tokendb-studio` extra (adds `aiohttp`); see the [studio README](../tokendb_studio/README.md).

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

The live web viewer (chat-first analysis of one run or every registered run) is the studio app; see the [studio README](../tokendb_studio/README.md) for the quickstart.

## Run registry

Every coordinator `TokenDbWriter` registers its run in a local **run registry**: one JSON file per run (so concurrent jobs never share an append target) recording `run_id`, `log_path`, `model_name`, `recipe_name`, `started_at`, `pid`, and `hostname`. This is what lets cross-run readers (and the studio's multi-run dashboard) discover every run, including concurrently running experiments, with no extra configuration.

- Default location: `~/.cache/tinker-cookbook/tokendb/runs/`
- Override with the `TINKER_TOKENDB_REGISTRY` environment variable, or per run via `TokenDbConfig(registry_dir=...)` / `TokenDbWriter(..., registry_dir=...)`
- Disable by setting either to an empty string (`TINKER_TOKENDB_REGISTRY=""` or `registry_dir=""`)

Registration is best-effort: a broken registry logs a warning and never breaks training. Worker-mode writers (explicit `run_id` / `run_attempt` in context) never register; only the coordinator does.

### Cross-run SQL (`RegistryBackend`)

`RegistryBackend` is one SQL surface spanning **all** registered runs: read-only DuckDB over cross-run `rollouts` / `rollouts_latest` / `trajectories` / `labels` / `runs` views (plus the promoted `correct` / `parse_errors` / `context_overflows`), with `run_id` as an ordinary column. This is what makes config-vs-outcome questions one query:

```sql
SELECT r.temperature, avg(t.total_reward) AS mean_reward
FROM trajectories t JOIN runs r USING (run_id, run_attempt)
GROUP BY 1 ORDER BY 1
```

The `superseded` flag is computed per run (`PARTITION BY run_id, split, iteration`), so a resume in one run never hides another run's rows; prefer `rollouts_latest` for cross-run aggregates. The same backend powers the studio's registry chat and dashboard aggregation (one `GROUP BY run_id` pass instead of a reader per run). Programmatic use:

```python
from tinker_cookbook.tokendb import RegistryBackend

backend = RegistryBackend()  # resolves the registry the same way readers do
rows = backend.sql("SELECT run_id, count(*) AS n FROM rollouts GROUP BY run_id")
```

Unlike the single-run reader, the cross-run reader is **lazy**: it keeps a DuckDB `read_parquet` scan over an explicit per-run file list (rebuilt on a TTL-gated refresh, default 5s) instead of materializing segments in memory, so nothing is pinned in RAM between queries. Segments that cannot be scanned in place are staged once into a local **segcache**:

- Cloud stores (`gs://`, `s3://`): each segment is fetched once through `Storage` and cached (segments are immutable, so existence is validity).
- v1-shaped segments are normalized to the v2 schema at cache fill (`upgraded/`); the original files are never rewritten.

The segcache defaults to `~/.cache/tinker-cookbook/tokendb/segcache`; override it with the `RegistryBackend(segcache_dir=...)` argument (the studio server exposes it as the `segcache_dir` config knob) or the `TINKER_TOKENDB_SEGCACHE` environment variable. There is no automatic eviction yet: the cache grows with the cloud/v1 segments you have actually queried, and it is always safe to delete (`rm -rf ~/.cache/tinker-cookbook/tokendb/segcache`); segments re-stage on the next refresh.

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

## Row schema

One row per turn (`schema_version = 2`):

| Column | Type | Notes |
|---|---|---|
| `run_id`, `run_attempt`, `writer_id` | string, int32, string | stamped by the writer |
| `split`, `iteration`, `group_idx`, `traj_idx`, `step_idx` | string, int32 ×4 | row identity, 1:1 with rollout-summary JSONL keys |
| `sampling_client_step` | int32, nullable | sampler weight version |
| `tags` | list\<string\> | logging tags |
| `env_row_id` | string, nullable | dataset row identity (promoted from `logs["env/row_id"]`) |
| `ts`, `source` | timestamp(us, UTC), string | `source` is `rollout` / `filtered` / `sample` |
| `ob_tokens`, `ob_is_delta` | list\<int32\>, bool | observation tokens, delta-encoded per trajectory |
| `ac_tokens`, `ac_logprobs` | list\<int32\>, list\<float32\> nullable | sampled action |
| `stop_reason` | string, nullable | null tolerated (some env adapters never set it) |
| `has_images` | bool | image chunks flagged, not stored |
| `reward`, `episode_done`, `total_reward`, `final_reward` | float32, bool, float32, float32 | trajectory-level rewards denormalized onto every row |
| `ob_text`, `ac_text` | string, nullable | decoded conveniences; token IDs are canonical |
| `metrics` | map\<string, float32\> | numeric per-row values: `Transition.metrics` keys, group-level metrics under a `group/` prefix (denormalized onto every row of the trajectory), per-turn scalar tool aggregates under `tool/`. Values are float-coerced at capture (non-coercible values dropped with a warning); NaN is stored as a real NaN. Numeric dimensions (difficulty, level, counts) belong here even when used as dimensions. |
| `attrs` | map\<string, string\> | categorical dimensions (dataset, task name, player id, ...) |
| `token_metrics` | map\<string, list\<float32\>\> | named per-token float arrays parallel to `ac_tokens`: on-policy distillation teacher logprobs (`teacher/logprobs`, multi-teacher as `teacher/<name>/logprobs`), per-token KL, token-level rewards/advantages, per-token entropy. Arrays whose length differs from `len(ac_tokens)` are dropped at capture with a warning. Empty by default. |
| `tool_calls` | list\<struct\>, nullable | per-turn tool calls: `name`, `args_json`, `error_type` (nullable), `should_stop`. No result payload: tool results are part of the next turn's observation. |
| `logs`, `extra` | string (JSON) | free-form escape hatches |
| `filtered_reason` | string, nullable | why a `filtered` row was dropped |

Query the maps directly in DuckDB: `metrics['group/rubric/score']`, `attrs['dataset'] = 'gsm8k'`.

## Populating metadata from your env

Two first-class channels feed the extensible columns; both are additive with zero-cost defaults, so envs that don't use them are unaffected.

**Group-scoped dimensions** come from `EnvGroupBuilder.metadata()` (a sibling of `logging_tags()`), written onto every row of the group. Numeric values route to the `metrics` map, strings to `attrs`, and the reserved key `row_id` promotes to the `env_row_id` column (overriding the legacy `logs["env/row_id"]` fallback):

```python
class MyGroupBuilder(EnvGroupBuilder):
    def metadata(self) -> Mapping[str, str | int | float]:
        return {
            "dataset": "gsm8k",     # -> attrs['dataset']
            "difficulty": 3,        # -> metrics['difficulty']
            "row_id": "gsm8k-42",   # -> env_row_id column
        }
```

**Per-step dimensions** go on `StepResult.attrs` (categorical, coerced to str) and `StepResult.tool_calls` (structured tool calls). On an attrs key collision, the per-step value wins over the group-level one:

```python
return StepResult(
    ...,
    attrs={"phase": "solve"},
    tool_calls=[ToolCallRecord(name="search", args_json='{"q": "..."}',
                               error_type=None, should_stop=False)],
)
```

Tool-using envs built on `AgentToolMessageEnv` populate `tool_calls` automatically (one record per call, with `error_type` set when a call fails).

Old stores keep working: v1 segments (`metrics` as a JSON string, no `attrs` / `token_metrics` / `tool_calls`) are normalized to the v2 shape when the reader loads them, so mixed-version stores query uniformly. v1 files are never rewritten.

## Capturing synthetic-data / sampling runs

Two opt-in paths for code outside the RL loop:

`capture_samples` observes every sample made through `TinkerTokenCompleter` / `TinkerMessageCompleter` inside the block, writing `source="sample"` rows (prompt as `ob_tokens`, sampled tokens/logprobs/stop reason as the action (`ac_*`) fields). Categorical dimensions (teacher model, dataset, ...) go to the typed `attrs` map and numeric values to the typed `metrics` map, so they are directly filterable in SQL; extra keyword metadata lands in the free-form `extra` JSON column:

```python
from tinker_cookbook.tokendb import TokenDbWriter, capture_samples

with TokenDbWriter("~/runs/distill-data") as writer:
    with capture_samples(
        writer,
        attrs={"teacher_model": "Qwen/Qwen3.6-35B-A3B", "source_dataset": "multilingual"},
        metrics={"temperature": 0.15},
    ):
        results = await asyncio.gather(*[completer(mi, stop) for mi in prompts])
```

Code that calls `sample_async` directly can record explicitly; the reserved attrs key `"row_id"` promotes to the `env_row_id` column (same routing as rollout capture):

```python
result = await sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
writer.record_sample(
    model_input,
    result.sequences[0],
    group_idx=i,
    tokenizer=tokenizer,
    attrs={"teacher_model": teacher_model, "source_dataset": dataset_name, "row_id": pid},
)
```

See `recipes/prompt_distillation/create_data.py` (`token_db_path=...`) for a worked `record_sample` example; the on-policy distillation and SDFT recipes (`recipes/distillation/`, `recipes/sdft/`) expose the same `token_db_path` flag and capture the student's on-policy samples through `capture_samples`.

## Storage layout

```
{log_path}/tokens/
  run.json                                    # run identity (run_id, run_attempt, model_name); latest attempt
  run-attempts.jsonl                          # one line appended per attempt (feeds the reader's `runs` view)
  segments/seg-{writer_id}-{seq:06d}.parquet  # immutable, one file per writer flush
  manifest-{writer_id}.jsonl                  # one line per segment (row counts, iteration/ts ranges, observed metrics/attrs/tag keys)
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
