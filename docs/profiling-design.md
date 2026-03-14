# Unified Tracing & Profiling via Extended `@scope`

## Context

Training loops involve many overlapping async operations (sampling, forward/backward, optimizer steps, evals, checkpointing). Understanding where wall-clock time goes requires both:

1. **Per-iteration scalar metrics** (wandb charts) — "sampling got slower after step 200"
2. **Span timelines** (Perfetto / wandb Gantt) — "160 sample calls overlapped with pipelined forward/backward"

Previously, the cookbook had `@scope` for Perfetto traces and `timed()` for manual wall-clock metrics. These were completely separate systems annotating the same functions. This design unifies them: **`@scope` is the single source of truth** for both Perfetto traces and wandb timing metrics.

## API

### Setup

```python
from tinker_cookbook.utils.trace import trace_init, trace_iteration

ml_logger = ml_log.setup_logging(log_dir=cfg.log_path, wandb_project=cfg.wandb_project, ...)

trace_init(
    output_file="trace_events.jsonl",    # Perfetto trace (existing)
    wandb_logger=ml_logger,              # Auto-log timing metrics per iteration
    span_chart_every=20,                 # Log Plotly Gantt chart every N steps
)
```

`trace_init` also auto-instruments Tinker SDK client classes (`TrainingClient`, `SamplingClient`) with `@scope`, so API calls like `forward_backward_async` and `sample_async` appear in both Perfetto traces and wandb timing metrics without any manual annotation.

### Training loop

```python
for i_batch in range(n_batches):
    metrics = {"progress/batch": i_batch, ...}

    with trace_iteration(step=i_batch):
        await run_evals(...)           # @scope already on this
        await gather_rollouts(...)     # @scope on each rollout
        await train_step(...)          # @scope already on this
        metrics.update(train_step_metrics)

    # time/* metrics auto-logged by trace_iteration
    ml_logger.log_metrics(metrics, step=i_batch)
```

### What gets logged

`trace_iteration` aggregates all `@scope` spans within the window:

- **Single call**: `time/run_evaluations_parallel = 4.2`
- **Repeated calls** (e.g., 160 `sample_async` in parallel): `time/sample_async:total`, `time/sample_async:count`, `time/sample_async:mean`, `time/sample_async:max`
- **Total**: `time/total` (wall-clock of the entire iteration)

Span names are flat (`func.__name__`), not hierarchical. If `train_step` calls `forward_backward_async`, both appear as separate top-level keys with inclusive timing. This keeps the output simple and avoids ambiguity about parent/child attribution.

Every N steps (configurable via `span_chart_every`), a Plotly Gantt chart is logged showing span overlap — you can see concurrent sample calls, pipelined forward/backward + optim_step, sequential evals, all interactive in the wandb dashboard.

## Architecture

### Before

```
@scope ─────────── Perfetto JSONL ──────── Perfetto UI (engineer debugging)
timed() ─────────── metrics dict ──────── wandb (researcher monitoring)
@profiled (PR #420) ─ metrics dict ──────── wandb (duplicate of above)
```

### After

```
@scope ─┬─ Perfetto JSONL ────────────── Perfetto UI
        ├─ IterationWindow ──┬─ timing dict ──── wandb scalar charts
        │                    └─ Gantt chart ───── wandb Plotly panels
        └─ SDK client patches ── API call spans in both outputs
```

### Key components

- **`IterationWindow`** — Collects `(name, start_time, end_time)` span records during one iteration. Produces aggregated timing dict and raw span records for Gantt charts.
- **`trace_iteration(step)`** — Context manager that creates an `IterationWindow`, makes it available via `ContextVar`, and on exit aggregates + auto-logs.
- **`_instrument_sdk_clients()`** — Called by `trace_init()`. Patches `TrainingClient` and `SamplingClient` class methods with `@scope` so all instances are automatically traced.
- **`@scope` (extended)** — In addition to emitting Perfetto events, records spans into the active `IterationWindow` if one exists.

### Concurrency

`IterationWindow.record_span()` is thread-safe (uses `threading.Lock`). The `_iteration_window` `ContextVar` is inherited by child async tasks created within the window, so `asyncio.gather()` and `create_task()` work correctly.

## Relationship to existing systems

| System | Status | Purpose |
|---|---|---|
| `@scope` | Extended | Perfetto traces + wandb metrics + Gantt charts |
| `timed()` | Kept | Dynamic keys (`timed(f"train/substep_{i}", metrics)`) |
| `@profiled` (PR #420) | Superseded | Was a parallel system; now subsumed by `@scope` + `trace_iteration` |

## Limitations

- **Gantt charts require wandb** — the Plotly chart is logged via `wandb.log()` directly (not through the `Logger` abstraction) since `wandb.Plotly()` is a wandb-specific type. Other logging backends (JSON, Neptune) receive scalar timing metrics but not charts.
- **Multiprocess workers are invisible** — when rollouts run in `ProcessPoolExecutor` or Ray, child processes don't inherit `ContextVar` state. The parent sees wall-clock time for `await run_in_executor(...)` but not internal spans. Cross-process spans can be shipped back via `IterationWindow.merge_spans()` (see below).
- **`SpanRecord` stores dual timestamps** — `perf_counter` (high-res, process-local) for durations and `time.time()` (cross-process) for Gantt chart positioning. This enables future cross-process span merging without clock alignment issues.

## Future: cross-process span collection

Workers can collect `SpanRecord`s locally and return them alongside results. The parent merges them before the iteration window closes:

```python
window.merge_spans(worker_span_records)
```

Since `SpanRecord.wall_start`/`wall_end` use `time.time()` (synchronized across processes on the same machine), merged spans are correctly positioned on the Gantt chart timeline.

## Dependencies

- **plotly** (optional) — for Gantt charts. Skipped gracefully if not installed.
- **wandb** (optional) — gated by `wandb_logger` parameter.
- No new required dependencies.
