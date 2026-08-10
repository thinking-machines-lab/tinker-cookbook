# capture

Token-level capture of training and sampling traffic, with near-zero changes
to recipe code. Wrap a training loop in one context manager (or point an
external agent harness at a local proxy) and every sampled sequence, with its
prompt tokens, sampled tokens, logprobs, and run coordinates, lands in a
queryable local store while the run is still going.

## Why

Debugging RL and SFT runs usually means asking "what did the model actually
see and produce at iteration N, trajectory K?" and answering it from ad-hoc
print statements or JSONL scraps. Capture makes that data a first-class,
addressable artifact: rows are tagged with their run coordinates at the
moment the SDK call is made, exported in the background, and queryable or
streamable immediately. The pipeline borrows deliberately from public prior
art: OpenTelemetry (ambient context propagation, batch processors) and
experiment trackers like Weights & Biases (a background daemon owning the
data; training code never blocks on it).

## Architecture

```
your training loop                     external agent harness
  with capture(...):                     (Claude Code, opencode, ...)
    sampling_client.sample(...)            ANTHROPIC_BASE_URL=...proxy...
        |                                        |
        | instrumented SDK                       | capture/proxy/
        | (capture/instrument.py)                | renders chat -> tokens,
        |                                        | samples via the same
        |                                        | instrumented SDK
        v                                        v
  CaptureExporter (bounded queue, background flusher)
        |
        v
  capture/store/ daemon (SQLite WAL, HTTP ingest / query / SSE)
```

- **`capture/` (this directory): in-process capture.** `capture(**pairs)`
  pushes an ambient scope (`run_id`, `iteration`, `traj_idx`, ...) onto a
  ContextVar; `instrument_tinker(exporter)` patches the stable public Tinker
  SDK methods (`SamplingClient.sample`/`sample_async`,
  `TrainingClient.forward_backward`/`optim_step`/`save_weights_for_sampler`
  and their `_async` variants) so every call is tagged with the ambient
  scope and exported through a batching `CaptureExporter`.
- **`capture/store/`: the local store daemon.** A per-data-dir daemon that
  ingests records over HTTP into SQLite and serves queries and SSE
  streaming. See `store/README.md`.
- **`capture/proxy/`: the chat API proxy.** An Anthropic/OpenAI-compatible
  endpoint for harnesses that cannot use the SDK; it renders chat to tokens
  and samples through the instrumented SDK, so its traffic funnels through
  the exact same path. See `proxy/README.md`.

## Quickstart

```python
from tinker_cookbook.capture.store import capture_to_store

with capture_to_store("my-run", data_dir="~/.cache/tinker-capture"):
    # any SDK sampling/training calls in here are captured
    run_training()
```

`capture_to_store` spawns (or reuses) the store daemon, instruments the SDK,
turns on thread scope propagation (`instrument_threads=True` by default; no
manual `propagate.instrument_threads()` call needed), and enters the run's
scope; on exit it restores the previous instrumentation state (thread
patches included, nesting-aware) and drains the exporter. For process pools
see the fork-vs-spawn notes under Limitations. Without the store daemon, wire the pieces
directly:

```python
from tinker_cookbook.capture import CaptureExporter, JsonlFileSink, capture, instrument_tinker, uninstrument_tinker
from tinker_cookbook.stores.storage import LocalStorage

exporter = CaptureExporter(JsonlFileSink(LocalStorage("/tmp/my-run")))
instrument_tinker(exporter)

with capture(run_id="my-run"):
    run_training()

# Teardown order matters: stop creating records first (this also resolves
# guards for never-started coroutines), drain outstanding futures, THEN shut
# the exporter down. Shutting down first would count late completions as
# drops. capture_to_store does exactly this for you.
uninstrument_tinker()
exporter.wait_pending(timeout=5.0)
exporter.shutdown()
```

Add finer-grained coordinates anywhere below:

```python
from tinker_cookbook.capture import capture

with capture(iteration=i, group_idx=g, traj_idx=t):
    await sampling_client.sample_async(...)
```

## Design principles

- **Capture never blocks or breaks training.** The exporter queue is bounded
  and drop-newest (with a `dropped` counter); sink failures are counted, not
  raised; a dead store daemon degrades to counters, never to a training
  crash.
- **Snapshot at call time.** The scope AND the active exporter are
  snapshotted synchronously when the SDK method is invoked, never when its
  future completes. Futures routinely outlive the `with capture(...)` block
  that issued them; snapshotting at completion would attribute results to
  whatever scope happens to be active then (often none). For async methods
  the outer wrapper is a plain sync function that snapshots immediately and
  returns an inner coroutine, so coroutines created inside a scope and
  gathered later still attribute to creation time.
- **In-flight calls are tracked.** Every instrumented call holds a pending
  slot on its exporter from before submission until its outcome is recorded,
  so teardown can grace-drain outstanding futures
  (`exporter.wait_pending(...)`).
- **Scopes nest and isolate.** Inner `capture(...)` blocks merge over outer
  ones and restore exactly on exit; ContextVars give each asyncio task and
  each thread its own view. `capture.propagate.instrument_threads()` patches
  `threading.Thread` and `ThreadPoolExecutor.submit` so scope reaches worker
  threads automatically, including calls through `asyncio.run_in_executor`
  (OTel `ThreadingInstrumentor`-shaped); the one-context-manager
  integrations built on this module turn it on by default.

## Scope keys

Any JSON-scalar key is accepted; these have conventional meanings across the
pipeline:

| Key | Meaning |
| --- | --- |
| `run_id` | stable identifier for the training run |
| `run_attempt` | restart counter for the run |
| `split` | e.g. `train` / `test` |
| `iteration` | training iteration / step index |
| `group_idx` | trajectory-group index within the iteration |
| `traj_idx` | trajectory index within the group |
| `purpose` | free-form tag, e.g. `rollout` or `eval` |

## Exporter

`CaptureExporter` is an OpenTelemetry-BatchProcessor-shaped pipeline:
bounded in-memory queue, background flusher thread, flush on batch size or
timer, `force_flush(timeout)`, idempotent `shutdown(timeout)` (also
registered with `atexit`). Sinks are pluggable: `JsonlFileSink` appends
`capture/<kind>.jsonl` under a run directory; `store.StoreSink` POSTs to the
store daemon. Enqueue and shutdown are serialized so a record enqueued
concurrently with teardown is either flushed or counted dropped, never
silently stranded.

## Limitations

- **Process pools: fork vs spawn.** With the `fork` start method the
  capture pipeline survives into children BY DESIGN: forked children inherit
  the patched SDK methods, the ambient scope (fork copies contextvar
  memory), and the module-global exporter, whose `os.register_at_fork` hook
  restarts the flusher thread in the child. Inherited queued records are
  discarded without being counted as drops (the parent still owns and
  exports them, so they are not losses), and the child's loss counters are
  reset to zero so it reports only its own lifetime; records enqueued in
  the child export normally. With `spawn`, children start clean (no
  patches, no scope); run a separate `capture_to_store` inside the worker.
  The reason to keep Tinker SDK calls in the parent is an SDK limitation,
  not a capture one: the SDK's background event-loop thread does not survive
  fork, so creating or using SDK clients in forked children can deadlock.
- **Python 3.11:** `inspect.iscoroutinefunction` reports False for the
  patched async methods (there is no `markcoroutinefunction` before 3.12);
  call-time snapshotting still holds on every version.
- **Single machine.** The store daemon and its data dir are local; capture
  from distributed workers lands in each machine's own store.
