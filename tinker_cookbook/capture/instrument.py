"""Instrumentation of the Tinker SDK public surface.

``instrument_tinker(exporter)`` monkeypatches a small, stable set of public
SDK methods so that every call is tagged with the ambient capture scope
(:mod:`tinker_cookbook.capture.scope`) and its request/outcome metadata is
enqueued to a :class:`~tinker_cookbook.capture.exporter.CaptureExporter`.

Patched methods (public surface only):

- ``tinker.SamplingClient.sample`` / ``sample_async``
- ``tinker.TrainingClient.forward_backward`` / ``optim_step`` /
  ``save_weights_for_sampler`` and their ``*_async`` variants (the training
  loops in ``rl/train.py`` and ``supervised/train.py`` call the async entry
  points).

Rules:

- The scope AND the active exporter are snapshotted synchronously on the
  caller thread at call time, never at future-completion time, so records
  attribute correctly (and go to the right sink) even when the future
  outlives the ``capture(...)`` block or the exporter is swapped while
  requests are outstanding. For async methods the outer wrapper is a plain
  sync function that snapshots immediately and returns an inner coroutine
  closed over the snapshot, so building a list of coroutines inside a scope
  and ``gather``-ing later still attributes to the scope at creation time.
- Outcome attachment is enqueue-only and never raises: any failure inside a
  done-callback increments ``callback_failures`` on the snapshotted exporter
  and is otherwise swallowed.
- Every in-flight instrumented call is tracked on its exporter
  (``track_pending``/``pending_done``) so teardown can grace-drain
  outstanding futures via ``exporter.wait_pending(timeout)``.
- Patching is idempotent (double-instrument is a no-op) and reversible
  (``uninstrument_tinker`` restores the original methods).

Precedent: the SDK's own ``lib/telemetry.py`` and
``tinker_cookbook.utils.trace._instrument_sdk_clients`` wrap public client
methods via decorators/setattr in the same way.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import CancelledError, Future
from contextvars import ContextVar
from typing import Any

# Typing note: this module wraps SDK methods it deliberately treats as opaque
# (real tinker clients in production, lightweight fakes in tests), so the
# values flowing through the wrappers are typed ``object`` and inspected only
# via getattr; ``Any`` remains where a value must be called or unpacked
# (wrapped method signatures, free-form JSON metadata), which is the honest
# annotation at that boundary.
from tinker_cookbook.capture.exporter import CaptureExporter, CaptureRecord
from tinker_cookbook.capture.scope import current_scope

_ORIGINAL_ATTR = "_capture_original"

# Re-entrancy guard: the SDK's async methods delegate to their instrumented
# sync counterparts SYNCHRONOUSLY inside the coroutine body (e.g.
# ``sample_async`` awaits ``AwaitableConcurrentFuture(self.sample(...))``,
# and every ``*_async`` train op does the same), so without this flag each
# async call would be recorded twice: once by the async wrapper and once by
# the delegated sync wrapper. The flag is a ContextVar: the delegation runs
# on the same task/context as the outer wrapper's coroutine, so the inner
# call sees it, while independent concurrent calls each run in their own
# task context (contextvars are per-task) and are never suppressed.
_capture_in_progress: ContextVar[bool] = ContextVar("capture_in_progress", default=False)

# (class, method_name) -> original function, populated by instrument_tinker.
_patched: dict[tuple[type, str], Callable[..., Any]] = {}
_exporter: CaptureExporter | None = None


def _enqueue_to(exporter: CaptureExporter | None, record: CaptureRecord) -> None:
    if exporter is not None:
        exporter.enqueue(record)


def _count_failure(exporter: CaptureExporter | None) -> None:
    if exporter is not None:
        exporter.callback_failures += 1


def _track(exporter: CaptureExporter | None) -> None:
    if exporter is not None:
        exporter.track_pending()


def _untrack(exporter: CaptureExporter | None) -> None:
    if exporter is not None:
        exporter.pending_done()


def _sample_request_metadata(
    exporter: CaptureExporter | None,
    client: object,
    prompt: object,
    num_samples: int,
    sampling_params: object,
) -> dict[str, Any]:
    """Best-effort request metadata for a sample call. Never raises."""
    meta: dict[str, Any] = {"num_samples": num_samples}
    try:
        meta["sampling_session_id"] = getattr(client, "_sampling_session_id", None)
        # Stable per-call identifier for the store's
        # (sampling_session_id, seq_id, sample_idx) dedupe key. The SDK stamps
        # its own seq_id from a private per-client counter, but reading that
        # counter here races concurrent calls on the same client, so we mint
        # our own: generated once at call time, it stays stable across
        # exporter retries/replays of this record, which is all dedupe needs.
        meta["seq_id"] = int(uuid.uuid4()) & ((1 << 62) - 1)
        for attr in ("model_path", "base_model"):
            value = getattr(client, attr, None)
            if value is not None:
                meta[attr] = str(value)
        to_ints: Callable[[], Sequence[int]] | None = getattr(prompt, "to_ints", None)
        if callable(to_ints):
            meta["prompt_tokens"] = list(to_ints())
            meta["prompt_length"] = len(meta["prompt_tokens"])
        else:
            meta["prompt_length"] = getattr(prompt, "length", None)
        if sampling_params is not None:
            meta["sampling_params"] = {
                key: getattr(sampling_params, key, None)
                for key in ("max_tokens", "temperature", "top_p", "top_k", "seed")
            }
    except Exception:
        _count_failure(exporter)
    return meta


def _sample_outcome(record: CaptureRecord, response: object) -> None:
    """Attach sampled tokens / logprobs / stop reasons to the record. Never raises."""
    sequences = getattr(response, "sequences", None) or []
    samples: list[dict[str, Any]] = []
    for seq in sequences:
        tokens = getattr(seq, "tokens", None)
        logprobs = getattr(seq, "logprobs", None)
        stop_reason = getattr(seq, "stop_reason", None)
        samples.append(
            {
                "tokens": list(tokens) if tokens is not None else None,
                "logprobs": list(logprobs) if logprobs is not None else None,
                "stop_reason": str(stop_reason) if stop_reason is not None else None,
            }
        )
    record["samples"] = samples
    record["status"] = "ok"


def _finalize(record: CaptureRecord, started: float, error: BaseException | None) -> None:
    record["latency_sec"] = time.perf_counter() - started
    if error is not None:
        record["status"] = "error"
        record["error"] = repr(error)


def _attach_done_callback(future: object, on_done: Callable[[Future[object]], None]) -> bool:
    """Attach ``on_done`` to a future-like object.

    Handles both a ``concurrent.futures.Future`` (``sample``) and the SDK's
    ``APIFuture`` wrapper, which exposes only ``result``/``result_async`` but
    wraps an inner concurrent future as ``_future``.
    """
    add_cb = getattr(future, "add_done_callback", None)
    if add_cb is None:
        inner = getattr(future, "_future", None)
        add_cb = getattr(inner, "add_done_callback", None)
    if add_cb is None:
        return False
    add_cb(on_done)
    return True


def _attach_outcome_or_finish(
    exporter: CaptureExporter | None,
    future: object,
    record: CaptureRecord,
    started: float,
    on_result: Callable[[CaptureRecord, object], None],
) -> None:
    """Attach the outcome done-callback, or emit a request-only record.

    ``exporter`` was tracked (``track_pending``) by the caller; this function
    guarantees exactly one matching ``pending_done``.
    """

    def on_done(fut: Future[object]) -> None:
        try:
            try:
                try:
                    error = fut.exception()
                except CancelledError:
                    # fut.exception() raises for cancelled futures; record the
                    # cancellation instead of dropping the call.
                    _record_cancellation(exporter, record, started)
                    return
                if error is None:
                    on_result(record, fut.result())
                _finalize(record, started, error)
                _enqueue_to(exporter, record)
            except Exception:
                _count_failure(exporter)
        finally:
            _untrack(exporter)

    if not _attach_done_callback(future, on_done):
        _finalize(record, started, None)
        record["status"] = "submitted"
        _enqueue_to(exporter, record)
        _untrack(exporter)


# Live never-started-coroutine guards (see _guard_unstarted_coroutine).
# uninstrument_tinker() resolves any still outstanding so a Task retained in
# a registry cannot hold a guard past exporter shutdown, where its cancelled
# record would only be a counted drop.
_unstarted_guards: set[weakref.finalize] = set()

# Serializes async-wrapper prologues (exporter snapshot + guard registration)
# with uninstrument_tinker()'s restore/clear/flush transition. Without it, a
# wrapper entered just before uninstrument could snapshot the live exporter
# but register its guard after the flush, recreating the counted-drop loss
# the registry exists to prevent. RLock: registration happens while the
# prologue already holds the lock.
_guards_lock = threading.RLock()


def _detach_guard(finalizer: weakref.finalize) -> None:
    """Detach a guard at first poll and drop it from the live registry."""
    finalizer.detach()
    _unstarted_guards.discard(finalizer)


def flush_unstarted_guards() -> None:
    """Resolve all outstanding never-started guards NOW.

    Called by :func:`uninstrument_tinker` (i.e. before the teardown path
    shuts the exporter down) so cancelled-before-first-poll calls whose Task
    objects are still retained get their records enqueued while the
    snapshotted exporter is alive, and the guards release their references.
    In the rare case a flushed coroutine is polled afterwards, its normal
    record is emitted too; a visible duplicate beats a silent loss.
    """
    with _guards_lock:
        for finalizer in list(_unstarted_guards):
            finalizer()  # fires the callback once; no-op if already fired
        _unstarted_guards.clear()


def _guard_unstarted_coroutine(
    coro: Coroutine[Any, Any, Any],
    exporter: CaptureExporter | None,
    record: CaptureRecord,
    started: float,
) -> weakref.finalize:
    """Emit a cancelled record for a coroutine that is never first-polled.

    A task cancelled before its first poll (immediate teardown) closes the
    inner coroutine WITHOUT ever executing its body, so no in-coroutine
    handler can observe it. A ``weakref.finalize`` on the coroutine object
    fires when it is reclaimed and records the call as
    ``status="cancelled"``. The same accounting applies to a coroutine that
    is created and simply discarded. Pending tracking stays consistent: an
    unstarted body never called ``_track``, so there is no slot to release.

    The wrapper DETACHES the returned finalizer at the coroutine's first
    poll: from then on the in-coroutine handlers cover every outcome, and a
    live finalizer would strongly retain ``record`` (prompt tokens included)
    and the exporter for as long as the caller keeps the completed Task.
    """

    def on_reclaimed() -> None:
        _unstarted_guards.discard(finalizer)
        _record_cancellation(exporter, record, started)

    finalizer = weakref.finalize(coro, on_reclaimed)
    with _guards_lock:
        _unstarted_guards.add(finalizer)
    return finalizer


def _record_cancellation(
    exporter: CaptureExporter | None, record: CaptureRecord, started: float
) -> None:
    """Enqueue a ``status="cancelled"`` record. Never raises."""
    try:
        _finalize(record, started, None)
        record["status"] = "cancelled"
        _enqueue_to(exporter, record)
    except Exception:
        _count_failure(exporter)


def _record_submission_failure(
    exporter: CaptureExporter | None,
    record: CaptureRecord,
    started: float,
    error: BaseException,
) -> None:
    """Enqueue an error record for a call that raised before returning a future."""
    try:
        _finalize(record, started, error)
        _enqueue_to(exporter, record)
    except Exception:
        _count_failure(exporter)


def _train_op_status(record: CaptureRecord, result: Any) -> None:
    record["status"] = "ok"
    # Bounded outcome summary: scalar metrics only (ForwardBackwardOutput
    # exposes ``metrics: dict[str, float]``); never tensors or per-datum
    # outputs. Other op results have no ``metrics`` and are skipped.
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, dict):
        record["metrics"] = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }


def _make_sample_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(
        self: object,
        prompt: object,
        num_samples: int,
        sampling_params: object,
        *args: object,
        **kwargs: object,
    ) -> Any:
        if _capture_in_progress.get():
            # Delegated call from an already-recording wrapper (see
            # _capture_in_progress): pass straight through, no record.
            return original(self, prompt, num_samples, sampling_params, *args, **kwargs)
        exporter = _exporter
        record: CaptureRecord = {
            "kind": "sample",
            "scope": dict(current_scope()),
            "created_at": time.time(),
            **_sample_request_metadata(exporter, self, prompt, num_samples, sampling_params),
        }
        started = time.perf_counter()
        # Track BEFORE invoking the SDK so a teardown that calls
        # wait_pending() while the submission is still executing sees it as
        # in flight (tracking after the return would leave a window where
        # pending reads zero mid-submission).
        _track(exporter)
        reentry = _capture_in_progress.set(True)
        try:
            future = original(self, prompt, num_samples, sampling_params, *args, **kwargs)
        except BaseException as e:
            # The SDK raised before returning a future (validation error,
            # stopped client, ...): emit an error record, then re-raise.
            _record_submission_failure(exporter, record, started, e)
            _untrack(exporter)
            raise
        finally:
            _capture_in_progress.reset(reentry)
        _attach_outcome_or_finish(exporter, future, record, started, _sample_outcome)
        return future

    return wrapper


def _as_coroutine_function(
    starter: Callable[..., Any], original: Callable[..., Any]
) -> Callable[..., Any]:
    """Make ``starter`` (sync fn returning a coroutine) detectable as async.

    On Python 3.12+ ``inspect.markcoroutinefunction`` marks the sync starter
    directly. On 3.11 there is no such marker and no way to get both a real
    call-time body AND ``inspect.iscoroutinefunction`` detection, so we keep
    the sync starter unmarked there: call-time scope/exporter snapshotting
    (the core guarantee of this module) is preserved on every version, and
    the one in-repo consumer of coroutine detection (``utils/trace.py``'s
    ``@scope`` dispatch) is handled explicitly by the ordering integration in
    :func:`instrument_tinker`, which prevents trace from re-wrapping these
    methods at all.
    """
    del original
    mark = getattr(inspect, "markcoroutinefunction", None)
    return mark(starter) if mark is not None else starter


def _make_sample_async_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    # The outer wrapper is SYNC: it snapshots scope + exporter immediately at
    # call time and returns an inner coroutine closed over the snapshot, so
    # the create-coroutines-now-gather-later pattern attributes correctly.
    @functools.wraps(original)
    def wrapper(
        self: object,
        prompt: object,
        num_samples: int,
        sampling_params: object,
        *args: object,
        **kwargs: object,
    ) -> Any:
        # The lock spans the exporter snapshot through guard registration
        # (bottom of this wrapper) so uninstrument's flush cannot fall
        # between them; see _guards_lock.
        with _guards_lock:
            exporter = _exporter
            record: CaptureRecord = {
                "kind": "sample",
                "scope": dict(current_scope()),
                "created_at": time.time(),
                **_sample_request_metadata(exporter, self, prompt, num_samples, sampling_params),
            }
            started = time.perf_counter()
            guard: list[weakref.finalize | None] = [None]

            async def inner() -> Any:
                # First poll: the in-coroutine handlers cover every outcome from
                # here, so detach the unstarted-coroutine guard (releasing its
                # strong refs to record/exporter for callers that retain the
                # completed Task). Track only once the coroutine actually
                # starts: a coroutine that is created but never awaited must not
                # leak a pending slot and stall wait_pending().
                if guard[0] is not None:
                    _detach_guard(guard[0])
                _track(exporter)
                try:
                    # The SDK's sample_async calls self.sample() synchronously
                    # during this await's first poll, on THIS context: the
                    # flag makes that delegated (also-instrumented) call pass
                    # through instead of double-recording.
                    reentry = _capture_in_progress.set(True)
                    try:
                        response = await original(
                            self, prompt, num_samples, sampling_params, *args, **kwargs
                        )
                    except asyncio.CancelledError:
                        # Routine task cancellation (rollout timeout, task-group
                        # teardown) is not an SDK failure: record it as
                        # "cancelled" (matching the sync future path) and
                        # re-raise so cancellation semantics are preserved.
                        _record_cancellation(exporter, record, started)
                        raise
                    except BaseException as e:
                        try:
                            _finalize(record, started, e)
                            _enqueue_to(exporter, record)
                        except Exception:
                            _count_failure(exporter)
                        raise
                    finally:
                        _capture_in_progress.reset(reentry)
                    try:
                        _sample_outcome(record, response)
                        _finalize(record, started, None)
                        _enqueue_to(exporter, record)
                    except Exception:
                        _count_failure(exporter)
                    return response
                finally:
                    _untrack(exporter)

            coro = inner()
            guard[0] = _guard_unstarted_coroutine(coro, exporter, record, started)
            return coro

    return _as_coroutine_function(wrapper, original)


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    """Resolve a call argument passed either by keyword or positionally.

    The SDK's train ops are commonly called keyword-style
    (``save_weights_for_sampler(name=...)``, ``forward_backward(data=...)``);
    inspecting only positional args would silently lose the metadata.
    """
    if name in kwargs:
        return kwargs[name]
    return args[index] if index < len(args) else None


_ADAM_PARAM_FIELDS = ("learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")


def _train_op_request_record(
    exporter: CaptureExporter | None,
    client: object,
    op_name: str,
    # Any (not object): the resolved arguments are duck-typed below.
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CaptureRecord:
    record: CaptureRecord = {
        "kind": "train_op",
        "op": op_name,
        "scope": dict(current_scope()),
        "created_at": time.time(),
    }
    try:
        model_id = getattr(client, "model_id", None)
        if model_id is not None:
            record["model_id"] = str(model_id)
        if op_name == "forward_backward":
            data = _arg(args, kwargs, 0, "data")
            if data is not None:
                record["num_data"] = len(data)
        elif op_name == "save_weights_for_sampler":
            name = _arg(args, kwargs, 0, "name")
            if name is not None:
                record["name"] = str(name)
        elif op_name == "optim_step":
            adam_params = _arg(args, kwargs, 0, "adam_params")
            if adam_params is not None:
                record["hyperparams"] = {
                    field: getattr(adam_params, field, None) for field in _ADAM_PARAM_FIELDS
                }
    except Exception:
        _count_failure(exporter)
    return record


def _make_train_op_wrapper(original: Callable[..., Any], op_name: str) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: object, *args: Any, **kwargs: object) -> Any:
        if _capture_in_progress.get():
            # Delegated call from an already-recording wrapper (the SDK's
            # *_async train ops call their sync counterparts): pass through.
            return original(self, *args, **kwargs)
        exporter = _exporter
        record = _train_op_request_record(exporter, self, op_name, args, dict(kwargs))
        started = time.perf_counter()
        # Track before invoking the SDK (see _make_sample_wrapper).
        _track(exporter)
        reentry = _capture_in_progress.set(True)
        try:
            future = original(self, *args, **kwargs)
        except BaseException as e:
            _record_submission_failure(exporter, record, started, e)
            _untrack(exporter)
            raise
        finally:
            _capture_in_progress.reset(reentry)
        _attach_outcome_or_finish(exporter, future, record, started, _train_op_status)
        return future

    return wrapper


def _make_train_op_async_wrapper(original: Callable[..., Any], op_name: str) -> Callable[..., Any]:
    # Sync outer (snapshot at call time) returning an inner coroutine; the
    # awaited original returns an APIFuture whose completion carries the
    # outcome, exactly like the sync variant.
    @functools.wraps(original)
    def wrapper(self: object, *args: Any, **kwargs: object) -> Any:
        # Locked prologue-through-registration; see _make_sample_async_wrapper.
        with _guards_lock:
            exporter = _exporter
            record = _train_op_request_record(exporter, self, op_name, args, dict(kwargs))
            started = time.perf_counter()
            guard: list[weakref.finalize | None] = [None]

            async def inner() -> Any:
                # See _make_sample_async_wrapper: detach the unstarted guard and
                # track at first poll so an un-awaited coroutine cannot leak a
                # pending slot.
                if guard[0] is not None:
                    _detach_guard(guard[0])
                _track(exporter)
                # See _make_sample_async_wrapper: the SDK's *_async train ops
                # delegate to their instrumented sync counterparts on this
                # same context; the flag makes that inner call pass through.
                reentry = _capture_in_progress.set(True)
                try:
                    future = await original(self, *args, **kwargs)
                except asyncio.CancelledError:
                    # See _make_sample_async_wrapper: record as "cancelled",
                    # never as an SDK error, and re-raise.
                    _record_cancellation(exporter, record, started)
                    _untrack(exporter)
                    raise
                except BaseException as e:
                    try:
                        _finalize(record, started, e)
                        _enqueue_to(exporter, record)
                    except Exception:
                        _count_failure(exporter)
                    _untrack(exporter)
                    raise
                finally:
                    _capture_in_progress.reset(reentry)
                # _attach_outcome_or_finish takes over the pending slot.
                _attach_outcome_or_finish(exporter, future, record, started, _train_op_status)
                return future

            coro = inner()
            guard[0] = _guard_unstarted_coroutine(coro, exporter, record, started)
            return coro

    return _as_coroutine_function(wrapper, original)


def _methods_to_patch() -> list[
    tuple[type, str, Callable[[Callable[..., Any]], Callable[..., Any]]]
]:
    import tinker

    def train_op(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda f: _make_train_op_wrapper(f, name)

    def train_op_async(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda f: _make_train_op_async_wrapper(f, name)

    entries: list[tuple[type, str, Callable[[Callable[..., Any]], Callable[..., Any]]]] = [
        (tinker.SamplingClient, "sample", _make_sample_wrapper),
        (tinker.SamplingClient, "sample_async", _make_sample_async_wrapper),
    ]
    for op in ("forward_backward", "optim_step", "save_weights_for_sampler"):
        entries.append((tinker.TrainingClient, op, train_op(op)))
        async_name = f"{op}_async"
        if hasattr(tinker.TrainingClient, async_name):
            entries.append((tinker.TrainingClient, async_name, train_op_async(op)))
    return entries


def current_exporter() -> CaptureExporter | None:
    """The exporter new instrumented calls snapshot, or None if uninstrumented.

    Lets nesting-aware wiring (e.g. ``capture_to_store``) save the active
    exporter on entry and restore it on exit instead of tearing down
    instrumentation that an enclosing context still relies on.
    """
    return _exporter


def instrument_tinker(exporter: CaptureExporter) -> None:
    """Patch the stable public Tinker SDK methods to emit capture records.

    Idempotent: repeated calls only swap the active exporter; already-patched
    methods are not wrapped twice. Calls that are already in flight keep the
    exporter they snapshotted at call time.

    Args:
        exporter: Destination for capture records.
    """
    global _exporter
    _exporter = exporter
    # Ordering integration with utils/trace.py, which patches overlapping SDK
    # methods with @scope at trace_init() time. Apply trace's (idempotent,
    # no-op-when-tracing-inactive) instrumentation FIRST so our wrapper is
    # always the outer layer, then mark our wrapper _scope_instrumented so a
    # later trace_init() will not re-wrap it (which would defer our call-time
    # snapshot to first poll). Result is correct regardless of whether
    # trace_init() runs before or after instrument_tinker().
    from tinker_cookbook.utils.trace import _instrument_sdk_clients

    _instrument_sdk_clients()
    # Thread scope propagation is deliberately NOT coupled here: it is a
    # separate concern owned by capture.propagate, and the store integration
    # (capture_to_store) enables it by default with nesting-aware restore.
    # Direct instrument_tinker users opt in via propagate.instrument_threads().
    for cls, name, make_wrapper in _methods_to_patch():
        current = getattr(cls, name)
        if getattr(current, _ORIGINAL_ATTR, None) is not None:
            continue  # already instrumented
        wrapped = make_wrapper(current)
        setattr(wrapped, _ORIGINAL_ATTR, current)
        # Prevents trace_init() re-wrap (ordering note above); setattr keeps
        # pyright happy about dynamic function attributes.
        setattr(wrapped, "_scope_instrumented", True)  # noqa: B010
        _patched[(cls, name)] = current
        setattr(cls, name, wrapped)


def uninstrument_tinker() -> None:
    """Restore all patched SDK methods. Idempotent.

    In-flight calls keep their snapshotted exporter, so records for futures
    that are still outstanding are not lost; drain them with
    ``exporter.wait_pending(...)`` followed by ``exporter.shutdown()``.
    """
    global _exporter
    for (cls, name), original in list(_patched.items()):
        setattr(cls, name, original)
        del _patched[(cls, name)]
    # Under the guards lock: clear the exporter FIRST, then resolve the
    # never-started guards while their snapshotted exporters are still alive
    # (teardown shuts the exporter down after uninstrumenting). Any wrapper
    # prologue serialized after this transition snapshots None and its guard
    # is a harmless no-op; any serialized before has already registered its
    # guard and is flushed here.
    with _guards_lock:
        _exporter = None
        flush_unstarted_guards()
