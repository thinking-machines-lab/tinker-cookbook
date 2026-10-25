"""Tests for SDK instrumentation.

Uses fake SamplingClient/TrainingClient-shaped objects for behavior tests,
plus one test that patches the real ``tinker`` classes (no network calls).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import pytest

from tinker_cookbook.capture import instrument as instrument_mod
from tinker_cookbook.capture.exporter import CaptureExporter, CaptureRecord
from tinker_cookbook.capture.instrument import (
    _make_sample_async_wrapper,
    _make_sample_wrapper,
    _make_train_op_async_wrapper,
    _make_train_op_wrapper,
    instrument_tinker,
    uninstrument_tinker,
)
from tinker_cookbook.capture.scope import capture


class ImmediateSink:
    def __init__(self) -> None:
        self.records: list[CaptureRecord] = []

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        self.records.extend(records)


class FakeSequence:
    def __init__(self) -> None:
        self.tokens = [1, 2, 3]
        self.logprobs = [-0.1, -0.2, -0.3]
        self.stop_reason = "length"


class FakeResponse:
    def __init__(self) -> None:
        self.sequences = [FakeSequence(), FakeSequence()]


class FakeSamplingClient:
    """SamplingClient-shaped: sample returns an unresolved concurrent Future."""

    _sampling_session_id = "sess-1"

    def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Future[FakeResponse]:
        self.future: Future[FakeResponse] = Future()
        return self.future

    async def sample_async(
        self, prompt: Any, num_samples: int, sampling_params: Any
    ) -> FakeResponse:
        return FakeResponse()


class FakeAPIFuture:
    """APIFuture-shaped: no add_done_callback, wraps an inner ``_future``."""

    def __init__(self) -> None:
        self._future: Future[str] = Future()


class FakeTrainingClient:
    model_id = "model-1"

    def forward_backward(self, data: list[Any], loss_fn: str) -> FakeAPIFuture:
        self.future = FakeAPIFuture()
        return self.future

    async def forward_backward_async(self, data: list[Any], loss_fn: str) -> FakeAPIFuture:
        self.future = FakeAPIFuture()
        return self.future


class FakePrompt:
    length = 4

    def to_ints(self) -> list[int]:
        return [10, 20, 30, 40]


@pytest.fixture
def pipeline():  # type: ignore[no-untyped-def]
    sink = ImmediateSink()
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    instrument_mod._exporter = exporter
    yield sink, exporter
    instrument_mod._exporter = None
    exporter.shutdown()


def _wait_records(sink: ImmediateSink, n: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(sink.records) >= n:
            return
        time.sleep(0.01)
    raise AssertionError(f"expected {n} records, got {len(sink.records)}")


def test_sample_snapshot_at_call_survives_scope_exit(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    with capture(run_id="r1", iteration=7):
        fut = wrapped(client, FakePrompt(), 2, None)
    # Scope has exited; resolve the future afterwards.
    fut.set_result(FakeResponse())
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["scope"] == {"run_id": "r1", "iteration": 7}
    assert record["kind"] == "sample"
    assert record["num_samples"] == 2
    assert record["prompt_length"] == 4
    assert record["prompt_tokens"] == [10, 20, 30, 40]
    assert record["sampling_session_id"] == "sess-1"
    assert record["status"] == "ok"
    assert record["samples"][0]["tokens"] == [1, 2, 3]
    assert record["samples"][1]["logprobs"] == [-0.1, -0.2, -0.3]
    assert record["samples"][0]["stop_reason"] == "length"
    assert record["latency_sec"] >= 0


def test_sample_error_recorded(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    fut = wrapped(client, FakePrompt(), 1, None)
    fut.set_exception(RuntimeError("nope"))
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "error"
    assert "nope" in sink.records[0]["error"]


@pytest.mark.asyncio
async def test_sample_async_records(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    with capture(run_id="r2"):
        response = await wrapped(client, FakePrompt(), 1, None)
    assert isinstance(response, FakeResponse)
    _wait_records(sink, 1)
    assert sink.records[0]["scope"] == {"run_id": "r2"}
    assert sink.records[0]["status"] == "ok"


def test_train_op_api_future_path(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeTrainingClient()
    wrapped = _make_train_op_wrapper(FakeTrainingClient.forward_backward, "forward_backward")
    with capture(run_id="r3"):
        fut = wrapped(client, [1, 2, 3], "cross_entropy")
    fut._future.set_result("done")
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["kind"] == "train_op"
    assert record["op"] == "forward_backward"
    assert record["num_data"] == 3
    assert record["model_id"] == "model-1"
    assert record["scope"] == {"run_id": "r3"}
    assert record["status"] == "ok"


def test_callback_never_raises(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, exporter = pipeline

    class BadResponse:
        @property
        def sequences(self) -> list[Any]:
            raise ValueError("bad response")

    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    fut = wrapped(client, FakePrompt(), 1, None)
    before = exporter.callback_failures
    fut.set_result(BadResponse())  # done-callback must swallow the failure
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and exporter.callback_failures == before:
        time.sleep(0.01)
    assert exporter.callback_failures == before + 1


def test_instrument_real_tinker_classes() -> None:
    """Patch the REAL tinker classes: identity swapped, signature preserved, reversible."""
    import tinker

    sink = ImmediateSink()
    exporter = CaptureExporter(sink, flush_interval_sec=0.05)
    method_names: list[tuple[type, str]] = [
        (tinker.SamplingClient, "sample"),
        (tinker.SamplingClient, "sample_async"),
    ]
    for op in ("forward_backward", "optim_step", "save_weights_for_sampler"):
        method_names.append((tinker.TrainingClient, op))
        method_names.append((tinker.TrainingClient, f"{op}_async"))
    originals = {(cls, name): getattr(cls, name) for cls, name in method_names}
    try:
        instrument_tinker(exporter)
        for (cls, name), original in originals.items():
            patched = getattr(cls, name)
            assert patched is not original, f"{cls.__name__}.{name} not swapped"
            assert inspect.signature(patched) == inspect.signature(original)
            assert patched.__name__ == original.__name__

        # Idempotent: instrumenting again does not re-wrap.
        once_patched = tinker.SamplingClient.sample
        instrument_tinker(exporter)
        assert tinker.SamplingClient.sample is once_patched

        # Coroutine-function detection holds on 3.12+ via
        # inspect.markcoroutinefunction (not available on 3.11).
        if sys.version_info >= (3, 12):
            assert inspect.iscoroutinefunction(tinker.SamplingClient.sample_async)
            assert inspect.iscoroutinefunction(tinker.TrainingClient.forward_backward_async)
    finally:
        uninstrument_tinker()
        exporter.shutdown()

    for (cls, name), original in originals.items():
        restored = getattr(cls, name)
        if restored is original:
            continue
        # instrument_tinker applies trace's @scope layer up front (ordering
        # integration); uninstrument removes OUR layer, leaving trace's
        # process-wide idempotent wrapper, which unwraps to the original.
        assert getattr(restored, "_scope_instrumented", False) is True, (
            f"{cls.__name__}.{name} not restored"
        )
        assert inspect.unwrap(restored) is inspect.unwrap(original), (
            f"{cls.__name__}.{name} does not unwrap to the original"
        )
    # Uninstrument is idempotent.
    uninstrument_tinker()


@pytest.mark.asyncio
async def test_sample_async_gather_later_attributes_call_time_scope(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Coroutines created inside a scope but awaited after it attribute to call time."""
    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    coros = []
    with capture(run_id="r-gather", iteration=5):
        coros.append(wrapped(client, FakePrompt(), 1, None))
        coros.append(wrapped(client, FakePrompt(), 1, None))
    # Scope exited before anything is awaited.
    await asyncio.gather(*coros)
    _wait_records(sink, 2)
    for record in sink.records:
        assert record["scope"] == {"run_id": "r-gather", "iteration": 5}


@pytest.mark.asyncio
async def test_train_op_async_wrapper(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeTrainingClient()
    wrapped = _make_train_op_async_wrapper(
        FakeTrainingClient.forward_backward_async, "forward_backward"
    )
    with capture(run_id="r-async-train"):
        coro = wrapped(client, [1, 2], "cross_entropy")
    future = await coro
    future._future.set_result("done")
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["op"] == "forward_backward"
    assert record["num_data"] == 2
    assert record["scope"] == {"run_id": "r-async-train"}
    assert record["status"] == "ok"


def test_exporter_snapshot_survives_swap() -> None:
    """Outstanding futures report to the exporter active at CALL time."""
    sink_a, sink_b = ImmediateSink(), ImmediateSink()
    exporter_a = CaptureExporter(sink_a, max_batch_size=1, flush_interval_sec=0.02)
    exporter_b = CaptureExporter(sink_b, max_batch_size=1, flush_interval_sec=0.02)
    try:
        client = FakeSamplingClient()
        wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
        instrument_mod._exporter = exporter_a
        fut = wrapped(client, FakePrompt(), 1, None)
        # Exporter swapped (or cleared) while the future is outstanding.
        instrument_mod._exporter = exporter_b
        fut.set_result(FakeResponse())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not sink_a.records:
            time.sleep(0.01)
        assert len(sink_a.records) == 1
        assert sink_b.records == []
        # And the pending slot was released on exporter A.
        assert exporter_a.wait_pending(timeout=1.0) is True
    finally:
        instrument_mod._exporter = None
        exporter_a.shutdown()
        exporter_b.shutdown()


def test_pending_tracked_until_future_resolves(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, exporter = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    fut = wrapped(client, FakePrompt(), 1, None)
    assert exporter.wait_pending(timeout=0.1) is False  # future outstanding
    fut.set_result(FakeResponse())
    assert exporter.wait_pending(timeout=5.0) is True


def test_patched_async_methods_detected_as_coroutine_functions(pipeline) -> None:  # type: ignore[no-untyped-def]
    """On 3.12+ the sync-outer wrappers are marked as coroutine functions.

    On 3.11 (supported floor) there is no markcoroutinefunction and no way to
    combine detection with call-time snapshotting; snapshotting wins, and the
    in-repo consumer of detection (trace.py's @scope dispatch) is neutralized
    by the instrument_tinker ordering integration instead.
    """
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    wrapped_train = _make_train_op_async_wrapper(
        FakeTrainingClient.forward_backward_async, "forward_backward"
    )
    if sys.version_info >= (3, 12):
        assert inspect.iscoroutinefunction(wrapped)
        assert inspect.iscoroutinefunction(wrapped_train)
    # Regardless of detection, calling returns an awaitable coroutine.
    coro = wrapped(FakeSamplingClient(), FakePrompt(), 1, None)
    assert asyncio.iscoroutine(coro)
    coro.close()


def test_cancelled_future_records_cancelled_status(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, exporter = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    with capture(run_id="r-cancel"):
        fut = wrapped(client, FakePrompt(), 1, None)
    assert fut.cancel()
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["status"] == "cancelled"
    assert record["scope"] == {"run_id": "r-cancel"}
    assert "error" not in record
    # Pending slot released despite cancellation.
    assert exporter.wait_pending(timeout=5.0) is True


def test_unawaited_coroutine_does_not_leak_pending_slot(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, exporter = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    coro = wrapped(client, FakePrompt(), 1, None)
    # Never awaited: no pending slot may be held.
    assert exporter.wait_pending(timeout=0.2) is True
    coro.close()
    assert exporter.wait_pending(timeout=0.2) is True


def test_trace_ordering_integration() -> None:
    """instrument_tinker is order-independent with trace's SDK instrumentation.

    Order A (instrument first): our wrapper is marked _scope_instrumented so a
    later trace_init()/_instrument_sdk_clients() must NOT re-wrap it (a trace
    async-def wrapper on top would defer our call-time snapshot to first poll).
    Order A also applies trace's scope UNDER our wrapper up front, so trace
    spans still work. Order B (trace first) wraps trace's wrapper directly.
    """
    import tinker

    from tinker_cookbook.utils.trace import _instrument_sdk_clients

    sink = ImmediateSink()
    exporter = CaptureExporter(sink, flush_interval_sec=0.05)
    try:
        # Order A: instrument, then trace instruments.
        instrument_tinker(exporter)
        ours = tinker.SamplingClient.sample_async
        assert getattr(ours, "_scope_instrumented", False) is True
        # The layer underneath ours is trace's scope wrapper (applied by
        # instrument_tinker up front).
        underneath = ours.__dict__["_capture_original"]
        assert getattr(underneath, "_scope_instrumented", False) is True
        _instrument_sdk_clients()  # what trace_init() does
        assert tinker.SamplingClient.sample_async is ours, "trace re-wrapped our wrapper"
        uninstrument_tinker()
        # After uninstrument, the trace-scope-wrapped method remains (trace
        # instrumentation is process-wide and idempotent by design).
        assert tinker.SamplingClient.sample_async is underneath

        # Order B: trace already instrumented (from above), then instrument.
        instrument_tinker(exporter)
        ours_b = tinker.SamplingClient.sample_async
        assert ours_b.__dict__["_capture_original"] is underneath
        _instrument_sdk_clients()
        assert tinker.SamplingClient.sample_async is ours_b
    finally:
        uninstrument_tinker()
        exporter.shutdown()


class RaisingSamplingClient:
    _sampling_session_id = "sess-raise"

    def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Future[Any]:
        raise ValueError("bad request")


def test_sync_submission_failure_recorded_and_reraised(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    wrapped = _make_sample_wrapper(RaisingSamplingClient.sample)
    with capture(run_id="r-boom"), pytest.raises(ValueError, match="bad request"):
        wrapped(RaisingSamplingClient(), FakePrompt(), 1, None)
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["status"] == "error"
    assert "bad request" in record["error"]
    assert record["scope"] == {"run_id": "r-boom"}


def test_train_op_submission_failure_recorded_and_reraised(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline

    class RaisingTrainingClient:
        model_id = "model-raise"

        def forward_backward(self, data: list[Any], loss_fn: str) -> Any:
            raise RuntimeError("stopped client")

    wrapped = _make_train_op_wrapper(RaisingTrainingClient.forward_backward, "forward_backward")
    with pytest.raises(RuntimeError, match="stopped client"):
        wrapped(RaisingTrainingClient(), [1], "cross_entropy")
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "error"
    assert sink.records[0]["op"] == "forward_backward"


def test_seq_id_minted_per_call_and_stable(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_wrapper(FakeSamplingClient.sample)
    fut1 = wrapped(client, FakePrompt(), 1, None)
    fut1.set_result(FakeResponse())
    fut2 = wrapped(client, FakePrompt(), 1, None)
    fut2.set_result(FakeResponse())
    _wait_records(sink, 2)
    seq_ids = [r["seq_id"] for r in sink.records]
    assert all(isinstance(sid, int) and 0 <= sid < 2**62 for sid in seq_ids)
    assert seq_ids[0] != seq_ids[1]  # unique per call


def test_sync_submission_tracked_before_sdk_call(pipeline) -> None:  # type: ignore[no-untyped-def]
    """The pending count must cover the submission itself: a teardown calling
    wait_pending while the SDK method is still executing must see it."""
    _, exporter = pipeline
    observed: list[bool] = []

    class ObservingClient:
        _sampling_session_id = "sess-obs"

        def sample(
            self, prompt: Any, num_samples: int, sampling_params: Any
        ) -> Future[FakeResponse]:
            observed.append(exporter.wait_pending(timeout=0))
            future: Future[FakeResponse] = Future()
            future.set_result(FakeResponse())
            return future

    wrapped = _make_sample_wrapper(ObservingClient.sample)
    wrapped(ObservingClient(), FakePrompt(), 1, None)
    assert observed == [False]  # in flight DURING the submission
    assert exporter.wait_pending(timeout=5.0)


def test_sync_submission_failure_releases_pending(pipeline) -> None:  # type: ignore[no-untyped-def]
    _, exporter = pipeline

    class RaisingClient:
        _sampling_session_id = "sess-raise"

        def sample(
            self, prompt: Any, num_samples: int, sampling_params: Any
        ) -> Future[FakeResponse]:
            raise ValueError("boom")

    wrapped = _make_sample_wrapper(RaisingClient.sample)
    with pytest.raises(ValueError, match="boom"):
        wrapped(RaisingClient(), FakePrompt(), 1, None)
    assert exporter.wait_pending(timeout=1.0)  # slot released on failure


def test_async_cancellation_recorded_as_cancelled_and_reraised(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Task cancellation is not an SDK failure: the record must say
    "cancelled" (matching the sync future path) and CancelledError must
    propagate."""
    sink, exporter = pipeline

    class CancellingClient:
        _sampling_session_id = "sess-cancel"

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            raise asyncio.CancelledError

    wrapped = _make_sample_async_wrapper(CancellingClient.sample_async)

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await wrapped(CancellingClient(), FakePrompt(), 1, None)

    asyncio.run(run())
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "cancelled"
    assert "error" not in sink.records[0]
    assert exporter.wait_pending(timeout=1.0)


def test_train_op_async_cancellation_recorded_as_cancelled(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, exporter = pipeline

    class CancellingTrainingClient:
        model_id = "model-cancel"

        async def forward_backward_async(self, data: list[Any], loss_fn: str) -> Any:
            raise asyncio.CancelledError

    wrapped = _make_train_op_async_wrapper(
        CancellingTrainingClient.forward_backward_async, "forward_backward"
    )

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await wrapped(CancellingTrainingClient(), [1], "ce")

    asyncio.run(run())
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "cancelled"
    assert exporter.wait_pending(timeout=1.0)


def test_task_cancelled_before_first_poll_still_recorded(pipeline) -> None:  # type: ignore[no-untyped-def]
    """A task cancelled before its first poll never runs the inner coroutine
    body; the weakref guard on the coroutine object must still emit the
    "cancelled" record, and pending tracking must stay at zero (the body
    never tracked)."""
    import gc

    sink, exporter = pipeline

    class NeverPolledClient:
        _sampling_session_id = "sess-unpolled"

        async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
            raise AssertionError("body must never run")

    wrapped = _make_sample_async_wrapper(NeverPolledClient.sample_async)

    async def run() -> None:
        task = asyncio.get_running_loop().create_task(
            wrapped(NeverPolledClient(), FakePrompt(), 1, None)
        )
        task.cancel()  # before the loop ever polls it
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    gc.collect()  # reclaim the never-started coroutine deterministically
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "cancelled"
    assert exporter.wait_pending(timeout=1.0)  # no leaked pending slot


def test_discarded_coroutine_recorded_as_cancelled(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Creating the coroutine and never awaiting it gets the same
    cancelled accounting (documented in _guard_unstarted_coroutine)."""
    import gc
    import warnings

    sink, _ = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "never awaited"
        coro = wrapped(client, FakePrompt(), 1, None)
        del coro
        gc.collect()
    _wait_records(sink, 1)
    assert sink.records[0]["status"] == "cancelled"


def test_unstarted_guard_detached_once_body_begins(pipeline, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Once the coroutine's first poll happens the guard must be detached,
    so callers retaining completed Tasks do not keep record/exporter alive
    through the finalizer (and no spurious record can ever fire)."""
    captured: list[Any] = []
    real_guard = instrument_mod._guard_unstarted_coroutine

    def spy(coro: Any, exporter: Any, record: Any, started: Any) -> Any:
        finalizer = real_guard(coro, exporter, record, started)
        captured.append(finalizer)
        return finalizer

    monkeypatch.setattr(instrument_mod, "_guard_unstarted_coroutine", spy)
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    asyncio.run(wrapped(client, FakePrompt(), 1, None))
    assert len(captured) == 1
    assert captured[0].alive is False  # detached at first poll


def test_unstarted_guards_flushed_at_uninstrument(pipeline) -> None:  # type: ignore[no-untyped-def]
    """A cancelled-before-first-poll call whose Task is still retained (no
    GC yet) must get its record when uninstrument_tinker() flushes the guard
    registry, while the exporter is still alive; the fired guard must not
    double-record on a later GC."""
    import gc
    import warnings

    sink, exporter = pipeline
    client = FakeSamplingClient()
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coro = wrapped(client, FakePrompt(), 1, None)  # retained, never polled
        uninstrument_tinker()
        _wait_records(sink, 1)
        assert sink.records[0]["status"] == "cancelled"
        assert instrument_mod._unstarted_guards == set()
        coro.close()
        del coro
    gc.collect()
    time.sleep(0.1)
    assert len(sink.records) == 1  # no double record after GC


def test_wrapper_after_uninstrument_is_harmless(pipeline) -> None:  # type: ignore[no-untyped-def]
    """A wrapper reference invoked after uninstrument_tinker() (post-flush)
    snapshots a None exporter under the guards lock, so its guard is a no-op
    and nothing is recorded or leaked."""
    import gc
    import warnings

    sink, _ = pipeline
    wrapped = _make_sample_async_wrapper(FakeSamplingClient.sample_async)
    uninstrument_tinker()  # clears _exporter and flushes guards under the lock
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coro = wrapped(FakeSamplingClient(), FakePrompt(), 1, None)
        del coro
    gc.collect()
    time.sleep(0.1)
    assert sink.records == []


class DelegatingSamplingClient:
    """Mirrors the real SDK: ``sample_async`` delegates to ``self.sample()``
    synchronously inside its coroutine body (the SDK awaits
    ``AwaitableConcurrentFuture(self.sample(...))``)."""

    _sampling_session_id = "sess-delegate"

    def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Future[FakeResponse]:
        future: Future[FakeResponse] = Future()
        future.set_result(FakeResponse())
        return future

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        return self.sample(prompt, num_samples, sampling_params).result()


def _instrumented_delegating_client() -> DelegatingSamplingClient:
    client = DelegatingSamplingClient()
    client.sample = _make_sample_wrapper(  # type: ignore[method-assign]
        DelegatingSamplingClient.sample
    ).__get__(client)
    client.sample_async = _make_sample_async_wrapper(  # type: ignore[method-assign]
        DelegatingSamplingClient.sample_async
    ).__get__(client)
    return client


def test_sample_async_delegation_records_exactly_once(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Regression: the SDK's sample_async delegates to the ALSO-instrumented
    sample(); a live run stored two identical rows per async call. The
    re-entrancy contextvar must suppress the inner record."""
    sink, exporter = pipeline
    client = _instrumented_delegating_client()
    asyncio.run(client.sample_async(FakePrompt(), 1, None))
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 1)
    time.sleep(0.1)  # a duplicate would land immediately after
    assert len(sink.records) == 1
    assert sink.records[0]["kind"] == "sample"


def test_sync_sample_still_records_exactly_once(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, exporter = pipeline
    client = _instrumented_delegating_client()
    client.sample(FakePrompt(), 1, None)
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 1)
    time.sleep(0.1)
    assert len(sink.records) == 1


def test_concurrent_sample_async_tasks_record_two(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Independent concurrent calls are never suppressed: each asyncio task
    runs in its own copied context, so one task's in-progress flag is
    invisible to the other."""
    sink, exporter = pipeline
    client = _instrumented_delegating_client()

    async def run() -> None:
        await asyncio.gather(
            client.sample_async(FakePrompt(), 1, None),
            client.sample_async(FakePrompt(), 1, None),
        )

    asyncio.run(run())
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 2)
    time.sleep(0.1)
    assert len(sink.records) == 2


class DelegatingTrainingClient:
    """The SDK's *_async train ops delegate to their sync counterparts too."""

    model_id = "model-delegate"

    def forward_backward(self, data: list[Any], loss_fn: str) -> FakeAPIFuture:
        future = FakeAPIFuture()
        future._future.set_result("ok")
        return future

    async def forward_backward_async(self, data: list[Any], loss_fn: str) -> FakeAPIFuture:
        return self.forward_backward(data, loss_fn)


def test_train_op_async_delegation_records_exactly_once(pipeline) -> None:  # type: ignore[no-untyped-def]
    sink, exporter = pipeline
    client = DelegatingTrainingClient()
    client.forward_backward = _make_train_op_wrapper(  # type: ignore[method-assign]
        DelegatingTrainingClient.forward_backward, "forward_backward"
    ).__get__(client)
    client.forward_backward_async = _make_train_op_async_wrapper(  # type: ignore[method-assign]
        DelegatingTrainingClient.forward_backward_async, "forward_backward"
    ).__get__(client)
    asyncio.run(client.forward_backward_async([1], "ce"))
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 1)
    time.sleep(0.1)
    assert len(sink.records) == 1
    assert sink.records[0]["kind"] == "train_op"


def test_train_op_keyword_arguments_captured(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Keyword-style calls (the common SDK style) must carry the same
    request metadata as positional ones."""
    sink, exporter = pipeline

    class KwargClient:
        model_id = "model-kw"

        def forward_backward(self, data: list[Any] | None = None, loss_fn: str = "") -> Any:
            future = FakeAPIFuture()
            future._future.set_result("ok")
            return future

        def save_weights_for_sampler(self, name: str = "") -> Any:
            future = FakeAPIFuture()
            future._future.set_result("ok")
            return future

        def optim_step(self, adam_params: Any = None) -> Any:
            future = FakeAPIFuture()
            future._future.set_result("ok")
            return future

    class Adamish:
        learning_rate = 3e-4
        beta1 = 0.9
        beta2 = 0.95
        eps = 1e-8
        weight_decay = 0.0
        grad_clip_norm = 1.0

    client = KwargClient()
    _make_train_op_wrapper(KwargClient.forward_backward, "forward_backward")(
        client, data=[1, 2, 3], loss_fn="ce"
    )
    _make_train_op_wrapper(KwargClient.save_weights_for_sampler, "save_weights_for_sampler")(
        client, name="ckpt-7"
    )
    _make_train_op_wrapper(KwargClient.optim_step, "optim_step")(client, adam_params=Adamish())
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 3)
    by_op = {r["op"]: r for r in sink.records}
    assert by_op["forward_backward"]["num_data"] == 3
    assert by_op["save_weights_for_sampler"]["name"] == "ckpt-7"
    assert by_op["optim_step"]["hyperparams"]["learning_rate"] == 3e-4
    assert by_op["optim_step"]["hyperparams"]["grad_clip_norm"] == 1.0


def test_forward_backward_outcome_metrics_summary(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Completion records carry a bounded scalar summary of the result's
    metrics dict (floats/ints only; bools and non-scalars dropped)."""
    sink, exporter = pipeline

    class FBOutput:
        metrics = {"loss:sum": 1.5, "clipped": True, "tokens": 128, "junk": [1, 2]}

    class MetricsFuture:
        def __init__(self) -> None:
            self._future: Future[Any] = Future()

    class MetricsClient:
        model_id = "model-metrics"

        def forward_backward(self, data: list[Any], loss_fn: str) -> Any:
            future = MetricsFuture()
            future._future.set_result(FBOutput())
            return future

    _make_train_op_wrapper(MetricsClient.forward_backward, "forward_backward")(
        MetricsClient(), [1], "ce"
    )
    exporter.wait_pending(timeout=5.0)
    _wait_records(sink, 1)
    assert sink.records[0]["metrics"] == {"loss:sum": 1.5, "tokens": 128.0}
