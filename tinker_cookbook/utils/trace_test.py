import asyncio
import contextlib
import inspect
import json
import tempfile
import threading
import time
from unittest.mock import patch

import tinker

from tinker_cookbook.utils.trace import (
    IterationWindow,
    SpanRecord,
    _build_gantt_chart,
    get_scope_context,
    scope,
    trace_init,
    trace_iteration,
    trace_shutdown,
    update_scope_context,
)


# --- Helpers ---


@contextlib.contextmanager
def trace_session(wandb_logger=None, span_chart_every=0):
    """Start a trace session backed by a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
        trace_init(
            output_file=f.name,
            wandb_logger=wandb_logger,
            span_chart_every=span_chart_every,
        )
        try:
            yield f.name
        finally:
            trace_shutdown()


class FakeLogger:
    """Minimal logger that captures log_metrics calls for testing."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.logged.append((dict(metrics), step))


# --- Decorated helpers for test_trace (multi-thread integration) ---
#
# These must be module-level because @scope captures __name__ at decoration time.
# They are used only by test_trace.


@scope
async def foo():
    await asyncio.sleep(0.1)
    context = get_scope_context()
    context.attributes["foo"] = "foo"
    context.attributes["foo2"] = 1
    await bar()


@scope
async def bar():
    await asyncio.sleep(0.05)
    context = get_scope_context()
    context.attributes["bar"] = 1
    await baz()


@scope
def ced():
    pass


@scope
async def baz():
    await asyncio.sleep(0.02)
    update_scope_context({"baz": "baz"})
    ced()


@scope
async def coroutine1():
    await foo()
    await asyncio.sleep(0.05)


@scope
async def coroutine2():
    await asyncio.sleep(0.15)
    await foo()


@scope
def sync_func():
    pass


@scope
async def work(thread_name: str):
    task1 = asyncio.create_task(coroutine1(), name=f"{thread_name}-coroutine-1")
    task2 = asyncio.create_task(coroutine2(), name=f"{thread_name}-coroutine-2")
    sync_func()
    await asyncio.gather(task1, task2)


@scope
async def example_program():
    @scope
    def thread_target():
        asyncio.run(work("secondary_thread"))

    thread = threading.Thread(target=thread_target, name="secondary_thread")
    thread.start()

    await work("main_thread")

    thread.join()


# --- @scope decorator ---


def test_trace():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as temp_file:
        trace_init(output_file=temp_file.name)
        asyncio.run(example_program())
        trace_shutdown()

        with open(temp_file.name, "r") as f:
            events = [json.loads(line) for line in f]

    # There should be 2 process metadata events
    num_metadata_pid_events = sum(
        1 for event in events if event["ph"] == "M" and event["tid"] == 0
    )
    assert num_metadata_pid_events == 2
    num_unique_pids = len(set(event["pid"] for event in events if event["ph"] != "M"))
    assert num_unique_pids == 2

    # main thread has 3: main, coroutine-1, coroutine-2
    # secondary thread has 4: thread_target, work, coroutine-1, coroutine-2
    num_metadata_tid_events = sum(
        1 for event in events if event["ph"] == "M" and event["tid"] != 0
    )
    assert num_metadata_tid_events == 7
    num_unique_tids = len(set(event["tid"] for event in events if event["ph"] != "M"))
    assert num_unique_tids == 7

    # Validate that attributes are set correctly
    for event in events:
        if event["ph"] != "E":
            continue
        if event["name"] == "foo":
            assert event["args"]["foo"] == "foo"
            assert event["args"]["foo2"] == 1
        if event["name"] == "bar":
            assert event["args"]["bar"] == 1
        if event["name"] == "baz":
            assert event["args"]["baz"] == "baz"


def test_scope_noop_async():
    """Async @scope passes through when no collector and no iteration window."""

    @scope
    async def noop_async():
        return 42

    # No trace_init, no trace_iteration — should just return the value
    result = asyncio.run(noop_async())
    assert result == 42


def test_scope_noop_sync():
    """Sync @scope passes through when no collector and no iteration window."""

    @scope
    def noop_sync():
        return 99

    # No trace_init, no trace_iteration — should just return the value
    result = noop_sync()
    assert result == 99


# --- IterationWindow ---


def test_iteration_window_single_span():
    window = IterationWindow()
    window.record_span("train_step", 0.0, 1.5)
    metrics = window.aggregate()
    assert metrics == {"time/train_step": 1.5}


def test_iteration_window_multiple_spans_same_name():
    window = IterationWindow()
    window.record_span("sample", 0.0, 2.0)
    window.record_span("sample", 0.1, 3.0)
    window.record_span("sample", 0.2, 1.5)
    metrics = window.aggregate()
    assert metrics["time/sample:count"] == 3
    assert metrics["time/sample:total"] == 2.0 + 2.9 + 1.3
    assert abs(metrics["time/sample:mean"] - (2.0 + 2.9 + 1.3) / 3) < 1e-9
    assert metrics["time/sample:max"] == 2.9


def test_iteration_window_mixed_spans():
    window = IterationWindow()
    window.record_span("eval", 0.0, 1.0)
    window.record_span("sample", 1.0, 3.0)
    window.record_span("sample", 1.1, 2.5)
    window.record_span("train", 3.0, 4.0)
    metrics = window.aggregate()
    # eval: single call
    assert metrics["time/eval"] == 1.0
    # sample: two calls
    assert metrics["time/sample:count"] == 2
    # train: single call
    assert metrics["time/train"] == 1.0


def test_iteration_window_empty():
    window = IterationWindow()
    assert window.aggregate() == {}
    assert window.get_span_records() == []


def test_iteration_window_span_records():
    window = IterationWindow()
    window.record_span("a", 100.0, 101.0)
    window.record_span("b", 100.5, 102.0)
    records = window.get_span_records()
    assert len(records) == 2
    assert records[0]["task"] == "a"
    assert records[1]["task"] == "b"
    # start times should be relative (first span starts at 0)
    assert records[0]["start"] < records[1]["start"]


def test_merge_spans():
    """merge_spans integrates external spans into the window."""
    window = IterationWindow()
    window.record_span("local", 0.0, 1.0)

    external = [
        SpanRecord(name="worker", start_time=0.5, end_time=2.0, wall_start=1000.5, wall_end=1002.0),
    ]
    window.merge_spans(external)

    metrics = window.aggregate()
    assert "time/local" in metrics
    assert "time/worker" in metrics

    records = window.get_span_records()
    assert len(records) == 2


# --- trace_iteration ---


def test_trace_iteration_collects_scoped_spans():
    """trace_iteration collects spans from @scope-decorated functions."""

    @scope
    async def fast_op():
        await asyncio.sleep(0.01)

    @scope
    async def slow_op():
        await asyncio.sleep(0.05)

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger):
            with trace_iteration(step=0):
                await fast_op()
                await slow_op()

    asyncio.run(run())

    assert len(fake_logger.logged) == 1
    metrics, step = fake_logger.logged[0]
    assert step == 0
    assert "time/total" in metrics
    assert "time/fast_op" in metrics
    assert "time/slow_op" in metrics
    assert metrics["time/slow_op"] > metrics["time/fast_op"]


def test_trace_iteration_aggregates_repeated_calls():
    """Repeated calls to the same @scope function produce aggregate metrics."""

    @scope
    async def repeated_op():
        await asyncio.sleep(0.01)

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger):
            with trace_iteration(step=5):
                await asyncio.gather(
                    repeated_op(),
                    repeated_op(),
                    repeated_op(),
                )

    asyncio.run(run())

    metrics, step = fake_logger.logged[0]
    assert step == 5
    assert metrics["time/repeated_op:count"] == 3
    assert "time/repeated_op:mean" in metrics
    assert "time/repeated_op:max" in metrics
    assert "time/repeated_op:total" in metrics


def test_trace_iteration_without_trace_init():
    """trace_iteration works even without trace_init (no Perfetto, just span collection)."""

    @scope
    async def some_work():
        await asyncio.sleep(0.01)

    async def run():
        # No trace_init — _trace_collector is None
        with trace_iteration(step=0):
            await some_work()

    asyncio.run(run())
    # No crash, no logger configured so nothing logged — just verifying no error


def test_trace_iteration_no_op_without_logger():
    """trace_iteration with Perfetto but no wandb_logger still works (spans go to Perfetto only)."""

    @scope
    async def op():
        await asyncio.sleep(0.01)

    async def run():
        with trace_session():  # No wandb_logger
            with trace_iteration(step=0):
                await op()

    asyncio.run(run())


def test_trace_iteration_sync_functions():
    """trace_iteration collects spans from sync @scope-decorated functions."""

    @scope
    def sync_work():
        time.sleep(0.01)

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger):
            with trace_iteration(step=0):
                sync_work()
                sync_work()

    asyncio.run(run())

    metrics, _ = fake_logger.logged[0]
    assert metrics["time/sync_work:count"] == 2


def test_trace_iteration_on_exception():
    """trace_iteration still logs partial timing when an exception occurs."""

    @scope
    async def succeeds():
        await asyncio.sleep(0.01)

    @scope
    async def fails():
        await asyncio.sleep(0.01)
        raise ValueError("boom")

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger):
            try:
                with trace_iteration(step=0):
                    await succeeds()
                    await fails()
            except ValueError:
                pass

    asyncio.run(run())

    # Should still have logged timing for the spans that completed
    assert len(fake_logger.logged) == 1
    metrics, _ = fake_logger.logged[0]
    assert "time/total" in metrics
    assert "time/succeeds" in metrics
    assert "time/fails" in metrics


def test_trace_iteration_nested():
    """Nested trace_iteration: inner window is independent from outer."""

    @scope
    async def outer_op():
        await asyncio.sleep(0.01)

    @scope
    async def inner_op():
        await asyncio.sleep(0.01)

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger):
            with trace_iteration(step=0):
                await outer_op()
                with trace_iteration(step=100):
                    await inner_op()

    asyncio.run(run())

    # Two log calls: one for inner (step=100), one for outer (step=0)
    assert len(fake_logger.logged) == 2
    steps = {s for _, s in fake_logger.logged}
    assert steps == {0, 100}

    # Inner should only have inner_op
    inner_metrics = next(m for m, s in fake_logger.logged if s == 100)
    assert "time/inner_op" in inner_metrics
    assert "time/outer_op" not in inner_metrics

    # Outer should have outer_op (inner_op was captured by inner window, not outer)
    outer_metrics = next(m for m, s in fake_logger.logged if s == 0)
    assert "time/outer_op" in outer_metrics


def test_trace_iteration_span_chart_every():
    """span_chart_every controls when Gantt charts are built."""

    @scope
    async def op():
        await asyncio.sleep(0.01)

    fake_logger = FakeLogger()

    async def run():
        with trace_session(wandb_logger=fake_logger, span_chart_every=3):
            # step 0 should trigger chart (0 % 3 == 0)
            with trace_iteration(step=0):
                await op()
            # step 1 should NOT trigger chart
            with trace_iteration(step=1):
                await op()
            # step 3 should trigger chart (3 % 3 == 0)
            with trace_iteration(step=3):
                await op()

    # Patch _build_gantt_chart to track calls without requiring plotly
    call_steps: list[int] = []

    def tracking_build(span_records, step):
        call_steps.append(step)
        return None  # Return None to skip wandb.log

    with patch("tinker_cookbook.utils.trace._build_gantt_chart", side_effect=tracking_build):
        asyncio.run(run())

    assert len(fake_logger.logged) == 3
    # _build_gantt_chart should have been called for step 0 and step 3
    assert call_steps == [0, 3]


def test_trace_shutdown_resets_globals():
    """trace_shutdown resets _wandb_logger and _span_chart_every."""
    fake_logger = FakeLogger()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
        trace_init(output_file=f.name, wandb_logger=fake_logger, span_chart_every=10)
        trace_shutdown()

    # After shutdown, trace_iteration should not log anything
    # (because _wandb_logger was reset to None)
    async def run():
        with trace_iteration(step=0):
            pass

    asyncio.run(run())
    # FakeLogger should have zero log calls (nothing logged after shutdown)
    assert len(fake_logger.logged) == 0


# --- Gantt chart ---


def test_build_gantt_chart_success():
    """_build_gantt_chart returns a figure when plotly is available and spans are non-empty."""
    import datetime

    span_records = [
        {"task": "a", "start": datetime.datetime(2000, 1, 1), "end": datetime.datetime(2000, 1, 1, 0, 0, 1)},
        {"task": "b", "start": datetime.datetime(2000, 1, 1, 0, 0, 0, 500000), "end": datetime.datetime(2000, 1, 1, 0, 0, 2)},
    ]
    fig = _build_gantt_chart(span_records, step=0)
    # If plotly is installed, we get a figure; if not, None
    try:
        import plotly  # noqa: F401

        assert fig is not None
    except ImportError:
        assert fig is None


def test_build_gantt_chart_empty_spans():
    """_build_gantt_chart returns None for empty span list."""
    # Even if plotly is installed, empty spans should return None
    fig = _build_gantt_chart([], step=0)
    assert fig is None


def test_build_gantt_chart_no_plotly():
    """_build_gantt_chart returns None when plotly is not importable."""
    import datetime

    span_records = [
        {"task": "a", "start": datetime.datetime(2000, 1, 1), "end": datetime.datetime(2000, 1, 1, 0, 0, 1)},
    ]
    with patch.dict("sys.modules", {"plotly": None, "plotly.express": None}):
        fig = _build_gantt_chart(span_records, step=0)
    assert fig is None


# --- SDK client instrumentation ---


def test_sdk_client_instrumentation_covers_all_async_methods():
    """Tripwire: catches new async methods added to Tinker SDK clients.

    If this test fails after a tinker dependency bump, add the new method(s) to
    _instrument_sdk_clients in trace.py.
    """
    # Collect all public async methods from SDK clients
    sdk_async_methods: dict[type, set[str]] = {}
    for cls in (tinker.TrainingClient, tinker.SamplingClient):
        methods = set()
        for name in dir(cls):
            if name.startswith("_"):
                continue
            attr = getattr(cls, name, None)
            if attr is not None and inspect.iscoroutinefunction(attr):
                methods.add(name)
        sdk_async_methods[cls] = methods

    # Instrument via trace_init and check all are wrapped
    with trace_session():
        for cls, methods in sdk_async_methods.items():
            for method_name in methods:
                original = getattr(cls, method_name)
                assert getattr(original, "_scope_instrumented", False), (
                    f"{cls.__name__}.{method_name} is an async method but not instrumented by "
                    f"_instrument_sdk_clients. Add it to the method list in trace.py."
                )


def test_scope_double_wrapping_prevention():
    """_instrument_sdk_clients is idempotent — calling trace_init twice doesn't double-wrap."""
    with trace_session():
        first_ref = getattr(tinker.TrainingClient, "forward_backward_async")
        assert getattr(first_ref, "_scope_instrumented", False)

    # Second trace_init — should not re-wrap
    with trace_session():
        second_ref = getattr(tinker.TrainingClient, "forward_backward_async")
        assert getattr(second_ref, "_scope_instrumented", False)
        # Same wrapper object — not double-wrapped
        assert first_ref is second_ref


if __name__ == "__main__":
    trace_init()
    asyncio.run(example_program())
    trace_shutdown()
