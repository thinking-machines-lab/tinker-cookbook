import asyncio
import contextlib
import inspect
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import tinker

from tinker_cookbook.utils.trace import (
    IterationWindow,
    SpanRecord,
    _build_gantt_chart,
    get_scope_context,
    save_gantt_chart_html,
    scope,
    scope_span,
    scope_span_sync,
    trace_init,
    trace_iteration,
    trace_shutdown,
    update_scope_context,
)

# --- Helpers ---


@contextlib.contextmanager
def trace_session():
    """Start a trace session backed by a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
        trace_init(output_file=f.name)
        try:
            yield f.name
        finally:
            trace_shutdown()


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

        with open(temp_file.name) as f:
            events = [json.loads(line) for line in f]

        # There should be 2 process metadata events
        num_metadata_pid_events = sum(
            1 for event in events if event["ph"] == "M" and event["tid"] == 0
        )
        assert num_metadata_pid_events == 2
        num_unique_pids = len({event["pid"] for event in events if event["ph"] != "M"})
        assert num_unique_pids == 2

        # main thread has 3: main, coroutine-1, coroutine-2
        # secondary thread has 4: thread_target, work, coroutine-1, coroutine-2
        num_metadata_tid_events = sum(
            1 for event in events if event["ph"] == "M" and event["tid"] != 0
        )
        assert num_metadata_tid_events == 7
        num_unique_tids = len({event["tid"] for event in events if event["ph"] != "M"})
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


def test_get_timing_metrics():
    """get_timing_metrics includes time/total when set by trace_iteration."""
    window = IterationWindow()
    window.record_span("op", 0.0, 1.0)
    window._total_time = 2.5
    metrics = window.get_timing_metrics()
    assert metrics["time/op"] == 1.0
    assert metrics["time/total"] == 2.5


def test_get_timing_metrics_without_total():
    """get_timing_metrics works without time/total (no trace_iteration)."""
    window = IterationWindow()
    window.record_span("op", 0.0, 1.0)
    metrics = window.get_timing_metrics()
    assert metrics["time/op"] == 1.0
    assert "time/total" not in metrics


# --- write_spans_jsonl ---


def test_write_spans_jsonl():
    """write_spans_jsonl appends one JSON line per call."""
    window = IterationWindow()
    window.record_span("a", 100.0, 101.5)
    window.record_span("b", 100.2, 102.0)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True, mode="w") as f:
        path = f.name

    window.write_spans_jsonl(path, step=0)
    window.write_spans_jsonl(path, step=1)

    with open(path) as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["step"] == 0
    assert lines[1]["step"] == 1
    assert len(lines[0]["spans"]) == 2
    assert lines[0]["spans"][0]["name"] == "a"
    assert lines[0]["spans"][1]["name"] == "b"
    assert abs(lines[0]["spans"][0]["duration"] - 1.5) < 1e-9
    # wall_start of first span should be ~0 (relative)
    assert lines[0]["spans"][0]["wall_start"] < 0.1

    Path(path).unlink(missing_ok=True)


def test_write_spans_jsonl_empty_window():
    """write_spans_jsonl is a no-op for empty windows."""
    window = IterationWindow()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True, mode="w") as f:
        path = f.name

    window.write_spans_jsonl(path, step=0)
    assert not Path(path).exists()


# --- trace_iteration ---


def test_trace_iteration_collects_scoped_spans():
    """trace_iteration collects spans from @scope-decorated functions."""

    @scope
    async def fast_op():
        await asyncio.sleep(0.01)

    @scope
    async def slow_op():
        await asyncio.sleep(0.05)

    async def run():
        with trace_session():
            with trace_iteration(step=0) as window:
                await fast_op()
                await slow_op()
            return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert "time/total" in metrics
    assert "time/fast_op" in metrics
    assert "time/slow_op" in metrics
    assert metrics["time/slow_op"] > metrics["time/fast_op"]


def test_trace_iteration_aggregates_repeated_calls():
    """Repeated calls to the same @scope function produce aggregate metrics."""

    @scope
    async def repeated_op():
        await asyncio.sleep(0.01)

    async def run():
        with trace_session():
            with trace_iteration(step=5) as window:
                await asyncio.gather(
                    repeated_op(),
                    repeated_op(),
                    repeated_op(),
                )
            return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
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
        with trace_iteration(step=0) as window:
            await some_work()
        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert "time/some_work" in metrics
    assert "time/total" in metrics


def test_trace_iteration_with_perfetto_only():
    """trace_iteration with Perfetto but caller doesn't use timing metrics."""

    @scope
    async def op():
        await asyncio.sleep(0.01)

    async def run():
        with trace_session():
            with trace_iteration(step=0) as window:
                await op()
            return window

    window = asyncio.run(run())
    # Caller can choose to ignore the window — no crash
    assert "time/op" in window.get_timing_metrics()


def test_trace_iteration_sync_functions():
    """trace_iteration collects spans from sync @scope-decorated functions."""

    @scope
    def sync_work():
        time.sleep(0.01)

    async def run():
        with trace_session():
            with trace_iteration(step=0) as window:
                sync_work()
                sync_work()
            return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert metrics["time/sync_work:count"] == 2


def test_trace_iteration_on_exception():
    """trace_iteration still captures partial timing when an exception occurs."""

    @scope
    async def succeeds():
        await asyncio.sleep(0.01)

    @scope
    async def fails():
        await asyncio.sleep(0.01)
        raise ValueError("boom")

    async def run():
        with trace_session():
            with trace_iteration(step=0) as window:
                try:
                    await succeeds()
                    await fails()
                except ValueError:
                    pass
            return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
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

    async def run():
        with trace_session():
            with trace_iteration(step=0) as outer_window:
                await outer_op()
                with trace_iteration(step=100) as inner_window:
                    await inner_op()
            return outer_window, inner_window

    outer_window, inner_window = asyncio.run(run())

    # Inner should only have inner_op
    inner_metrics = inner_window.get_timing_metrics()
    assert "time/inner_op" in inner_metrics
    assert "time/outer_op" not in inner_metrics

    # Outer should have outer_op (inner_op was captured by inner window, not outer)
    outer_metrics = outer_window.get_timing_metrics()
    assert "time/outer_op" in outer_metrics


# --- scope_span ---


def test_scope_span_async():
    """scope_span records to iteration window."""

    async def run():
        with trace_iteration(step=0) as window:
            async with scope_span("my_span"):
                await asyncio.sleep(0.01)
        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert "time/my_span" in metrics
    assert metrics["time/my_span"] >= 0.01


def test_scope_span_with_perfetto():
    """scope_span records to both Perfetto and iteration window."""

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        trace_file = tmp.name

    try:

        async def run():
            trace_init(output_file=trace_file)
            try:
                with trace_iteration(step=0) as window:
                    async with scope_span("traced_span"):
                        await asyncio.sleep(0.01)
                return window
            finally:
                trace_shutdown()

        window = asyncio.run(run())

        # Check iteration window
        metrics = window.get_timing_metrics()
        assert "time/traced_span" in metrics

        # Check Perfetto trace file has the span
        with open(trace_file) as f:
            events = [json.loads(line) for line in f]
        span_events = [e for e in events if e.get("name") == "traced_span"]
        assert len(span_events) >= 2  # BEGIN + END
    finally:
        Path(trace_file).unlink(missing_ok=True)


def test_scope_span_noop():
    """scope_span is a no-op when no collector and no iteration window."""

    async def run():
        async with scope_span("should_not_crash"):
            return 42

    result = asyncio.run(run())
    assert result == 42


def test_scope_span_sync():
    """scope_span_sync records to iteration window."""

    with trace_iteration(step=0) as window:
        with scope_span_sync("sync_span"):
            time.sleep(0.01)

    metrics = window.get_timing_metrics()
    assert "time/sync_span" in metrics


def test_scope_span_multiple():
    """Multiple scope_span calls with the same name produce aggregates."""

    async def run():
        with trace_iteration(step=0) as window:
            for _ in range(3):
                async with scope_span("repeated"):
                    await asyncio.sleep(0.01)
        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert metrics["time/repeated:count"] == 3
    assert "time/repeated:total" in metrics


def test_scope_span_on_exception():
    """scope_span still records the span when the block raises."""

    async def run():
        with trace_iteration(step=0) as window:
            async with scope_span("before_error"):
                await asyncio.sleep(0.01)
            try:
                async with scope_span("erroring"):
                    await asyncio.sleep(0.01)
                    raise ValueError("boom")
            except ValueError:
                pass
        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()
    assert "time/before_error" in metrics
    assert "time/erroring" in metrics


# --- Gantt chart ---


def test_build_gantt_chart_success():
    """_build_gantt_chart returns a figure when plotly is available and spans are non-empty."""
    import datetime

    span_records = [
        {
            "task": "a",
            "start": datetime.datetime(2000, 1, 1),
            "end": datetime.datetime(2000, 1, 1, 0, 0, 1),
        },
        {
            "task": "b",
            "start": datetime.datetime(2000, 1, 1, 0, 0, 0, 500000),
            "end": datetime.datetime(2000, 1, 1, 0, 0, 2),
        },
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
        {
            "task": "a",
            "start": datetime.datetime(2000, 1, 1),
            "end": datetime.datetime(2000, 1, 1, 0, 0, 1),
        },
    ]
    with patch.dict("sys.modules", {"plotly": None, "plotly.express": None}):
        fig = _build_gantt_chart(span_records, step=0)
    assert fig is None


def test_save_gantt_chart_html():
    """save_gantt_chart_html writes an HTML file when plotly is available."""
    window = IterationWindow()
    window.record_span("a", 100.0, 101.0)
    window.record_span("b", 100.5, 102.0)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = Path(f.name)

    save_gantt_chart_html(window, step=0, path=path)

    try:
        import plotly  # noqa: F401

        assert path.exists()
        content = path.read_text()
        assert "plotly" in content.lower() or "Plotly" in content
    except ImportError:
        # plotly not installed — file should not be created
        pass
    finally:
        path.unlink(missing_ok=True)


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
        first_ref = tinker.TrainingClient.forward_backward_async
        assert getattr(first_ref, "_scope_instrumented", False)

    # Second trace_init — should not re-wrap
    with trace_session():
        second_ref = tinker.TrainingClient.forward_backward_async
        assert getattr(second_ref, "_scope_instrumented", False)
        # Same wrapper object — not double-wrapped
        assert first_ref is second_ref


# --- Before/after telemetry verification ---


def test_metrics_keys_sync_rl_training():
    """Verify the metric keys produced by the sync RL training pattern.

    Simulates the do_sync_training iteration structure:
    trace_iteration wrapping scope_span("sampling"), scope_span("train"),
    plus @scope-decorated sub-functions.
    """

    @scope
    async def run_evaluations_parallel():
        await asyncio.sleep(0.01)

    @scope
    async def train_step():
        await asyncio.sleep(0.01)

    async def do_train_step_and_get_sampling_client():
        """Simulates the real function: train_step has @scope, no outer wrapper."""
        await train_step()

    async def run():
        with trace_iteration(step=0) as window:
            # Evals (via @scope on run_evaluations_parallel)
            await run_evaluations_parallel()

            # Sampling phase
            async with scope_span("sampling"):
                await asyncio.sleep(0.02)

            # Train phase (train_step has @scope, no call-site wrapper)
            await do_train_step_and_get_sampling_client()

        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()

    # Phase-level metrics
    assert "time/total" in metrics
    assert "time/sampling" in metrics
    # Sub-function metrics
    assert "time/run_evaluations_parallel" in metrics
    assert "time/train_step" in metrics
    # time/train is no longer emitted (was redundant with time/train_step)
    assert "time/train" not in metrics


def test_metrics_keys_streaming_rl_training():
    """Verify the metric keys produced by the streaming RL training pattern.

    Simulates do_sync_training_with_stream_minibatch: trace_iteration wrapping
    scope_span("run_evals") with per-substep spans inside.
    """

    async def run():
        with trace_iteration(step=0) as window:
            async with scope_span("run_evals"):
                await asyncio.sleep(0.01)

            # Simulate streaming train with per-substep spans
            async with scope_span("train"):
                async with scope_span("train/fwd_bwd_substep_0_mb_0_enqueue"):
                    await asyncio.sleep(0.01)
                async with scope_span("train/optim_substep_0_enqueue"):
                    await asyncio.sleep(0.01)
                async with scope_span("train/fwd_bwd_substep_0_mb_0_consume"):
                    await asyncio.sleep(0.01)
                async with scope_span("train/optim_substep_0_consume"):
                    await asyncio.sleep(0.01)

        return window

    window = asyncio.run(run())
    metrics = window.get_timing_metrics()

    assert "time/total" in metrics
    assert "time/run_evals" in metrics
    assert "time/train" in metrics
    assert "time/train/fwd_bwd_substep_0_mb_0_enqueue" in metrics
    assert "time/train/optim_substep_0_enqueue" in metrics
    assert "time/train/fwd_bwd_substep_0_mb_0_consume" in metrics
    assert "time/train/optim_substep_0_consume" in metrics


def test_metrics_keys_async_rl_training():
    """Verify the metric keys produced by the async RL training pattern.

    In async training, the training_loop gets its own trace_iteration per step,
    and the evaluation_loop gets a separate trace_iteration per eval.
    """

    @scope
    async def train_step():
        await asyncio.sleep(0.01)

    async def run():
        # Training loop iteration
        with trace_iteration(step=0) as train_window:
            await train_step()

        # Evaluation loop iteration
        with trace_iteration(step=0) as eval_window:
            async with scope_span("run_evals"):
                await asyncio.sleep(0.01)

        return train_window, eval_window

    train_window, eval_window = asyncio.run(run())

    train_metrics = train_window.get_timing_metrics()
    assert "time/total" in train_metrics
    assert "time/train_step" in train_metrics

    eval_metrics = eval_window.get_timing_metrics()
    assert "time/total" in eval_metrics
    assert "time/run_evals" in eval_metrics


def test_gantt_chart_sync_rl_layout():
    """Verify Gantt chart spans for the sync RL training pattern."""
    window = IterationWindow()
    # Simulate: evals, then sampling (wide, overlapping sub-spans), then train_step
    window.record_span("run_evaluations_parallel", 0.0, 1.0)
    window.record_span("sampling", 1.0, 5.0)
    # Overlapping rollout spans within sampling
    window.record_span("do_group_rollout_and_filter_constant_reward", 1.0, 3.5)
    window.record_span("do_group_rollout_and_filter_constant_reward", 1.1, 4.0)
    window.record_span("do_group_rollout_and_filter_constant_reward", 1.2, 5.0)
    window.record_span("train_step", 5.0, 6.8)
    window.record_span("save_checkpoint_and_get_sampling_client", 6.9, 7.0)

    records = window.get_span_records()
    task_names = [r["task"] for r in records]
    assert "sampling" in task_names
    assert "train_step" in task_names
    assert "run_evaluations_parallel" in task_names

    # Verify aggregation: rollout spans appear with :total (3 calls)
    metrics = window.aggregate()
    assert metrics["time/do_group_rollout_and_filter_constant_reward:count"] == 3
    assert "time/do_group_rollout_and_filter_constant_reward:total" in metrics
    # Phase-level spans are single calls
    assert "time/sampling" in metrics
    assert "time/train_step" in metrics

    # Gantt chart renders without error
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = Path(f.name)
    try:
        save_gantt_chart_html(window, step=0, path=path)
        try:
            import plotly  # noqa: F401

            assert path.exists()
        except ImportError:
            pass
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    trace_init()
    asyncio.run(example_program())
    trace_shutdown()
