import argparse
import asyncio
import atexit
import contextlib
import datetime
import functools
import inspect
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Chrome Trace/Perfetto Event type"""

    BEGIN = "B"
    END = "E"
    METADATA = "M"


@dataclass
class TraceEvent:
    """Represents a trace event in Chrome Trace/Perfetto Format"""

    name: str
    ph: EventType
    pid: int
    tid: int
    ts: float
    args: dict[str, Any] = field(default_factory=dict)
    cat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the TraceEvent to a dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "ph": self.ph.value,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
            "args": self.args,
        }
        if self.cat is not None:
            result["cat"] = self.cat
        return result


@dataclass
class ScopeContext:
    # Additional attributes to log into the trace for this function call
    attributes: dict[str, Any] = field(default_factory=dict)


# Context variable to track the current coroutine's trace context
trace_context: ContextVar[ScopeContext | None] = ContextVar("trace_context", default=None)


@dataclass
class SpanRecord:
    """A recorded span within an iteration window.

    We store two sets of timestamps:
    - ``start_time`` / ``end_time``: from ``time.perf_counter()``, used for duration
      calculations (aggregation metrics). High resolution but process-local — values
      cannot be compared across processes.
    - ``wall_start`` / ``wall_end``: from ``time.time()``, used for positioning spans
      on Gantt charts. Synchronized across processes on the same machine, so spans
      from multiprocess workers (ProcessPoolExecutor, Ray) can be placed on a shared
      timeline without clock alignment.
    """

    name: str
    start_time: float  # seconds (perf_counter, process-local)
    end_time: float  # seconds (perf_counter, process-local)
    wall_start: float  # seconds since epoch (time.time, cross-process comparable)
    wall_end: float  # seconds since epoch (time.time, cross-process comparable)


class IterationWindow:
    """Collects span records during a single training iteration for aggregation."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self._lock = threading.Lock()

    def record_span(self, name: str, start_time: float, end_time: float) -> None:
        with self._lock:
            self.spans.append(
                SpanRecord(
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    wall_start=time.time() - (time.perf_counter() - start_time),
                    wall_end=time.time() - (time.perf_counter() - end_time),
                )
            )

    def aggregate(self) -> dict[str, float]:
        """Aggregate collected spans into a flat timing dict."""
        with self._lock:
            spans = list(self.spans)

        if not spans:
            return {}

        # Group durations by name
        durations_by_name: dict[str, list[float]] = defaultdict(list)
        for span in spans:
            durations_by_name[span.name].append(span.end_time - span.start_time)

        metrics: dict[str, float] = {}
        for name, durations in durations_by_name.items():
            if len(durations) == 1:
                # Single call: just report the duration
                metrics[f"time/{name}"] = durations[0]
            else:
                # Multiple calls: report aggregates
                metrics[f"time/{name}:total"] = sum(durations)
                metrics[f"time/{name}:count"] = len(durations)
                metrics[f"time/{name}:mean"] = sum(durations) / len(durations)
                metrics[f"time/{name}:max"] = max(durations)

        return metrics

    def merge_spans(self, spans: list[SpanRecord]) -> None:
        """Merge externally-collected spans (e.g. from worker processes) into this window."""
        with self._lock:
            self.spans.extend(spans)

    def get_span_records(self) -> list[dict[str, Any]]:
        """Get span records for Gantt chart rendering.

        Uses wall-clock timestamps (time.time) so that spans from different
        processes can be placed on a shared timeline.
        """
        with self._lock:
            spans = list(self.spans)

        if not spans:
            return []

        # Use wall-clock times for positioning — comparable across processes
        t0 = min(s.wall_start for s in spans)
        return [
            {
                "task": s.name,
                "start": datetime.datetime(2000, 1, 1)
                + datetime.timedelta(seconds=s.wall_start - t0),
                "end": datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=s.wall_end - t0),
            }
            for s in spans
        ]


# Context variable to track the current iteration window
_iteration_window: ContextVar[IterationWindow | None] = ContextVar(
    "_iteration_window", default=None
)


class TraceCollector:
    """Collects trace events and exports them in Chrome Trace/Perfetto Format."""

    def __init__(self, flush_interval_sec: float = 1.0, output_file: str = "trace_events.jsonl"):
        self.event_queue: queue.Queue[TraceEvent] = queue.Queue()
        self.flush_interval_sec = flush_interval_sec
        self.output_file = output_file
        self.shutdown_event = threading.Event()
        self.flusher_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flusher_thread.start()

        # Map of (pid, tid) to metadata event
        self.metadata_events: dict[tuple[int, int], TraceEvent] = {}
        self.next_fake_pid = 0
        self.thread_id_to_fake_pid: dict[int, int] = {}

    def add_event(self, event: TraceEvent):
        """Thread-safe addition of trace events."""
        self.event_queue.put(event)

    def get_timestamp(self) -> float:
        """Get current timestamp in microseconds relative to start."""
        return time.perf_counter() * 1e6

    def get_all_events_immediately_available(self) -> list[TraceEvent]:
        """Get all events that are immediately available."""
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _write_events(self, events: list[TraceEvent], f: TextIOWrapper) -> None:
        for event in events:
            # Map the event pids (thread ids) to fake pids. If pid numbers are large,
            # Perfetto has issues rendering these as different groups of tracks
            if event.pid not in self.thread_id_to_fake_pid:
                self.thread_id_to_fake_pid[event.pid] = self.next_fake_pid
                self.next_fake_pid += 1
            event.pid = self.thread_id_to_fake_pid[event.pid]

            # Only log the first metadata event for each pid/tid pair
            if event.ph == EventType.METADATA:
                if (event.pid, event.tid) in self.metadata_events:
                    continue
                self.metadata_events[(event.pid, event.tid)] = event

            json.dump(event.to_dict(), f)
            f.write("\n")
        f.flush()

    def _flush_worker(self):
        """Background thread worker that periodically flushes events to file."""
        # Use append mode to avoid overwriting previous events when resuming
        # from a checkpoint
        with open(self.output_file, "a") as f:
            while not self.shutdown_event.is_set():
                events_to_write = self.get_all_events_immediately_available()

                # Collect events with a timeout to check shutdown periodically
                try:
                    # Get first event with timeout and any additional events that are immediately available
                    event = self.event_queue.get(timeout=self.flush_interval_sec)
                    events_to_write.append(event)
                    events_to_write.extend(self.get_all_events_immediately_available())
                except queue.Empty:
                    # No events to flush, continue checking for shutdown
                    continue
                self._write_events(events_to_write, f)

            # Flush remaining events on shutdown
            self._write_events(self.get_all_events_immediately_available(), f)

    def shutdown(self):
        """Shutdown the background flusher thread."""
        self.shutdown_event.set()
        self.flusher_thread.join(timeout=5.0)


# Global trace collector instance
_trace_collector: TraceCollector | None = None

# Global logger for auto-logging timing metrics
_wandb_logger: Any = None  # ml_log.Logger instance
_span_chart_every: int = 0  # Log Gantt chart every N steps (0 = disabled)


def _atexit_trace_shutdown():
    global _trace_collector
    if _trace_collector is not None:
        _trace_collector.shutdown()
        _trace_collector = None


atexit.register(_atexit_trace_shutdown)


def _instrument_sdk_clients() -> None:
    """Patch Tinker SDK client classes with @scope for automatic tracing."""
    try:
        import tinker
    except ImportError:
        logger.debug("tinker SDK not installed, skipping client instrumentation")
        return

    # TrainingClient methods
    _methods_to_patch = {
        tinker.TrainingClient: [
            "forward_async",
            "forward_backward_async",
            "forward_backward_custom_async",
            "get_info_async",
            "optim_step_async",
            "save_state_async",
            "load_state_async",
            "load_state_with_optimizer_async",
            "save_weights_for_sampler_async",
            "save_weights_and_get_sampling_client_async",
            "create_sampling_client_async",
        ],
        tinker.SamplingClient: [
            "sample_async",
            "compute_logprobs_async",
            "get_base_model_async",
        ],
    }

    for cls, method_names in _methods_to_patch.items():
        for method_name in method_names:
            if hasattr(cls, method_name):
                original = getattr(cls, method_name)
                # Avoid double-wrapping
                if not getattr(original, "_scope_instrumented", False):
                    wrapped = scope(original)
                    wrapped._scope_instrumented = True  # type: ignore[attr-defined]
                    setattr(cls, method_name, wrapped)


def trace_init(
    flush_interval_sec: float = 1.0,
    output_file: str = "trace_events.jsonl",
    wandb_logger: Any = None,
    span_chart_every: int = 0,
) -> None:
    """Initialize the trace collector.

    Args:
        flush_interval_sec: How often to flush trace events to disk.
        output_file: Path for Perfetto trace output (JSONL format).
        wandb_logger: Optional ml_log.Logger instance. When provided, trace_iteration()
            will auto-log timing metrics and SDK client classes will be auto-instrumented.
        span_chart_every: Log a Plotly Gantt chart every N steps (0 = disabled).
            Requires plotly to be installed.
    """
    global _trace_collector, _wandb_logger, _span_chart_every
    _trace_collector = TraceCollector(flush_interval_sec, output_file)
    _wandb_logger = wandb_logger
    _span_chart_every = span_chart_every
    _instrument_sdk_clients()


def trace_shutdown() -> None:
    """Shutdown the trace collector and flush any remaining events."""
    global _trace_collector, _wandb_logger, _span_chart_every
    if _trace_collector is None:
        return
    _trace_collector.shutdown()
    _trace_collector = None
    _wandb_logger = None
    _span_chart_every = 0


@dataclass
class FunctionCallContext:
    """Context information for a function call"""

    scope_context: ScopeContext
    coroutine_name: str
    thread_name: str
    category: str
    thread_id: int


@dataclass
class CreateTraceEventsResult:
    begin_event: TraceEvent
    metadata_coroutine_event: TraceEvent
    metadata_thread_event: TraceEvent
    function_call_context: FunctionCallContext


def _create_trace_events(func: Callable[..., Any]) -> CreateTraceEventsResult:
    """Create trace events and context information for a function call."""
    assert _trace_collector is not None, (
        "Trace collector must be initialized before creating trace events"
    )

    # Get current task and thread info
    thread_id = threading.current_thread().ident or 0
    thread_name = threading.current_thread().name
    try:
        task = asyncio.current_task()
        if task is None:
            coroutine_name = f"sync:{thread_name}"
        else:
            coroutine_name = task.get_name()
    except RuntimeError:
        coroutine_name = f"sync:{thread_name}"
    thread_id = threading.current_thread().ident or 0
    thread_name = threading.current_thread().name
    category = "async"

    # Begin event for this function call
    begin_event = TraceEvent(
        name=func.__name__,
        ph=EventType.BEGIN,
        pid=thread_id,  # Process ID (we use thread ID as process)
        tid=hash(coroutine_name) % 1000000,  # Track ID within the thread
        ts=_trace_collector.get_timestamp(),
        args={
            "track": coroutine_name,
            "thread": thread_name,
        },
        cat=category,
    )

    # Metadata events to identify the track names.
    # In typical perfetto setups, a process has a group of tracks, where each track represnets a thread.
    # In our case, a group of tracks represents a thread, and a track represents a coroutine running
    # on that thread.
    metadata_coroutine_event = TraceEvent(
        name="thread_name",
        ph=EventType.METADATA,
        pid=thread_id,
        tid=hash(coroutine_name) % 1000000,
        ts=0,
        args={"name": coroutine_name},
    )
    metadata_thread_event = TraceEvent(
        name="process_name",
        ph=EventType.METADATA,
        pid=thread_id,
        tid=0,
        ts=0,
        args={"name": f"{thread_name} Thread"},
    )

    return CreateTraceEventsResult(
        begin_event,
        metadata_coroutine_event,
        metadata_thread_event,
        FunctionCallContext(
            scope_context=ScopeContext(),
            coroutine_name=coroutine_name,
            thread_name=thread_name,
            category=category,
            thread_id=thread_id,
        ),
    )


def _create_end_event(
    func: Callable[..., Any],
    function_call_context: FunctionCallContext,
) -> TraceEvent:
    """Create an end trace event for a function call."""
    assert _trace_collector is not None, (
        "Trace collector must be initialized before creating trace events"
    )

    return TraceEvent(
        name=func.__name__,
        ph=EventType.END,
        pid=function_call_context.thread_id,
        tid=hash(function_call_context.coroutine_name) % 1000000,
        ts=_trace_collector.get_timestamp(),
        args={
            "track": function_call_context.coroutine_name,
            "thread": function_call_context.thread_name,
            **function_call_context.scope_context.attributes,
        },
        cat=function_call_context.category,
    )


def scope(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for tracing both async and sync functions. In the resulting trace:
    - Each track represents a coroutine (or a sync function if not a coroutine)
    - A thread is a group of tracks, representing all the coroutines running on that thread

    For better tracking, make sure to name all coroutines so that we can group them
    properly in the trace.

    Example usage:

    from tinker_cookbook.utils.trace import scope, trace_init, get_scope_context

    @scope
    async def foo():
        await asyncio.sleep(0.1)
        # Log additional attributes for this function call into the trace
        context = get_scope_context()
        context.attributes["foo"] = 1
        context.attributes["foo2"] = "abc"
        await bar()

    @scope
    async def bar():
        # Name the coroutines so that we can group them properly in the trace
        await asyncio.gather(
            asyncio.create_task(baz(), name="baz"),
            asyncio.create_task(baz(), name="baz2"),
        )

    @scope
    async def main():
        await foo()

    if __name__ == "__main__":
        trace_init()
        asyncio.run(main())
    """

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            if _trace_collector is None:
                # Still record into iteration window even without Perfetto tracing
                window = _iteration_window.get(None)
                if window is not None:
                    t_start = time.perf_counter()
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        window.record_span(func.__name__, t_start, time.perf_counter())
                return await func(*args, **kwargs)

            events_result = _create_trace_events(func)
            _trace_collector.add_event(events_result.begin_event)
            _trace_collector.add_event(events_result.metadata_coroutine_event)
            _trace_collector.add_event(events_result.metadata_thread_event)

            t_start = time.perf_counter()
            token = None
            try:
                # Set context for nested calls
                token = trace_context.set(events_result.function_call_context.scope_context)

                # Execute the actual function
                result = await func(*args, **kwargs)
                return result

            finally:
                end_event = _create_end_event(func, events_result.function_call_context)
                _trace_collector.add_event(end_event)

                # Record into iteration window if active
                window = _iteration_window.get(None)
                if window is not None:
                    window.record_span(func.__name__, t_start, time.perf_counter())

                # Reset context
                if token is not None:
                    trace_context.reset(token)

        return async_wrapper

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            if _trace_collector is None:
                # Still record into iteration window even without Perfetto tracing
                window = _iteration_window.get(None)
                if window is not None:
                    t_start = time.perf_counter()
                    try:
                        return func(*args, **kwargs)
                    finally:
                        window.record_span(func.__name__, t_start, time.perf_counter())
                return func(*args, **kwargs)

            events_result = _create_trace_events(func)
            _trace_collector.add_event(events_result.begin_event)
            _trace_collector.add_event(events_result.metadata_coroutine_event)
            _trace_collector.add_event(events_result.metadata_thread_event)

            t_start = time.perf_counter()
            token = None
            try:
                # Set context for nested calls
                token = trace_context.set(events_result.function_call_context.scope_context)

                # Execute the actual function
                result = func(*args, **kwargs)
                return result

            finally:
                end_event = _create_end_event(func, events_result.function_call_context)
                _trace_collector.add_event(end_event)

                # Record into iteration window if active
                window = _iteration_window.get(None)
                if window is not None:
                    window.record_span(func.__name__, t_start, time.perf_counter())

                # Reset context
                if token is not None:
                    trace_context.reset(token)

        return sync_wrapper


def get_scope_context() -> ScopeContext:
    """
    Call this to get the current scope's context. This allows the functions
    to log additional attributes into the trace.

    Example usage:

    @scope
    async def foo():
        context = get_scope_context()
        context.attributes["foo"] = 1
        context.attributes["foo2"] = "abc"
        await bar()
    """

    result = trace_context.get(ScopeContext())
    assert result is not None, "Trace context is not set"
    return result


def update_scope_context(values: dict[str, Any]) -> None:
    """Update the current scope's context. Example usage:

    @scope
    async def foo(step: int):
        update_scope_context({"step": step})
        await bar()

    """
    result = trace_context.get(ScopeContext())
    assert result is not None, "Trace context is not set"
    result.attributes.update(values)


def _build_gantt_chart(span_records: list[dict[str, Any]], step: int) -> Any:
    """Build a Plotly Gantt chart from span records. Returns a plotly Figure or None."""
    try:
        import plotly.express as px  # type: ignore[reportMissingImports]
    except ImportError:
        logger.debug("plotly not installed, skipping Gantt chart")
        return None

    if not span_records:
        return None

    fig = px.timeline(
        span_records,
        x_start="start",
        x_end="end",
        y="task",
        color="task",
        title=f"Iteration {step} — Span Timeline",
    )
    fig.update_layout(
        xaxis_title="Time (relative)",
        yaxis_title="",
        showlegend=False,
    )
    return fig


@contextlib.contextmanager
def trace_iteration(step: int) -> Generator[None, None, None]:
    """Context manager that marks a training iteration boundary.

    Collects all @scope spans within the window and, on exit:
    - Aggregates them into a flat timing dict and auto-logs to wandb_logger
    - Optionally logs a Plotly Gantt chart showing span overlap

    Span names are flat (``func.__name__``), not hierarchical. If ``train_step``
    calls ``forward_backward_async``, both appear as separate top-level keys::

        time/train_step = 5.0              # inclusive (contains forward_backward)
        time/forward_backward_async = 3.0  # just the inner call

    For functions called multiple times (e.g. 160 concurrent ``sample_async``
    calls), aggregated keys are produced::

        time/sample_async:total = 480.0
        time/sample_async:count = 160
        time/sample_async:mean  = 3.0
        time/sample_async:max   = 4.9

    Example::

        for i_batch in range(n_batches):
            with trace_iteration(step=i_batch):
                await run_evals(...)
                await gather_rollouts(...)
                await train_step(...)
            # time/* metrics auto-logged to wandb_logger
    """
    window = IterationWindow()
    token = _iteration_window.set(window)
    t_start = time.perf_counter()
    try:
        yield
    finally:
        total_time = time.perf_counter() - t_start
        _iteration_window.reset(token)

        # Aggregate timing metrics
        timing_metrics = window.aggregate()
        timing_metrics["time/total"] = total_time

        # Auto-log if wandb_logger is configured
        if _wandb_logger is not None:
            _wandb_logger.log_metrics(timing_metrics, step=step)

            # Log Gantt chart at configured intervals
            if _span_chart_every > 0 and step % _span_chart_every == 0:
                span_records = window.get_span_records()
                fig = _build_gantt_chart(span_records, step)
                if fig is not None:
                    try:
                        import wandb as _wandb  # type: ignore[reportMissingImports]

                        _wandb.log({"trace/spans": _wandb.Plotly(fig)}, step=step)
                    except ImportError:
                        logger.debug("wandb not installed, skipping Gantt chart logging")


def convert_jsonl_to_json_main():
    """Helper script to convert the trace events format into a visualizable format"""
    parser = argparse.ArgumentParser(
        description="Convert trace events from JSONL format to JSON format for visualization in chrome://tracing or https://ui.perfetto.dev/"
    )
    parser.add_argument("trace_events_jsonl_file", type=str)
    parser.add_argument("output_json_file", type=str)
    args = parser.parse_args()

    with open(args.trace_events_jsonl_file) as f:
        events = [json.loads(line) for line in f]
    with open(args.output_json_file, "w") as f:
        json.dump(events, f)
    print(f"""To view the trace:
1. Navigate to chrome://tracing or https://ui.perfetto.dev/
2. Load the trace file: {args.output_json_file}""")


if __name__ == "__main__":
    convert_jsonl_to_json_main()
