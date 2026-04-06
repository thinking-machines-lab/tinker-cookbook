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
from collections.abc import Callable, Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

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
        """Convert the TraceEvent to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Chrome Trace / Perfetto-compatible event dict.
        """
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
    """Mutable context attached to a traced function call.

    Use :func:`get_scope_context` inside a ``@scope``-decorated function to
    retrieve the current ``ScopeContext`` and add custom key/value attributes
    that will be included in the Perfetto trace event's ``args``.

    Attributes:
        attributes (dict[str, Any]): User-defined key/value pairs to attach
            to the trace span.
    """

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
    """Collects span records during a single training iteration for aggregation.

    Use with :func:`trace_iteration` to automatically capture all ``@scope`` and
    ``scope_span`` timings within a training iteration. After the block exits,
    call :meth:`get_timing_metrics` for a flat dict of timing metrics ready to log.

    Example — GRPO training loop::

        for i_batch in range(n_batches):
            with trace_iteration(step=i_batch) as window:
                # All @scope-decorated calls inside this block are recorded
                await run_evals(sampling_client, ...)
                trajectory_groups = await gather_rollouts(sampling_client, ...)
                await train_step(training_client, trajectory_groups, ...)

            # Aggregated metrics: time/total, time/run_evals, time/sample_async:total, ...
            metrics.update(window.get_timing_metrics())

            # Persist per-span data for post-hoc analysis
            window.write_spans_jsonl(log_path / "timing_spans.jsonl", step=i_batch)

            # Optional: save a Gantt chart every K steps
            if i_batch % 10 == 0:
                save_gantt_chart_html(window, i_batch, log_path / f"gantt_{i_batch}.html")

            ml_logger.log_metrics(metrics, step=i_batch)
    """

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self._lock = threading.Lock()
        self._total_time: float | None = None

    def record_span(self, name: str, start_time: float, end_time: float) -> None:
        """Record a completed span into this window.

        Thread-safe. Wall-clock timestamps are derived from the monotonic
        ``start_time``/``end_time`` pair so spans from different threads or
        processes can be compared on a shared timeline.

        Args:
            name (str): Span name (typically the function or block label).
            start_time (float): ``time.perf_counter()`` value at span start.
            end_time (float): ``time.perf_counter()`` value at span end.
        """
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
        """Aggregate collected spans into a flat timing dict.

        Single-occurrence spans produce ``time/{name}``. Multi-occurrence
        spans produce ``time/{name}:total``, ``time/{name}:count``,
        ``time/{name}:mean``, and ``time/{name}:max``.

        Returns:
            dict[str, float]: Flat dictionary of timing metrics.
        """
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

    def get_timing_metrics(self) -> dict[str, float]:
        """Get aggregated timing metrics including ``time/total``.

        Call this after the :func:`trace_iteration` context manager has exited,
        which sets the total wall-clock time for the iteration.

        Returns:
            dict[str, float]: Aggregated span metrics plus ``time/total``
                if the iteration has completed.
        """
        metrics = self.aggregate()
        if self._total_time is not None:
            metrics["time/total"] = self._total_time
        return metrics

    def merge_spans(self, spans: list[SpanRecord]) -> None:
        """Merge externally-collected spans (e.g. from worker processes) into this window.

        Args:
            spans (list[SpanRecord]): Span records collected elsewhere to
                incorporate into this window's aggregation.
        """
        with self._lock:
            self.spans.extend(spans)

    def get_span_records(self) -> list[dict[str, Any]]:
        """Get span records formatted for Gantt chart rendering.

        Uses wall-clock timestamps (``time.time``) so that spans from different
        processes can be placed on a shared timeline.

        Returns:
            list[dict[str, Any]]: List of dicts with keys ``task``, ``start``,
                and ``end`` suitable for passing to ``plotly.express.timeline``.
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

    def get_span_dicts(self) -> list[dict[str, Any]]:
        """Return span records as JSON-serializable dicts.

        Wall times are normalized relative to the earliest span.

        Returns:
            List of dicts with keys: ``name``, ``duration``, ``wall_start``, ``wall_end``.
        """
        with self._lock:
            spans = list(self.spans)

        if not spans:
            return []

        t0 = min(s.wall_start for s in spans)
        return [
            {
                "name": s.name,
                "duration": s.end_time - s.start_time,
                "wall_start": s.wall_start - t0,
                "wall_end": s.wall_end - t0,
            }
            for s in spans
        ]

    def write_spans_jsonl(self, path: Path | str, step: int) -> None:
        """Append span records for this iteration as one JSON line to the given file.

        Format: ``{"step": N, "spans": [{"name": ..., "duration": ..., "wall_start": ..., "wall_end": ...}, ...]}``

        .. deprecated::
            Prefer ``store.write_timing_spans(step, window.get_span_dicts())``
            which goes through the Storage protocol.

        Args:
            path (Path | str): File path to append to (created if missing).
            step (int): Training step number to include in the record.
        """
        span_dicts = self.get_span_dicts()
        if not span_dicts:
            return
        line = json.dumps({"step": step, "spans": span_dicts})
        with open(path, "a") as f:
            f.write(line + "\n")

    def save_timing(self, step: int, *, store: "TrainingRunStore | None") -> None:
        """Write timing spans to ``timing_spans.jsonl`` via the store.

        Args:
            step: Training step number.
            store: A ``TrainingRunStore``, or ``None`` to skip.
        """
        if store is not None:
            store.write_timing_spans(step, self.get_span_dicts())


# Context variable to track the current iteration window
_iteration_window: ContextVar[IterationWindow | None] = ContextVar(
    "_iteration_window", default=None
)


class TraceCollector:
    """Collects trace events and exports them in Chrome Trace/Perfetto Format.

    Events are queued in-memory and flushed to a JSONL file by a background
    daemon thread.  Call :meth:`shutdown` (or let ``atexit`` handle it) to
    flush remaining events and join the thread.

    Args:
        flush_interval_sec (float): Maximum seconds between disk flushes.
        output_file (str): Path for the JSONL output file.
    """

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

    def add_event(self, event: TraceEvent) -> None:
        """Enqueue a trace event for asynchronous flushing to disk.

        Args:
            event (TraceEvent): The event to record.
        """
        self.event_queue.put(event)

    def get_timestamp(self) -> float:
        """Get the current high-resolution timestamp in microseconds.

        Returns:
            float: Microseconds from ``time.perf_counter``.
        """
        return time.perf_counter() * 1e6

    def get_all_events_immediately_available(self) -> list[TraceEvent]:
        """Drain all events currently in the queue without blocking.

        Returns:
            list[TraceEvent]: All events that were available at call time.
        """
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

    def shutdown(self) -> None:
        """Shutdown the background flusher thread and flush remaining events."""
        self.shutdown_event.set()
        self.flusher_thread.join(timeout=5.0)


# Global trace collector instance
_trace_collector: TraceCollector | None = None


def _atexit_trace_shutdown():
    global _trace_collector
    if _trace_collector is not None:
        _trace_collector.shutdown()
        _trace_collector = None


atexit.register(_atexit_trace_shutdown)


def _instrument_sdk_clients() -> None:
    """Patch Tinker SDK client classes with @scope for automatic tracing."""
    import tinker

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
) -> None:
    """Initialize the trace collector.

    Args:
        flush_interval_sec: How often to flush trace events to disk.
        output_file: Path for Perfetto trace output (JSONL format).
    """
    global _trace_collector
    _trace_collector = TraceCollector(flush_interval_sec, output_file)
    _instrument_sdk_clients()


def trace_shutdown() -> None:
    """Shutdown the trace collector and flush any remaining events."""
    global _trace_collector
    if _trace_collector is None:
        return
    _trace_collector.shutdown()
    _trace_collector = None


@dataclass
class FunctionCallContext:
    """Context information captured at the start of a traced function call.

    Attributes:
        scope_context (ScopeContext): Mutable context for user-defined attributes.
        coroutine_name (str): Name of the asyncio task or ``sync:{thread_name}``.
        thread_name (str): Name of the OS thread executing the call.
        category (str): Trace event category (typically ``"async"``).
        thread_id (int): OS thread identifier used as the Perfetto process ID.
    """

    scope_context: ScopeContext
    coroutine_name: str
    thread_name: str
    category: str
    thread_id: int


@dataclass
class CreateTraceEventsResult:
    """Bundle of trace events and context created when a span begins.

    Attributes:
        begin_event (TraceEvent): The ``B`` (begin) event for the span.
        metadata_coroutine_event (TraceEvent): Metadata naming the coroutine track.
        metadata_thread_event (TraceEvent): Metadata naming the thread group.
        function_call_context (FunctionCallContext): Context to carry through
            the span's lifetime and use when emitting the end event.
    """

    begin_event: TraceEvent
    metadata_coroutine_event: TraceEvent
    metadata_thread_event: TraceEvent
    function_call_context: FunctionCallContext


def _get_trace_thread_info() -> tuple[int, str, str]:
    """Get thread/coroutine info for trace events.

    Returns (thread_id, thread_name, coroutine_name).
    """
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
    return thread_id, thread_name, coroutine_name


def _create_trace_events(name: str) -> CreateTraceEventsResult:
    """Create trace events and context information for a named span."""
    assert _trace_collector is not None, (
        "Trace collector must be initialized before creating trace events"
    )

    thread_id, thread_name, coroutine_name = _get_trace_thread_info()
    category = "async"

    # Begin event for this function call
    begin_event = TraceEvent(
        name=name,
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
    name: str,
    function_call_context: FunctionCallContext,
) -> TraceEvent:
    """Create an end trace event for a named span."""
    assert _trace_collector is not None, (
        "Trace collector must be initialized before creating trace events"
    )

    return TraceEvent(
        name=name,
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


def _make_scope_wrapper(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Create a scope wrapper for a function with the given span name."""

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
                        window.record_span(name, t_start, time.perf_counter())
                return await func(*args, **kwargs)

            events_result = _create_trace_events(name)
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
                end_event = _create_end_event(name, events_result.function_call_context)
                _trace_collector.add_event(end_event)

                # Record into iteration window if active
                window = _iteration_window.get(None)
                if window is not None:
                    window.record_span(name, t_start, time.perf_counter())

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
                        window.record_span(name, t_start, time.perf_counter())
                return func(*args, **kwargs)

            events_result = _create_trace_events(name)
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
                end_event = _create_end_event(name, events_result.function_call_context)
                _trace_collector.add_event(end_event)

                # Record into iteration window if active
                window = _iteration_window.get(None)
                if window is not None:
                    window.record_span(name, t_start, time.perf_counter())

                # Reset context
                if token is not None:
                    trace_context.reset(token)

        return sync_wrapper


def scope(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for tracing both async and sync functions.

    Wraps *func* so that each call emits begin/end events to the Perfetto
    trace (if :func:`trace_init` was called) and records a span in the
    current :class:`IterationWindow` (if inside :func:`trace_iteration`).

    In the resulting trace:

    - Each track represents a coroutine (or sync call on a thread).
    - A thread is a group of tracks for all coroutines on that thread.

    For better tracking, name ``asyncio`` tasks so they can be grouped
    properly in the trace viewer.

    Args:
        func (Callable[..., Any]): The function to instrument.

    Returns:
        Callable[..., Any]: Wrapped function with identical signature.

    Example::

        from tinker_cookbook.utils.trace import scope, trace_init, get_scope_context

        @scope
        async def foo():
            await asyncio.sleep(0.1)
            context = get_scope_context()
            context.attributes["foo"] = 1
            await bar()

        @scope
        async def bar():
            await asyncio.gather(
                asyncio.create_task(baz(), name="baz"),
                asyncio.create_task(baz(), name="baz2"),
            )

        if __name__ == "__main__":
            trace_init()
            asyncio.run(foo())
    """
    return _make_scope_wrapper(func, func.__name__)


@contextlib.asynccontextmanager
async def scope_span(name: str):
    """Async context manager for inline named spans.

    Records to both the Perfetto trace (if active) and the current IterationWindow.
    Use this when you want to time a block of code with a semantic name rather than
    decorating a function with ``@scope``.

    Example::

        async with scope_span("policy_sample"):
            result = await policy(observation, stop_condition)
    """
    window = _iteration_window.get(None)

    if _trace_collector is not None:
        events_result = _create_trace_events(name)
        _trace_collector.add_event(events_result.begin_event)
        _trace_collector.add_event(events_result.metadata_coroutine_event)
        _trace_collector.add_event(events_result.metadata_thread_event)

        t_start = time.perf_counter()
        try:
            yield
        finally:
            end_event = _create_end_event(name, events_result.function_call_context)
            _trace_collector.add_event(end_event)
            if window is not None:
                window.record_span(name, t_start, time.perf_counter())
    elif window is not None:
        t_start = time.perf_counter()
        try:
            yield
        finally:
            window.record_span(name, t_start, time.perf_counter())
    else:
        yield


@contextlib.contextmanager
def scope_span_sync(name: str):
    """Sync context manager for inline named spans.

    Same as ``scope_span`` but for synchronous code.

    Example::

        with scope_span_sync("data_processing"):
            result = process_data(batch)
    """
    window = _iteration_window.get(None)

    if _trace_collector is not None:
        events_result = _create_trace_events(name)
        _trace_collector.add_event(events_result.begin_event)
        _trace_collector.add_event(events_result.metadata_coroutine_event)
        _trace_collector.add_event(events_result.metadata_thread_event)

        t_start = time.perf_counter()
        try:
            yield
        finally:
            end_event = _create_end_event(name, events_result.function_call_context)
            _trace_collector.add_event(end_event)
            if window is not None:
                window.record_span(name, t_start, time.perf_counter())
    elif window is not None:
        t_start = time.perf_counter()
        try:
            yield
        finally:
            window.record_span(name, t_start, time.perf_counter())
    else:
        yield


def get_scope_context() -> ScopeContext:
    """Return the :class:`ScopeContext` for the current ``@scope``-decorated call.

    Use this inside a traced function to attach custom attributes that will
    appear in the Perfetto trace event's ``args`` dict.

    Returns:
        ScopeContext: The mutable context for the current span.

    Example::

        @scope
        async def foo():
            context = get_scope_context()
            context.attributes["batch_size"] = 32
            context.attributes["epoch"] = 5
            await bar()
    """

    result = trace_context.get(ScopeContext())
    assert result is not None, "Trace context is not set"
    return result


def update_scope_context(values: dict[str, Any]) -> None:
    """Convenience helper to merge key/value pairs into the current scope's context.

    Equivalent to ``get_scope_context().attributes.update(values)``.

    Args:
        values (dict[str, Any]): Key/value pairs to add to the current span's
            trace attributes.

    Example::

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


def save_gantt_chart_html(window: IterationWindow, step: int, path: Path | str) -> None:
    """Build a Plotly Gantt chart from the window's spans and save as standalone HTML.

    No-op if plotly is not installed or the window has no spans.

    Args:
        window (IterationWindow): The iteration window containing span data.
        step (int): Training step number, used in the chart title.
        path (Path | str): Output file path for the HTML chart.
    """
    span_records = window.get_span_records()
    fig = _build_gantt_chart(span_records, step)
    if fig is not None:
        fig.write_html(str(path))


@contextlib.contextmanager
def trace_iteration(step: int) -> Generator[IterationWindow, None, None]:
    """Context manager that marks a training iteration boundary.

    Yields an ``IterationWindow`` that collects all ``@scope`` and ``scope_span``
    spans within the block. After the block exits, call ``window.get_timing_metrics()``
    to retrieve the aggregated timing dict (including ``time/total``).

    Span names are flat (the function or span name), not hierarchical. If ``train_step``
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
            with trace_iteration(step=i_batch) as window:
                await run_evals(...)
                await gather_rollouts(...)
                await train_step(...)
            metrics.update(window.get_timing_metrics())
            window.write_spans_jsonl(log_path / "timing_spans.jsonl", step=i_batch)
            ml_logger.log_metrics(metrics, step=i_batch)
    """
    window = IterationWindow()
    token = _iteration_window.set(window)
    t_start = time.perf_counter()
    try:
        yield window
    finally:
        window._total_time = time.perf_counter() - t_start
        _iteration_window.reset(token)


def convert_jsonl_to_json_main() -> None:
    """CLI entry point to convert trace JSONL to JSON for Chrome Tracing / Perfetto.

    Reads a JSONL trace file (one event per line) and writes a single JSON
    array suitable for loading in ``chrome://tracing`` or ``https://ui.perfetto.dev/``.
    """
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
