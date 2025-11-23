import argparse
import asyncio
import atexit
import functools
import inspect
import json
import queue
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from io import TextIOWrapper
from typing import Any, Callable


class EventType(str, Enum):
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


def _atexit_trace_shutdown():
    global _trace_collector
    if _trace_collector is not None:
        _trace_collector.shutdown()
        _trace_collector = None


atexit.register(_atexit_trace_shutdown)


def trace_init(
    flush_interval_sec: float = 1.0,
    output_file: str = "trace_events.jsonl",
) -> None:
    """Initialize the trace collector. Must be called before any trace events are created."""
    global _trace_collector
    _trace_collector = TraceCollector(flush_interval_sec, output_file)


def trace_shutdown() -> None:
    """Shutdown the trace collector and flush any remaining events."""
    global _trace_collector
    if _trace_collector is None:
        return
    _trace_collector.shutdown()
    _trace_collector = None


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
                return await func(*args, **kwargs)

            events_result = _create_trace_events(func)
            _trace_collector.add_event(events_result.begin_event)
            _trace_collector.add_event(events_result.metadata_coroutine_event)
            _trace_collector.add_event(events_result.metadata_thread_event)

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

                # Reset context
                if token is not None:
                    trace_context.reset(token)

        return async_wrapper

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            if _trace_collector is None:
                return func(*args, **kwargs)

            events_result = _create_trace_events(func)
            _trace_collector.add_event(events_result.begin_event)
            _trace_collector.add_event(events_result.metadata_coroutine_event)
            _trace_collector.add_event(events_result.metadata_thread_event)

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


def convert_jsonl_to_json_main():
    """Helper script to convert the trace events format into a visualizable format"""
    parser = argparse.ArgumentParser(
        description="Convert trace events from JSONL format to JSON format for visualization in chrome://tracing or https://ui.perfetto.dev/"
    )
    parser.add_argument("trace_events_jsonl_file", type=str)
    parser.add_argument("output_json_file", type=str)
    args = parser.parse_args()

    with open(args.trace_events_jsonl_file, "r") as f:
        events = [json.loads(line) for line in f]
    with open(args.output_json_file, "w") as f:
        json.dump(events, f)
    print(f"""To view the trace:
1. Navigate to chrome://tracing or https://ui.perfetto.dev/
2. Load the trace file: {args.output_json_file}""")


if __name__ == "__main__":
    convert_jsonl_to_json_main()
