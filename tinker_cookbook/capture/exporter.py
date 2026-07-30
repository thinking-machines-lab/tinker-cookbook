"""In-process capture export pipeline.

OTel-shaped batch pipeline, mirroring the conventions of
``tinker_cookbook.utils.trace.TraceCollector`` (bounded in-memory queue,
background daemon flusher thread, timer- and size-triggered flushes, atexit
shutdown) rather than refactoring it: trace.py stays untouched and this module
carries the extra semantics capture needs (drop-newest on a bounded queue with
a drop counter, ``force_flush``, idempotent ``shutdown``, pluggable sinks).

Records are plain JSON-serializable dicts. Every record carries a ``kind``
key (e.g. ``"sample"``, ``"train_op"``); sinks may use it for routing. The
default :class:`JsonlFileSink` appends records under a run directory as
``capture/<kind>.jsonl``, following ``tinker_cookbook.stores`` conventions
(all I/O through the ``Storage`` protocol).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import weakref
from collections.abc import Sequence
from typing import Any, Protocol

from tinker_cookbook.stores.storage import Storage, storage_join

logger = logging.getLogger(__name__)

#: During a sink outage the flusher retries every flush interval (~1/s); one
#: full traceback per retry is bounded but chatty on long outages, so after
#: the first failure only a summary line is emitted at most this often.
_FAILURE_REPORT_INTERVAL_SEC = 30.0

# Records are free-form JSON payloads whose shape varies by kind (built from
# arbitrary SDK metadata, serialized by sinks via json.dumps): the value type
# is genuinely unknowable here, so Any is the honest annotation.
CaptureRecord = dict[str, Any]


class CaptureSink(Protocol):
    """Destination for batches of capture records."""

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        """Export a batch of records.

        Args:
            records: The batch to export (never empty).
            timeout: Soft deadline in seconds; sinks doing I/O with a natural
                timeout knob (e.g. HTTP) should honor it, others may ignore it.
        """
        ...


class JsonlFileSink:
    """Sink that appends records as JSONL under a run directory.

    Records land at ``<prefix>/capture/<kind>.jsonl`` via the ``Storage``
    protocol, matching the run-centric layout used by
    ``tinker_cookbook.stores.training_store.TrainingRunStore``.
    """

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        self._storage = storage
        self._prefix = prefix

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        """Append each record to ``capture/<kind>.jsonl`` grouped by kind."""
        del timeout  # local appends have no meaningful deadline
        by_kind: dict[str, list[str]] = {}
        for record in records:
            kind = str(record.get("kind", "unknown"))
            by_kind.setdefault(kind, []).append(json.dumps(record, default=str))
        for kind, lines in by_kind.items():
            path = storage_join(self._prefix, "capture", f"{kind}.jsonl")
            self._storage.append(path, ("\n".join(lines) + "\n").encode("utf-8"))


class CaptureExporter:
    """Bounded-queue batch exporter with a background flusher thread.

    Semantics (OTel BatchProcessor-shaped):

    - ``enqueue`` never blocks and never raises; when the queue is full the
      newest record is dropped and :attr:`dropped` is incremented.
    - The flusher exports when ``max_batch_size`` records are buffered or
      ``flush_interval_sec`` has elapsed since the last export, whichever
      comes first; sub-threshold records wait out the interval so steady
      traffic is batched rather than exported one record per sink call.
    - ``force_flush(timeout)`` drains everything enqueued so far, waiting (up
      to ``timeout``) for any batch the flusher already holds in flight, so a
      True return means all records enqueued before the call reached the sink.
    - ``shutdown(timeout)`` is idempotent, flushes remaining records, joins the
      flusher thread, and is also registered with ``atexit``.
    - In-flight instrumented calls can be tracked via ``track_pending`` /
      ``pending_done``; ``wait_pending(timeout)`` blocks until all tracked
      calls completed (used for grace-draining futures at teardown).
    - Sink failures are logged and counted (:attr:`export_failures`), never
      raised into the caller.
    - At shutdown, if any loss counter (:attr:`dropped`,
      :attr:`export_failures`, :attr:`callback_failures`) is nonzero, a single
      warning summarizing the losses is logged; clean runs stay silent.
    """

    def __init__(
        self,
        sink: CaptureSink,
        *,
        max_queue_size: int = 4096,
        max_batch_size: int = 256,
        flush_interval_sec: float = 1.0,
        export_timeout_sec: float = 10.0,
    ) -> None:
        self._sink = sink
        self._max_queue_size = max_queue_size
        self._max_batch_size = max_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._export_timeout_sec = export_timeout_sec
        self._queue: queue.Queue[CaptureRecord] = queue.Queue(maxsize=max_queue_size)
        self._export_lock = threading.Lock()
        # Guards _in_flight, _pending, and the pop-from-queue step of the
        # flusher, so force_flush can wait for both an empty queue and no
        # batch held by the flusher.
        self._cond = threading.Condition()
        self._in_flight = False
        self._flush_now = False
        self._pending = 0
        self._shutdown_event = threading.Event()
        self._shutdown_done = False
        self._shutdown_lock = threading.Lock()
        self.dropped = 0
        self.export_failures = 0
        self.callback_failures = 0
        # Sink-failure log rate limiting (see _export): 0.0 means "no
        # ongoing outage" so the next failure logs a full traceback.
        self._last_failure_report = 0.0
        self._failures_since_report = 0
        self._flusher = threading.Thread(
            target=self._flush_worker, name="capture-exporter", daemon=True
        )
        self._flusher.start()
        atexit.register(self.shutdown)
        # Fork safety: threads do not survive fork(), so a forked child
        # would otherwise inherit an exporter whose flusher is silently
        # dead. Pattern from opentelemetry-python's BatchProcessor
        # (Apache-2.0; opentelemetry-sdk/src/opentelemetry/sdk/
        # _shared_internal/__init__.py, `_at_fork_reinit` /
        # os.register_at_fork via weakref so an already-collected exporter
        # does not leak through the process-global hook); reimplemented
        # here, no code copied. Not available on Windows.
        if hasattr(os, "register_at_fork"):
            self_ref: weakref.ReferenceType[CaptureExporter] = weakref.ref(self)

            def _after_fork_in_child() -> None:
                exporter = self_ref()
                if exporter is not None:
                    exporter._reinit_after_fork()

            os.register_at_fork(after_in_child=_after_fork_in_child)

    def _reinit_after_fork(self) -> None:
        """Rebuild threading state in a freshly forked child.

        Runs via ``os.register_at_fork(after_in_child=...)``. The inherited
        locks/condition may be held by threads that no longer exist, so they
        are recreated, and a new flusher thread is started (unless the
        exporter was already shut down before the fork).

        Inherited queued records are discarded (NOT counted as dropped),
        diverging from OTel (which lets the child re-export the inherited
        buffer): the parent process still owns those records and will export
        them, so exporting them from the child as well would produce
        duplicates in the store, and counting them as drops in the child
        would report phantom losses. The child starts with an empty queue,
        zeroed loss counters, and only exports what is enqueued in the child.
        """
        inherited = self._queue.qsize()
        self._queue = queue.Queue(maxsize=self._max_queue_size)
        if inherited:
            logger.debug(
                "discarded %d inherited capture records after fork; "
                "the parent process exports them",
                inherited,
            )
        # A child reports only its own lifetime: reset loss counters and the
        # failure-report rate-limit state inherited from the parent.
        self.dropped = 0
        self.export_failures = 0
        self.callback_failures = 0
        self._last_failure_report = 0.0
        self._failures_since_report = 0
        self._cond = threading.Condition()
        self._export_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._in_flight = False
        self._flush_now = False
        self._pending = 0
        if not self._shutdown_event.is_set():
            self._flusher = threading.Thread(
                target=self._flush_worker, name="capture-exporter", daemon=True
            )
            self._flusher.start()

    def enqueue(self, record: CaptureRecord) -> None:
        """Enqueue a record for export. Never blocks; drops (newest) when full.

        The shutdown check and the insert happen atomically under the
        condition: the shutdown transition is also taken under it, so an
        enqueue that passes the check cannot be descheduled and slip a
        record into a queue the flusher has already drained for the last
        time (the flusher's exit re-checks emptiness under the same
        condition).

        Deliberately a condition-variable design rather than OTel 1.34's
        lock-free ``deque(maxlen)`` + ``Event`` hot path: the deque shape
        cannot express exact drop-NEWEST (``deque(maxlen)`` drops oldest,
        and a len-check-then-append is not atomic), nor the
        enqueue/shutdown serialization above, nor force_flush's in-flight
        coverage, without reintroducing a lock anyway. Correctness over
        contention; the critical section is a bounded put_nowait.
        """
        with self._cond:
            if self._shutdown_event.is_set():
                self.dropped += 1
                return
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1
                return
            self._cond.notify_all()

    # ── pending-call tracking (used by capture.instrument) ────────────

    def track_pending(self) -> None:
        """Mark one instrumented call as in flight."""
        with self._cond:
            self._pending += 1

    def pending_done(self) -> None:
        """Mark one previously tracked call as completed."""
        with self._cond:
            self._pending -= 1
            self._cond.notify_all()

    def wait_pending(self, timeout: float | None = None) -> bool:
        """Wait until all tracked in-flight calls completed.

        Returns:
            True if the pending count reached zero within ``timeout``.
        """
        with self._cond:
            return self._cond.wait_for(lambda: self._pending <= 0, timeout)

    def _drain_nowait(self, limit: int | None = None) -> list[CaptureRecord]:
        records: list[CaptureRecord] = []
        while limit is None or len(records) < limit:
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return records

    def _export(self, records: list[CaptureRecord], timeout: float | None = None) -> bool:
        """Export a batch through the sink. Returns True on sink success."""
        if not records:
            return True
        with self._export_lock:
            try:
                effective = self._export_timeout_sec if timeout is None else timeout
                self._sink.export(records, timeout=effective)
                # Success ends any ongoing outage: the next failure gets a
                # full traceback again.
                self._last_failure_report = 0.0
                self._failures_since_report = 0
                return True
            except Exception as e:
                self.export_failures += 1
                now = time.monotonic()
                if self._last_failure_report == 0.0:
                    # First failure of an outage: full traceback.
                    self._last_failure_report = now
                    self._failures_since_report = 0
                    logger.warning(
                        "Capture sink export failed (%d records)", len(records), exc_info=True
                    )
                else:
                    self._failures_since_report += 1
                    if now - self._last_failure_report >= _FAILURE_REPORT_INTERVAL_SEC:
                        logger.warning(
                            "capture sink still failing (%d batches since last report): %s: %s",
                            self._failures_since_report,
                            type(e).__name__,
                            e,
                        )
                        self._last_failure_report = now
                        self._failures_since_report = 0
                return False

    def _flush_worker(self) -> None:
        while True:
            with self._cond:
                # Wait until the batch threshold is reached OR the flush
                # interval elapses (or shutdown / force_flush), so steady
                # low-rate traffic is batched instead of exported
                # record-by-record.
                deadline = time.monotonic() + self._flush_interval_sec
                while (
                    not self._shutdown_event.is_set()
                    and not self._flush_now
                    and self._queue.qsize() < self._max_batch_size
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
                # Pop under the condition so there is never a moment where
                # records are neither in the queue nor flagged in flight.
                batch = self._drain_nowait(limit=self._max_batch_size)
                self._in_flight = bool(batch)
            if batch:
                self._export(batch)
                with self._cond:
                    self._in_flight = False
                    self._cond.notify_all()
            elif self._shutdown_event.is_set():
                # Exit only when the queue is confirmed empty under the
                # condition: an enqueue that raced the shutdown transition
                # may have inserted after this iteration's drain, and both
                # inserts and the shutdown flag flip happen under the
                # condition, so this re-check cannot miss a record.
                with self._cond:
                    if self._queue.empty():
                        return

    def force_flush(self, timeout: float | None = None) -> bool:
        """Export everything enqueued so far, including the flusher's in-flight batch.

        All exporting happens on the flusher thread; this call only signals
        it to flush immediately and waits, so the wait is GENUINELY bounded
        by ``timeout`` even against a sink that ignores its (soft) timeout
        argument and blocks. On expiry this returns False while the export
        completes in the background; no records are lost, they are either in
        the queue or in the flusher's hands.

        Args:
            timeout: Maximum total seconds this call may block.

        Returns:
            True only if, within ``timeout``, the queue fully drained, no
            flusher batch remained in flight, and no export failed during
            the flush window; False on timeout or sink failure.
        """
        failures_before = self.export_failures
        with self._cond:
            self._flush_now = True
            self._cond.notify_all()
            try:
                completed = self._cond.wait_for(
                    lambda: self._queue.empty() and not self._in_flight, timeout
                )
            finally:
                self._flush_now = False
        return completed and self.export_failures == failures_before

    def shutdown(self, timeout: float = 5.0) -> None:
        """Flush remaining records and stop the flusher thread. Idempotent."""
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        # Flip the flag under the condition so it serializes with enqueue's
        # check-and-insert (see enqueue).
        with self._cond:
            self._shutdown_event.set()
            self._cond.notify_all()
        # One total budget across the join AND the follow-up flush wait, so
        # a stuck sink cannot make shutdown(t) block for ~2t.
        deadline = time.monotonic() + timeout
        self._flusher.join(timeout=timeout)
        # The flusher drains everything on shutdown; if the join timed out
        # (sink stuck), wait boundedly for the drain rather than hanging.
        self.force_flush(timeout=max(0.0, deadline - time.monotonic()))
        if self.dropped or self.export_failures or self.callback_failures:
            try:  # noqa: SIM105 (keep the guard explicit and import-free at atexit)
                logger.warning(
                    "capture exporter shutting down with losses: "
                    "%d records dropped, %d export failures, %d callback failures",
                    self.dropped,
                    self.export_failures,
                    self.callback_failures,
                )
            except Exception:
                # shutdown may run under atexit with logging handlers
                # partially torn down; never let the summary raise.
                pass
