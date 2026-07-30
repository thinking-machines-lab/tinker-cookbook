"""Tests for the capture exporter pipeline."""

import json
import multiprocessing
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from tinker_cookbook.capture.exporter import CaptureExporter, CaptureRecord, JsonlFileSink
from tinker_cookbook.stores.storage import LocalStorage


class CollectingSink:
    def __init__(self, block: bool = False) -> None:
        self.batches: list[list[CaptureRecord]] = []
        self.lock = threading.Lock()
        self.block_event = threading.Event()
        self.block = block

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        if self.block:
            self.block_event.wait(timeout=5.0)
        with self.lock:
            self.batches.append(list(records))

    @property
    def records(self) -> list[CaptureRecord]:
        with self.lock:
            return [r for batch in self.batches for r in batch]


class FailingSink:
    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        raise RuntimeError("boom")


def _wait_for(predicate, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_flush_on_batch_size() -> None:
    sink = CollectingSink()
    exporter = CaptureExporter(sink, max_batch_size=5, flush_interval_sec=60.0)
    try:
        for i in range(5):
            exporter.enqueue({"kind": "t", "i": i})
        _wait_for(lambda: len(sink.records) == 5)
    finally:
        exporter.shutdown()


def test_flush_on_timer() -> None:
    sink = CollectingSink()
    exporter = CaptureExporter(sink, max_batch_size=1000, flush_interval_sec=0.05)
    try:
        exporter.enqueue({"kind": "t"})
        _wait_for(lambda: len(sink.records) == 1)
    finally:
        exporter.shutdown()


def test_drop_newest_when_full() -> None:
    sink = CollectingSink(block=True)
    exporter = CaptureExporter(sink, max_queue_size=3, max_batch_size=1000, flush_interval_sec=60.0)
    try:
        for i in range(10):
            exporter.enqueue({"kind": "t", "i": i})
        # The queue caps at 3; whatever the flusher grabbed before blocking is
        # in flight, everything else was dropped (newest).
        assert exporter.dropped >= 1
        sink.block_event.set()
        assert exporter.force_flush(timeout=5.0) is True
        # Conservation: every record was either exported or counted dropped.
        assert len(sink.records) + exporter.dropped == 10
        # Oldest records survive: the dropped ones are the highest indices.
        kept = sorted(r["i"] for r in sink.records)
        assert kept == list(range(len(kept)))
    finally:
        sink.block_event.set()
        exporter.shutdown()


def test_force_flush_drains() -> None:
    sink = CollectingSink()
    exporter = CaptureExporter(sink, max_batch_size=1000, flush_interval_sec=60.0)
    try:
        for i in range(7):
            exporter.enqueue({"kind": "t", "i": i})
        assert exporter.force_flush(timeout=5.0) is True
        # True means everything enqueued before the call reached the sink.
        assert sorted(r["i"] for r in sink.records) == list(range(7))
    finally:
        exporter.shutdown()


def test_force_flush_waits_for_in_flight_batch() -> None:
    sink = CollectingSink(block=True)
    exporter = CaptureExporter(sink, max_batch_size=1000, flush_interval_sec=0.05)
    try:
        exporter.enqueue({"kind": "t", "i": 0})
        # Wait until the flusher holds the record in flight (blocked in export).
        _wait_for(lambda: exporter._in_flight)
        # In-flight batch not done: force_flush must not claim success.
        assert exporter.force_flush(timeout=0.2) is False
        assert len(sink.records) == 0
        sink.block_event.set()
        assert exporter.force_flush(timeout=5.0) is True
        assert [r["i"] for r in sink.records] == [0]  # exactly once, no loss
    finally:
        sink.block_event.set()
        exporter.shutdown()


def test_pending_tracking() -> None:
    sink = CollectingSink()
    exporter = CaptureExporter(sink, flush_interval_sec=0.05)
    try:
        assert exporter.wait_pending(timeout=0.05) is True  # nothing pending
        exporter.track_pending()
        assert exporter.wait_pending(timeout=0.1) is False
        threading.Timer(0.2, exporter.pending_done).start()
        assert exporter.wait_pending(timeout=5.0) is True
    finally:
        exporter.shutdown()


def test_shutdown_idempotent_and_flushes() -> None:
    sink = CollectingSink()
    exporter = CaptureExporter(sink, max_batch_size=1000, flush_interval_sec=60.0)
    exporter.enqueue({"kind": "t"})
    exporter.shutdown()
    exporter.shutdown()  # no-op
    assert len(sink.records) == 1
    # Enqueue after shutdown drops.
    before = exporter.dropped
    exporter.enqueue({"kind": "t"})
    assert exporter.dropped == before + 1


def test_sink_failure_counted_not_raised() -> None:
    exporter = CaptureExporter(FailingSink(), max_batch_size=1, flush_interval_sec=0.05)
    try:
        exporter.enqueue({"kind": "t"})
        _wait_for(lambda: exporter.export_failures >= 1)
    finally:
        exporter.shutdown()


def test_jsonl_file_sink_layout(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    sink = JsonlFileSink(storage, prefix="runs/r1")
    sink.export(
        [{"kind": "sample", "i": 0}, {"kind": "train_op", "i": 1}, {"kind": "sample", "i": 2}]
    )
    sample_lines = (tmp_path / "runs/r1/capture/sample.jsonl").read_text().splitlines()
    assert [json.loads(line)["i"] for line in sample_lines] == [0, 2]
    train_lines = (tmp_path / "runs/r1/capture/train_op.jsonl").read_text().splitlines()
    assert [json.loads(line)["i"] for line in train_lines] == [1]


def test_batches_wait_for_threshold_or_timer() -> None:
    """Sub-threshold records are not exported one sink call per record."""
    sink = CollectingSink()
    exporter = CaptureExporter(sink, max_batch_size=10, flush_interval_sec=10.0)
    try:
        for i in range(3):
            exporter.enqueue({"kind": "t", "i": i})
        time.sleep(0.3)  # well past any per-enqueue wakeup
        assert sink.batches == []  # waiting for threshold or timer
        for i in range(3, 10):
            exporter.enqueue({"kind": "t", "i": i})
        _wait_for(lambda: len(sink.records) == 10)
        assert len(sink.batches) == 1  # one batch, not ten sink calls
    finally:
        exporter.shutdown()


def test_force_flush_false_when_sink_raises() -> None:
    exporter = CaptureExporter(FailingSink(), max_batch_size=1000, flush_interval_sec=60.0)
    try:
        exporter.enqueue({"kind": "t"})
        assert exporter.force_flush(timeout=5.0) is False
        assert exporter.export_failures >= 1
    finally:
        exporter.shutdown()


class BlockingThenFailingSink:
    """Blocks in export until released, then raises."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        self.release.wait(timeout=10.0)
        raise RuntimeError("sink down")


def test_force_flush_false_when_in_flight_export_failed() -> None:
    sink = BlockingThenFailingSink()
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    try:
        exporter.enqueue({"kind": "t"})
        _wait_for(lambda: exporter._in_flight)  # flusher holds the record

        result: list[bool] = []
        t = threading.Thread(target=lambda: result.append(exporter.force_flush(timeout=5.0)))
        t.start()
        sink.release.set()  # in-flight export now completes by RAISING
        t.join(timeout=10.0)
        assert result == [False]  # pre-call record was lost, must not report True
        assert exporter.export_failures >= 1
    finally:
        exporter.shutdown()


def test_force_flush_bounded_when_worker_stuck_in_sink() -> None:
    """force_flush must respect its timeout even with the export lock held."""
    sink = CollectingSink(block=True)
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    try:
        exporter.enqueue({"kind": "t", "i": 0})
        _wait_for(lambda: exporter._in_flight)  # worker stuck in blocked sink
        exporter.enqueue({"kind": "t", "i": 1})  # more work queued behind it

        start = time.monotonic()
        assert exporter.force_flush(timeout=0.3) is False
        assert time.monotonic() - start < 2.0  # bounded, no indefinite lock wait
        # The drained-but-unexported record was requeued, not lost.
        sink.block_event.set()
        assert exporter.force_flush(timeout=5.0) is True
        assert sorted(r["i"] for r in sink.records) == [0, 1]
    finally:
        sink.block_event.set()
        exporter.shutdown()


class TimeoutIgnoringSink:
    """Simulates a misbehaving sink that ignores its (soft) timeout argument."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.exported: list[CaptureRecord] = []

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        time.sleep(self.delay)  # ignores `timeout` entirely
        self.exported.extend(records)


def test_force_flush_bounded_against_timeout_ignoring_sink() -> None:
    sink = TimeoutIgnoringSink(delay=2.0)
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    try:
        exporter.enqueue({"kind": "t"})
        start = time.monotonic()
        assert exporter.force_flush(timeout=0.2) is False
        assert time.monotonic() - start < 1.0  # bounded despite the 2s sink
        # The export still completes in the background on the flusher thread.
        _wait_for(lambda: len(sink.exported) == 1, timeout=10.0)
    finally:
        exporter.shutdown()


def test_shutdown_single_total_budget_with_stuck_sink() -> None:
    """shutdown(t) must not block for ~2t when the sink is stuck."""
    sink = CollectingSink(block=True)
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    try:
        exporter.enqueue({"kind": "t"})
        _wait_for(lambda: exporter._in_flight)  # worker stuck in the sink
        start = time.monotonic()
        exporter.shutdown(timeout=0.5)
        elapsed = time.monotonic() - start
        assert elapsed < 0.95  # join + flush share ONE 0.5s budget
    finally:
        sink.block_event.set()


def test_enqueue_racing_shutdown_never_strands_records() -> None:
    """Every record enqueued concurrently with shutdown is either exported
    or counted dropped; none may silently strand in a drained queue."""
    for _ in range(20):
        sink = CollectingSink()
        exporter = CaptureExporter(sink, flush_interval_sec=60.0)
        n_threads = 8
        start = threading.Barrier(n_threads + 1)

        def enqueue_one(
            exporter: CaptureExporter = exporter, start: threading.Barrier = start
        ) -> None:
            start.wait()
            exporter.enqueue({"kind": "race"})

        threads = [threading.Thread(target=enqueue_one) for _ in range(n_threads)]
        for t in threads:
            t.start()
        start.wait()
        exporter.shutdown()
        for t in threads:
            t.join()
        # Late enqueues (after shutdown observed) are counted dropped; the
        # rest must have been flushed by shutdown's final drain.
        exported = sum(len(batch) for batch in sink.batches)
        assert exported + exporter.dropped == n_threads


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="platform without fork"
)
# Python 3.13 warns about fork() in multi-threaded processes; surviving
# exactly that is what the exporter's at-fork hook is for.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_exporter_reinitializes_after_fork() -> None:
    """A forked child gets a live flusher (rebuilt via os.register_at_fork)
    and discards inherited queued records WITHOUT counting them as dropped:
    the parent still owns and exports them, so child re-export would
    duplicate, and counting them would report phantom losses. The child's
    loss counters start at zero."""
    ctx = multiprocessing.get_context("fork")
    sink = CollectingSink()
    exporter = CaptureExporter(sink, flush_interval_sec=60.0)  # timer won't fire
    exporter.export_failures = 2  # parent-lifetime state, must not leak to child
    exporter.callback_failures = 1
    for i in range(3):
        exporter.enqueue({"kind": "parent", "i": i})  # sits in the queue at fork

    def child() -> None:
        # Inherited records are discarded but NOT counted; all counters reset.
        ok = exporter.dropped == 0
        ok = ok and exporter.export_failures == 0
        ok = ok and exporter.callback_failures == 0
        ok = ok and exporter._flusher.is_alive()  # reborn flusher
        exporter.enqueue({"kind": "child"})
        ok = ok and exporter.force_flush(timeout=10.0)
        child_kinds = [r["kind"] for r in sink.records]
        ok = ok and child_kinds == ["child"]  # none of the parent's records
        os._exit(0 if ok else 17)

    process = ctx.Process(target=child)
    process.start()
    process.join(timeout=30.0)
    assert process.exitcode == 0

    # The parent is unaffected and still owns the 3 inherited records.
    assert exporter.force_flush(timeout=10.0)
    assert [r["kind"] for r in sink.records] == ["parent", "parent", "parent"]
    assert exporter.dropped == 0
    assert exporter.export_failures == 2
    assert exporter.callback_failures == 1
    exporter.export_failures = 0  # undo the seeded values before shutdown
    exporter.callback_failures = 0
    exporter.shutdown()


def test_shutdown_warns_on_nonzero_loss_counters(caplog: pytest.LogCaptureFixture) -> None:
    """Shutdown emits exactly one warning summarizing losses when any counter
    is nonzero, and stays silent on a clean run."""
    import logging

    sink = CollectingSink()
    exporter = CaptureExporter(sink)
    exporter.dropped = 5
    exporter.export_failures = 2
    exporter.callback_failures = 1
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.capture.exporter"):
        exporter.shutdown()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "5 records dropped" in msg
    assert "2 export failures" in msg
    assert "1 callback failures" in msg

    caplog.clear()
    clean = CaptureExporter(CollectingSink())
    clean.enqueue({"kind": "a"})
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.capture.exporter"):
        clean.shutdown()
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_sink_failure_logging_rate_limited(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """First failure of an outage logs a full traceback; then at most one
    summary line per report interval; success resets so the next outage
    gets a traceback again."""
    import logging

    from tinker_cookbook.capture import exporter as exporter_mod

    monkeypatch.setattr(exporter_mod, "_FAILURE_REPORT_INTERVAL_SEC", 0.05)

    class ToggleSink:
        fail = True

        def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
            if self.fail:
                raise ConnectionError("store down")

    sink = ToggleSink()
    exporter = CaptureExporter(sink, flush_interval_sec=60.0)
    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.capture.exporter"):
        assert not exporter._export([{"k": 1}])  # first: full traceback
        assert not exporter._export([{"k": 2}])  # suppressed
        assert not exporter._export([{"k": 3}])  # suppressed
        first = [r for r in caplog.records if "export failed" in r.getMessage()]
        assert len(first) == 1
        assert first[0].exc_info is not None  # traceback on the first only
        assert not [r for r in caplog.records if "still failing" in r.getMessage()]

        time.sleep(0.06)
        assert not exporter._export([{"k": 4}])  # interval elapsed: summary
        summaries = [r for r in caplog.records if "still failing" in r.getMessage()]
        assert len(summaries) == 1
        message = summaries[0].getMessage()
        assert "3 batches" in message  # the two suppressed + this one
        assert "ConnectionError" in message

        sink.fail = False
        assert exporter._export([{"k": 5}])  # success resets the outage
        sink.fail = True
        assert not exporter._export([{"k": 6}])  # new outage: traceback again
        first = [r for r in caplog.records if "export failed" in r.getMessage()]
        assert len(first) == 2
    exporter.shutdown()
