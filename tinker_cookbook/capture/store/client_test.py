"""Tests for StoreSink, the daemon spawn helper, and capture_to_store."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from tinker_cookbook.capture.exporter import CaptureExporter
from tinker_cookbook.capture.store.client import (
    StoreSink,
    _get_json,
    _ordered_ingest_batches,
    capture_to_store,
    ensure_daemon,
    wire_rows_from_sample_record,
)
from tinker_cookbook.capture.store.daemon import DAEMON_INFO_FILENAME, acquire_lock


def _sample_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "sample",
        "scope": {"run_id": "run-1", "run_attempt": 0, "iteration": 3, "purpose": "rollout"},
        "sampling_session_id": "sess-1",
        "created_at": 123.0,
        "num_samples": 2,
        "model_path": "tinker://model/v7",
        "prompt_tokens": [7, 8, 9],
        "samples": [
            {"tokens": [1, 2], "logprobs": [-0.1, -0.2], "stop_reason": "length"},
            {"tokens": [3], "logprobs": [-0.3], "stop_reason": "stop"},
        ],
    }
    record.update(overrides)
    return record


def test_wire_rows_from_sample_record() -> None:
    rows = wire_rows_from_sample_record(_sample_record(seq_id=777))
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["iteration"] == 3
    assert rows[0]["sample_idx"] == 0
    assert rows[1]["sample_idx"] == 1
    assert rows[0]["sampled_tokens"] == [1, 2]
    assert rows[1]["logprobs"] == [-0.3]
    assert rows[0]["policy_version"] == "tinker://model/v7"
    assert rows[0]["seq_id"] == 777
    assert rows[0]["prompt_tokens"] == [7, 8, 9]
    assert rows[1]["prompt_tokens"] == [7, 8, 9]
    assert "prompt_tokens" not in rows[0]["metadata"]
    assert rows[0]["metadata"]["stop_reason"] == "length"
    assert rows[1]["metadata"]["num_samples"] == 2
    # No non-reserved scope pairs in this record: no metadata.scope key.
    assert "scope" not in rows[0]["metadata"]


def test_wire_rows_preserve_non_reserved_scope_pairs() -> None:
    """Regression: rows from a run with capture(phase=..., worker=...) lost
    those pairs entirely. Non-reserved scope keys must persist under
    metadata.scope (a dedicated key, so they cannot clobber request
    metadata)."""
    record = _sample_record(
        scope={
            "run_id": "run-1",
            "iteration": 3,
            "phase": "thread-pool",
            "worker": 3,
            "num_samples": 99,  # would clobber request metadata if merged flat
        }
    )
    rows = wire_rows_from_sample_record(record)
    for row in rows:
        assert row["metadata"]["scope"] == {
            "phase": "thread-pool",
            "worker": 3,
            "num_samples": 99,
        }
        assert row["metadata"]["num_samples"] == 2  # request metadata untouched
        assert row["iteration"] == 3  # reserved keys still in their columns


def test_store_down_drops_with_counter_never_raises() -> None:
    sink = StoreSink("http://127.0.0.1:1")  # nothing listens here
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.05)
    try:
        exporter.enqueue(_sample_record())  # must not raise into the caller
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and exporter.export_failures == 0:
            time.sleep(0.01)
        assert exporter.export_failures >= 1
    finally:
        exporter.shutdown()


def _daemon_pid(data_dir: Path) -> int:
    return int(json.loads((data_dir / DAEMON_INFO_FILENAME).read_text())["pid"])


def _kill_daemon(data_dir: Path) -> None:
    with contextlib.suppress(OSError, ValueError):
        os.kill(_daemon_pid(data_dir), signal.SIGKILL)


@pytest.mark.timeout(120)
def test_ensure_daemon_spawn_reuse_and_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "capture-store"
    try:
        base_url = ensure_daemon(data_dir, idle_shutdown_minutes=5.0)
        assert _get_json(f"{base_url}/healthz", timeout=5.0)["status"] == "ok"

        # Second call reuses the healthy daemon (same URL and pid).
        pid = _daemon_pid(data_dir)
        assert ensure_daemon(data_dir) == base_url
        assert _daemon_pid(data_dir) == pid

        # End-to-end: sink -> daemon -> query, with a mixed batch whose
        # arrival order must be preserved on the shared cursor.
        StoreSink(base_url).export(
            [
                {"kind": "note", "event_id": "n1", "scope": {"run_id": "run-1"}},
                _sample_record(scope={"run_id": "run-1", "run_attempt": 0}),
                {"kind": "note", "event_id": "n2", "scope": {"run_id": "run-1"}},
            ]
        )
        runs = _get_json(f"{base_url}/runs", timeout=5.0)["runs"]
        assert runs[0]["run_id"] == "run-1"
        assert runs[0]["num_wire_rows"] == 2
        assert runs[0]["num_annotations"] == 2

        # Replaying a fully keyed record (instrumented records carry a
        # call-time seq_id) dedupes instead of inflating the run.
        keyed = _sample_record(scope={"run_id": "run-1", "run_attempt": 0}, seq_id=4242)
        StoreSink(base_url).export([keyed])
        StoreSink(base_url).export([keyed])
        runs = _get_json(f"{base_url}/runs", timeout=5.0)["runs"]
        assert runs[0]["num_wire_rows"] == 4  # 2 original + 2 keyed, replay deduped
        from tinker_cookbook.capture.store.db import CaptureDB

        db = CaptureDB(data_dir / "capture.sqlite")
        try:
            kinds = [e.event_type for e in db.stream_events("run-1")]
        finally:
            db.close()
        # First four events are the mixed batch in arrival order; the keyed
        # replay rows land after.
        assert kinds[:4] == ["annotation", "wire_row", "wire_row", "annotation"]
        assert kinds[4:] == ["wire_row", "wire_row"]
    finally:
        _kill_daemon(data_dir)


@pytest.mark.timeout(120)
def test_daemon_idle_shutdown(tmp_path: Path) -> None:
    data_dir = tmp_path / "capture-store"
    base_url = ensure_daemon(data_dir, idle_shutdown_minutes=0.02)  # 1.2s idle budget
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                _get_json(f"{base_url}/healthz", timeout=2.0)
            except OSError:
                break  # stopped serving
            time.sleep(0.3)
        else:
            raise AssertionError(f"daemon at {base_url} did not idle-shutdown")
    finally:
        _kill_daemon(data_dir)


@pytest.mark.timeout(120)
def test_capture_to_store_wires_everything(tmp_path: Path) -> None:
    import tinker

    data_dir = tmp_path / "capture-store"
    original_sample = tinker.SamplingClient.sample
    try:
        with capture_to_store(
            "run-cts", data_dir=data_dir, run_attempt=1, purpose="rollout"
        ) as session:
            # SDK is instrumented inside the context.
            assert tinker.SamplingClient.sample is not original_sample
            # Simulate a capture record arriving through the exporter.
            session.exporter.enqueue(_sample_record(scope={"run_id": "run-cts", "run_attempt": 1}))
            session.exporter.force_flush()
            runs = _get_json(f"{session.base_url}/runs", timeout=5.0)["runs"]
            assert runs[0]["run_id"] == "run-cts"
            assert runs[0]["latest_attempt"] == 1
            base_url = session.base_url
        # Uninstrumented and shut down on exit.
        assert tinker.SamplingClient.sample is original_sample
        assert _get_json(f"{base_url}/healthz", timeout=5.0)["status"] == "ok"
    finally:
        _kill_daemon(data_dir)


@pytest.mark.timeout(120)
def test_ensure_daemon_retries_spawn_after_lock_race(tmp_path: Path) -> None:
    """A child that loses the flock race exits; ensure_daemon must respawn."""
    import threading

    data_dir = tmp_path / "capture-store"
    data_dir.mkdir(parents=True)
    # Simulate a dying-but-not-released daemon: hold the flock ourselves so
    # the first spawned child exits immediately, then release after a delay.
    lock_fd = acquire_lock(data_dir)
    assert lock_fd is not None
    threading.Timer(2.0, os.close, args=(lock_fd,)).start()
    try:
        base_url = ensure_daemon(data_dir, idle_shutdown_minutes=5.0, spawn_timeout=60.0)
        assert _get_json(f"{base_url}/healthz", timeout=5.0)["status"] == "ok"
        # The daemon serving is a respawn, not the first (lock-losing) child.
        assert (data_dir / "daemon.log").read_text().count("another capture store daemon") >= 1
    finally:
        _kill_daemon(data_dir)


@pytest.mark.timeout(120)
def test_capture_to_store_drains_outstanding_futures(tmp_path: Path) -> None:
    """Records completing after scope exit (but before teardown drain) are kept."""
    import threading

    data_dir = tmp_path / "capture-store"
    base_url = None
    try:
        with capture_to_store("run-drain", data_dir=data_dir, drain_timeout_sec=10.0) as session:
            base_url = session.base_url
            exporter = session.exporter
            # Simulate an instrumented call whose future is still outstanding
            # when the context exits: pending is tracked now, the record is
            # enqueued (by the done-callback holding this exporter) later.
            exporter.track_pending()

            def complete_later() -> None:
                time.sleep(1.0)  # context has exited by now
                exporter.enqueue(_sample_record(scope={"run_id": "run-drain", "run_attempt": 0}))
                exporter.pending_done()

            threading.Thread(target=complete_later).start()
        # Teardown waited for pending_done and flushed before shutdown.
        runs = _get_json(f"{base_url}/runs", timeout=5.0)["runs"]
        assert runs and runs[0]["run_id"] == "run-drain"
        assert runs[0]["num_wire_rows"] == 2
    finally:
        _kill_daemon(data_dir)


def test_ordered_ingest_batches_preserves_mixed_order() -> None:
    ann1 = {"kind": "note", "event_id": "a1", "scope": {"run_id": "r"}}
    ann2 = {"kind": "note", "event_id": "a2", "scope": {"run_id": "r"}}
    batches = _ordered_ingest_batches([ann1, _sample_record(), _sample_record(), ann2])
    assert [endpoint for endpoint, _ in batches] == [
        "/ingest/annotations",
        "/ingest/wire",
        "/ingest/annotations",
    ]
    # Contiguous same-kind records coalesce into one call.
    assert len(batches[1][1]["rows"]) == 4  # 2 sample records x 2 samples each
    assert [a["event_id"] for a in batches[0][1]["annotations"]] == ["a1"]
    assert [a["event_id"] for a in batches[2][1]["annotations"]] == ["a2"]


@pytest.mark.timeout(120)
def test_ensure_daemon_rejects_wrong_identity_and_respawns(tmp_path: Path) -> None:
    """A stale daemon.json pointing at a DIFFERENT daemon's port is not reused."""
    dir_a = tmp_path / "store-a"
    dir_b = tmp_path / "store-b"
    try:
        url_a = ensure_daemon(dir_a, idle_shutdown_minutes=5.0)
        # Simulate port reuse: dir_b's stale daemon.json points at daemon A.
        dir_b.mkdir(parents=True)
        stale = json.loads((dir_a / DAEMON_INFO_FILENAME).read_text())
        (dir_b / DAEMON_INFO_FILENAME).write_text(json.dumps(stale))

        url_b = ensure_daemon(dir_b, idle_shutdown_minutes=5.0)
        assert url_b != url_a  # did not adopt the unrelated daemon
        health_b = _get_json(f"{url_b}/healthz", timeout=5.0)
        assert health_b["data_dir"] == str(dir_b.resolve())
        # And rows go to B's store, not A's.
        StoreSink(url_b).export([_sample_record(scope={"run_id": "run-b"})])
        assert _get_json(f"{url_b}/runs", timeout=5.0)["runs"][0]["run_id"] == "run-b"
        assert _get_json(f"{url_a}/runs", timeout=5.0)["runs"] == []
    finally:
        _kill_daemon(dir_a)
        _kill_daemon(dir_b)


def test_no_fabricated_wire_rows_for_failed_or_empty_samples() -> None:
    # Failed call: no samples field at all.
    error_record = _sample_record(status="error", error="RuntimeError('x')")
    del error_record["samples"]
    assert wire_rows_from_sample_record(error_record) == []
    # Successful call with an empty sequence list: zero rows, nothing invented.
    empty_record = _sample_record(samples=[])
    assert wire_rows_from_sample_record(empty_record) == []
    # Both are preserved as annotations instead of dropped.
    batches = _ordered_ingest_batches([error_record, empty_record])
    assert [endpoint for endpoint, _ in batches] == ["/ingest/annotations"]
    payloads = batches[0][1]["annotations"]
    assert payloads[0]["payload"]["status"] == "error"
    assert payloads[0]["kind"] == "sample"


def test_store_sink_splits_batch_on_413(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from tinker_cookbook.capture.store import client as client_mod

    calls: list[int] = []

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        (items_key,) = payload.keys()
        n = len(payload[items_key])
        if n > 2:
            raise urllib.error.HTTPError(url, 413, "Payload Too Large", None, None)  # type: ignore[arg-type]
        calls.append(n)
        return {"inserted": n, "deduped": 0}

    monkeypatch.setattr(client_mod, "_post_json", fake_post)
    sink = StoreSink("http://127.0.0.1:1")
    # 3 sample records x 2 rows = 6 wire rows; server "accepts" at most 2.
    sink.export([_sample_record(), _sample_record(), _sample_record()])
    assert sum(calls) == 6
    assert all(n <= 2 for n in calls)


def test_store_sink_whole_batch_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from tinker_cookbook.capture.store import client as client_mod

    timeouts: list[float] = []

    def slow_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        timeouts.append(timeout)
        time.sleep(0.2)
        return {"inserted": 0, "deduped": 0}

    monkeypatch.setattr(client_mod, "_post_json", slow_post)
    sink = StoreSink("http://127.0.0.1:1")
    ann = {"kind": "note", "event_id": "a", "scope": {"run_id": "r"}}
    # Alternating batch -> 4 contiguous runs, but only ~0.5s total budget:
    # the shared deadline must cut it off instead of 4 x 0.5s.
    with pytest.raises(TimeoutError):
        sink.export([ann, _sample_record(), ann | {"event_id": "b"}, _sample_record()], timeout=0.5)
    assert len(timeouts) <= 3
    assert all(t <= 0.5 + 1e-6 for t in timeouts)
    # Remaining time shrinks across runs.
    assert timeouts == sorted(timeouts, reverse=True)


def test_store_sink_splits_by_size_before_posting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized payloads are chunked by encoded size BEFORE the first POST."""
    from tinker_cookbook.capture.store import client as client_mod

    sizes: list[int] = []

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        sizes.append(client_mod._encoded_size(payload))
        return {"inserted": 0, "deduped": 0}

    monkeypatch.setattr(client_mod, "_post_json", fake_post)
    monkeypatch.setattr(client_mod, "_MAX_POST_BYTES", 50_000)
    big_tokens = list(range(3_000))  # ~17KB encoded per sample record row
    records = [_sample_record(samples=[{"tokens": big_tokens, "logprobs": None}]) for _ in range(8)]
    StoreSink("http://127.0.0.1:1").export(records)
    # Everything delivered, every POST under the soft cap, more than one chunk.
    assert len(sizes) >= 2
    assert all(size <= 50_000 for size in sizes)


def test_capture_to_store_nested_restores_outer_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exiting an inner capture_to_store must restore the OUTER session's
    exporter and keep the SDK wrappers in place; only exiting the outermost
    context uninstruments."""
    import tinker

    from tinker_cookbook.capture import instrument as instrument_mod
    from tinker_cookbook.capture.store import client as client_mod

    monkeypatch.setattr(
        client_mod, "ensure_daemon", lambda data_dir, **kwargs: "http://127.0.0.1:9"
    )
    assert instrument_mod.current_exporter() is None
    with capture_to_store("outer", data_dir="/unused", drain_timeout_sec=1.0) as outer:
        assert instrument_mod.current_exporter() is outer.exporter
        with capture_to_store("inner", data_dir="/unused", drain_timeout_sec=1.0) as inner:
            assert instrument_mod.current_exporter() is inner.exporter
        assert instrument_mod.current_exporter() is outer.exporter
        assert getattr(tinker.SamplingClient.sample, "_capture_original", None) is not None
    assert instrument_mod.current_exporter() is None
    assert getattr(tinker.SamplingClient.sample, "_capture_original", None) is None


def test_annotation_event_id_stable_across_retries() -> None:
    """A generated event_id is stamped back into the source record, so
    re-exporting the same record (retry after an ambiguous failure) reuses
    the id and dedupes in the store instead of duplicating."""
    from tinker_cookbook.capture.store.client import _annotation_from_record

    record: dict[str, Any] = {"kind": "note", "scope": {"run_id": "r"}}
    first = _annotation_from_record(record)
    second = _annotation_from_record(record)
    assert first["event_id"] == second["event_id"] == record["event_id"]


def test_ensure_daemon_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """data_dir="~/..." must resolve to the user's home, not $PWD/~ ."""
    from tinker_cookbook.capture.store import client as client_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    seen: list[Path] = []

    def fake_healthy(data_dir: Path, timeout: float = 1.0) -> str:
        seen.append(data_dir)
        return "http://127.0.0.1:1"

    monkeypatch.setattr(client_mod, "_healthy_base_url", fake_healthy)
    monkeypatch.setattr(client_mod, "_claim_daemon", lambda base_url, timeout=1.0: True)
    assert ensure_daemon("~/capstore") == "http://127.0.0.1:1"
    assert seen == [tmp_path / "capstore"]
    assert (tmp_path / "capstore").is_dir()


def test_ensure_daemon_claims_before_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reusing a daemon must reset its idle lease via /touch; a failed claim
    treats the daemon as gone instead of handing out a dying URL."""
    from tinker_cookbook.capture.store import client as client_mod

    monkeypatch.setattr(
        client_mod, "_healthy_base_url", lambda data_dir, timeout=1.0: "http://127.0.0.1:1"
    )
    touched: list[str] = []

    def fake_claim(base_url: str, timeout: float = 1.0) -> bool:
        touched.append(base_url)
        return True

    monkeypatch.setattr(client_mod, "_claim_daemon", fake_claim)
    assert ensure_daemon(tmp_path) == "http://127.0.0.1:1"
    assert touched == ["http://127.0.0.1:1"]

    # Failed claim falls through to the spawn loop and times out here
    # (nothing real to spawn against a stubbed healthy URL that won't claim).
    monkeypatch.setattr(client_mod, "_claim_daemon", lambda base_url, timeout=1.0: False)
    monkeypatch.setattr(client_mod, "_healthy_base_url", lambda data_dir, timeout=1.0: None)
    with pytest.raises(TimeoutError):
        ensure_daemon(tmp_path, spawn_timeout=0.3)


def test_scope_extras_persist_end_to_end(tmp_path: Path) -> None:
    """capture(phase=..., worker=...) around an instrumented sample must land
    in the stored row's metadata.scope and come back via /runs/{id}/rows AND
    /stream."""
    import urllib.request
    from concurrent.futures import Future

    from tinker_cookbook.capture.instrument import _make_sample_wrapper
    from tinker_cookbook.capture.scope import capture

    class _Prompt:
        def to_ints(self) -> list[int]:
            return [1, 2]

    class _Seq:
        tokens = [3, 4]
        logprobs = [-0.1, -0.2]
        stop_reason = "stop"

    class _Resp:
        sequences = [_Seq()]

    class _Client:
        _sampling_session_id = "sess-extras"

        def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Future[Any]:
            future: Future[Any] = Future()
            future.set_result(_Resp())
            return future

    data_dir = tmp_path / "scope-store"
    try:
        with capture_to_store("run-extras", data_dir=data_dir) as session:
            wrapped = _make_sample_wrapper(_Client.sample).__get__(_Client())
            with capture(phase="thread-pool", worker=3):
                wrapped(_Prompt(), 1, None)
            session.exporter.wait_pending(timeout=10.0)
            assert session.exporter.force_flush(timeout=10.0)
            rows = _get_json(f"{session.base_url}/runs/run-extras/rows", timeout=5.0)["rows"]
            assert len(rows) == 1
            assert rows[0]["metadata"]["scope"] == {"phase": "thread-pool", "worker": 3}
            assert rows[0]["sampled_tokens"] == [3, 4]
            # And over /stream (SSE): the first data line carries the row.
            with urllib.request.urlopen(
                f"{session.base_url}/stream?run_id=run-extras", timeout=5.0
            ) as stream:
                payload = None
                for _ in range(20):
                    line = stream.readline().decode()
                    if line.startswith("data: "):
                        payload = json.loads(line.removeprefix("data: "))
                        break
                assert payload is not None
                assert payload["metadata"]["scope"] == {"phase": "thread-pool", "worker": 3}
    finally:
        _kill_daemon(data_dir)


def test_capture_to_store_propagates_scope_to_thread_pools(tmp_path: Path) -> None:
    """No manual propagate.instrument_threads() call: rows produced from
    thread-pool code inside capture_to_store carry the ambient scope."""
    import threading
    from concurrent.futures import Future, ThreadPoolExecutor

    from tinker_cookbook.capture.instrument import _make_sample_wrapper
    from tinker_cookbook.capture.scope import capture

    class _Prompt:
        def to_ints(self) -> list[int]:
            return [1]

    class _Seq:
        tokens = [2]
        logprobs = [-0.1]
        stop_reason = "stop"

    class _Resp:
        sequences = [_Seq()]

    class _Client:
        _sampling_session_id = "sess-threads"

        def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Future[Any]:
            future: Future[Any] = Future()
            future.set_result(_Resp())
            return future

    original_start = threading.Thread.start
    data_dir = tmp_path / "threads-store"
    try:
        with capture_to_store("run-threads", data_dir=data_dir) as session:
            assert threading.Thread.start is not original_start  # patched by default
            wrapped = _make_sample_wrapper(_Client.sample).__get__(_Client())

            def sample_from_worker() -> None:
                wrapped(_Prompt(), 1, None)

            with capture(phase="thread-pool", worker=3), ThreadPoolExecutor(1) as pool:
                pool.submit(sample_from_worker).result()
            session.exporter.wait_pending(timeout=10.0)
            assert session.exporter.force_flush(timeout=10.0)
            rows = _get_json(f"{session.base_url}/runs/run-threads/rows", timeout=5.0)["rows"]
            assert len(rows) == 1
            assert rows[0]["run_id"] == "run-threads"  # ambient scope reached the worker
            assert rows[0]["metadata"]["scope"] == {"phase": "thread-pool", "worker": 3}
        assert threading.Thread.start is original_start  # restored on exit
    finally:
        _kill_daemon(data_dir)


def test_capture_to_store_thread_instrumentation_nesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inner exit must not unpatch threading while an outer session (or an
    independent caller) still relies on it; instrument_threads=False leaves
    threading untouched."""
    import threading

    from tinker_cookbook.capture import propagate
    from tinker_cookbook.capture.store import client as client_mod

    monkeypatch.setattr(
        client_mod, "ensure_daemon", lambda data_dir, **kwargs: "http://127.0.0.1:9"
    )
    original_start = threading.Thread.start

    # instrument_threads=False: threading stays unpatched throughout.
    with capture_to_store(
        "run-off", data_dir="/unused", drain_timeout_sec=1.0, instrument_threads=False
    ):
        assert threading.Thread.start is original_start
    assert threading.Thread.start is original_start

    # Nested: inner exit keeps the outer session's patches; outer exit restores.
    with capture_to_store("outer", data_dir="/unused", drain_timeout_sec=1.0):
        assert threading.Thread.start is not original_start
        with capture_to_store("inner", data_dir="/unused", drain_timeout_sec=1.0):
            assert threading.Thread.start is not original_start
        assert threading.Thread.start is not original_start  # outer still owns it
    assert threading.Thread.start is original_start

    # Independent caller turned it on first: capture_to_store must not tear
    # it down on exit.
    propagate.instrument_threads()
    try:
        with capture_to_store("run-indep", data_dir="/unused", drain_timeout_sec=1.0):
            pass
        assert threading.Thread.start is not original_start
    finally:
        propagate.uninstrument_threads()
    assert threading.Thread.start is original_start
