"""Client side of the capture store daemon.

- :class:`StoreSink`: an exporter sink that POSTs capture records to the
  daemon (stdlib ``urllib``, safe from the exporter's flusher thread).
- :func:`ensure_daemon`: health-check / spawn-if-missing, race-safe via the
  daemon's data-dir flock (both racers may spawn; the loser exits and both
  callers converge on whichever daemon holds the lock).
- :func:`capture_to_store`: one context manager wiring
  ``instrument_tinker`` + ``CaptureExporter`` + ``StoreSink`` together.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook.capture import propagate
from tinker_cookbook.capture.exporter import CaptureExporter, CaptureRecord
from tinker_cookbook.capture.instrument import (
    current_exporter,
    instrument_tinker,
    uninstrument_tinker,
)
from tinker_cookbook.capture.scope import ScopeValue, capture
from tinker_cookbook.capture.store.daemon import DAEMON_INFO_FILENAME

_RESPAWN_INTERVAL_SEC = 0.5

# Soft cap on a single ingest POST body. Kept well under the daemon's
# client_max_size (256 MiB, see store/app.py); payloads are split by encoded
# size BEFORE posting so oversized uploads are never attempted, with the 413
# handler kept as a reactive fallback for daemons running a smaller cap.
_MAX_POST_BYTES = 8 * 1024 * 1024

_SCOPE_COLUMNS = (
    "run_id",
    "run_attempt",
    "split",
    "iteration",
    "group_idx",
    "traj_idx",
    "purpose",
)


# Ingest payloads and daemon responses are free-form JSON (shapes vary by
# record kind); Any is the honest value type at this HTTP boundary.
JsonDict = dict[str, Any]


def _post_json(url: str, payload: JsonDict, timeout: float) -> JsonDict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get_json(url: str, timeout: float) -> JsonDict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wire_rows_from_sample_record(record: CaptureRecord) -> list[JsonDict]:
    """Explode one ``kind="sample"`` capture record into wire rows."""
    scope: JsonDict = record.get("scope", {})
    base = {key: scope.get(key) for key in _SCOPE_COLUMNS}
    # Non-reserved scope pairs (e.g. capture(phase="eval", worker=3)) have no
    # dedicated column but MUST persist (the capture README promises unknown
    # keys are preserved). They ride the metadata JSON under a dedicated
    # "scope" key so they can never clobber request-metadata keys.
    extra_scope = {key: value for key, value in scope.items() if key not in _SCOPE_COLUMNS}
    base["sampling_session_id"] = record.get("sampling_session_id")
    base["seq_id"] = record.get("seq_id")
    base["policy_version"] = record.get("model_path") or record.get("policy_version")
    base["created_at"] = record.get("created_at")
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in (
            "kind",
            "scope",
            "samples",
            "sampling_session_id",
            "seq_id",
            "created_at",
            "prompt_tokens",
        )
    }
    if extra_scope:
        metadata["scope"] = extra_scope
    samples: list[JsonDict] = record.get("samples") or []
    return [
        {
            **base,
            "sample_idx": sample_idx,
            "prompt_tokens": record.get("prompt_tokens"),
            "sampled_tokens": sample.get("tokens"),
            "logprobs": sample.get("logprobs"),
            "metadata": {**metadata, "stop_reason": sample.get("stop_reason")},
        }
        for sample_idx, sample in enumerate(samples)
    ]


class StoreSink:
    """Exporter sink that POSTs batches to the capture store daemon.

    ``kind="sample"`` records become wire rows; everything else becomes an
    annotation (``event_id`` from the record, or a fresh UUID).
    """

    def __init__(self, base_url: str, *, default_timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = default_timeout

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        effective = timeout if timeout is not None else self._default_timeout
        # One deadline for the WHOLE batch, shared across the per-run POSTs
        # below; otherwise an alternating batch could block for
        # len(batches) * timeout.
        deadline = time.monotonic() + effective
        # Ingest per contiguous same-kind run, in arrival order, so the
        # store's shared cursor preserves the original record order across
        # the two tables (mixed batches would otherwise stream out of order).
        for endpoint, payload in _ordered_ingest_batches(records):
            self._post_within_deadline(endpoint, payload, deadline)

    def _post_within_deadline(self, endpoint: str, payload: JsonDict, deadline: float) -> None:
        """POST one ingest payload, splitting oversized bodies before sending.

        Payloads whose encoded size exceeds ``_MAX_POST_BYTES`` are split in
        half up front (recursively, preserving order), so an oversized upload
        never consumes the shared batch deadline just to be rejected. A 413
        from the daemon (e.g. one running a smaller cap) still triggers the
        same split reactively.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"capture store batch deadline exhausted before {endpoint}")
        (items_key,) = payload.keys()
        items = payload[items_key]
        if len(items) > 1 and _encoded_size(payload) > _MAX_POST_BYTES:
            mid = len(items) // 2
            self._post_within_deadline(endpoint, {items_key: items[:mid]}, deadline)
            self._post_within_deadline(endpoint, {items_key: items[mid:]}, deadline)
            return
        try:
            _post_json(f"{self._base_url}{endpoint}", payload, remaining)
        except urllib.error.HTTPError as e:
            if e.code != 413 or len(items) <= 1:
                raise
            mid = len(items) // 2
            self._post_within_deadline(endpoint, {items_key: items[:mid]}, deadline)
            self._post_within_deadline(endpoint, {items_key: items[mid:]}, deadline)


def _encoded_size(payload: JsonDict) -> int:
    """Encoded JSON body size in bytes, matching what _post_json sends."""
    return len(json.dumps(payload, default=str).encode("utf-8"))


def _annotation_from_record(record: CaptureRecord) -> JsonDict:
    scope: JsonDict = record.get("scope", {})
    if not record.get("event_id"):
        # Stamp the generated id back into the source record so a retry or
        # replay of the same record reuses it and dedupes in the store
        # instead of inserting a duplicate under a fresh id.
        record["event_id"] = str(uuid.uuid4())
    return {
        "event_id": record["event_id"],
        "run_id": scope.get("run_id") or "unattributed",
        "kind": record.get("kind", "unknown"),
        "payload": record,
        "created_at": record.get("created_at"),
    }


def _ordered_ingest_batches(
    records: Sequence[CaptureRecord],
) -> list[tuple[str, JsonDict]]:
    """Split a mixed batch into contiguous same-kind ingest calls, in order."""
    batches: list[tuple[str, JsonDict]] = []
    for record in records:
        rows: list[JsonDict] = []
        if record.get("kind") == "sample":
            rows = wire_rows_from_sample_record(record)
        if rows:
            if batches and batches[-1][0] == "/ingest/wire":
                batches[-1][1]["rows"].extend(rows)
            else:
                batches.append(("/ingest/wire", {"rows": rows}))
        else:
            # Non-sample records, and sample records with no sampled
            # sequences (errors, cancellations, empty responses), become
            # annotations; fabricating a sample_idx=0 wire row would make
            # /runs and /stream report a sequence that never existed.
            ann = _annotation_from_record(record)
            if batches and batches[-1][0] == "/ingest/annotations":
                batches[-1][1]["annotations"].append(ann)
            else:
                batches.append(("/ingest/annotations", {"annotations": [ann]}))
    return batches


def _healthy_base_url(data_dir: Path, timeout: float = 1.0) -> str | None:
    """Base URL of a healthy daemon for ``data_dir``, verified by identity.

    A stale ``daemon.json`` (left by an idle-shutdown daemon) can point at a
    port since reused by a DIFFERENT capture daemon; a generic 200 would then
    route rows to the wrong store. Only reuse when the live daemon reports
    the same canonical ``data_dir`` and the ``instance_token`` recorded in
    ``daemon.json``; otherwise treat the daemon as absent and remove the
    stale discovery file.
    """
    info_path = data_dir / DAEMON_INFO_FILENAME
    if not info_path.exists():
        return None
    try:
        raw = info_path.read_text()
        info = json.loads(raw)
        base_url = info["base_url"]
        health = _get_json(f"{base_url}/healthz", timeout)
        if (
            health.get("status") == "ok"
            and health.get("data_dir") == str(data_dir.resolve())
            and health.get("instance_token") is not None
            and health.get("instance_token") == info.get("instance_token")
        ):
            return base_url
        # A daemon answered but it is not ours: the discovery file is stale.
        # Only remove it if it still holds the exact content we checked, so a
        # concurrent ensure_daemon that just wrote a FRESH daemon.json does
        # not lose its discovery file.
        with contextlib.suppress(OSError):
            if info_path.read_text() == raw:
                info_path.unlink(missing_ok=True)
        return None
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return None


def _claim_daemon(base_url: str, timeout: float = 1.0) -> bool:
    """Reset a daemon's idle timer via the activity-counting ``/touch``.

    ``/healthz`` deliberately does not count as activity, so a daemon close
    to its idle deadline could pass the health probe and exit before the
    caller's first request; claiming extends its lease first. Returns False
    (treat the daemon as gone) if the touch fails.
    """
    try:
        return _get_json(f"{base_url}/touch", timeout).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_daemon(
    data_dir: str | Path,
    *,
    idle_shutdown_minutes: float | None = 60.0,
    spawn_timeout: float = 30.0,
) -> str:
    """Return the base URL of a healthy daemon for ``data_dir``, spawning if needed.

    The spawned process is detached (``start_new_session=True``) so it
    survives the parent. Racing spawners are safe: the daemon's data-dir
    flock guarantees a single owner, and this function polls until whichever
    daemon won is healthy.
    """
    # Expand ~ BEFORE creating anything: a literal Path("~/...") would
    # create and serve $PWD/~/... , fragmenting captures across daemons
    # that depend on the process working directory.
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    base_url = _healthy_base_url(data_dir)
    if base_url is not None and _claim_daemon(base_url):
        return base_url

    cmd = [
        sys.executable,
        "-m",
        "tinker_cookbook.capture.store.daemon",
        "--data-dir",
        str(data_dir),
        "--port",
        "0",
    ]
    if idle_shutdown_minutes is not None:
        cmd += ["--idle-shutdown-minutes", str(idle_shutdown_minutes)]

    def spawn() -> subprocess.Popen[bytes]:
        with open(data_dir / "daemon.log", "ab") as log_file:
            return subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    # Our child may lose the flock race against a daemon that is dying but
    # has not yet released the lock and exit without serving; in that case
    # no healthy daemon may ever appear. Watch the child and respawn it
    # (throttled) until something is healthy or the overall timeout expires.
    proc = spawn()
    deadline = time.monotonic() + spawn_timeout
    last_spawn = time.monotonic()
    while time.monotonic() < deadline:
        base_url = _healthy_base_url(data_dir)
        if base_url is not None and _claim_daemon(base_url):
            return base_url
        if proc.poll() is not None and time.monotonic() - last_spawn >= _RESPAWN_INTERVAL_SEC:
            proc = spawn()
            last_spawn = time.monotonic()
        time.sleep(0.1)
    raise TimeoutError(f"capture store daemon did not become healthy in {spawn_timeout}s")


@dataclass
class CaptureSession:
    """Handle yielded by :func:`capture_to_store`."""

    base_url: str
    exporter: CaptureExporter


@contextlib.contextmanager
def capture_to_store(
    run_id: str,
    *,
    data_dir: str | Path,
    run_attempt: int = 0,
    idle_shutdown_minutes: float | None = 60.0,
    flush_interval_sec: float = 1.0,
    drain_timeout_sec: float = 5.0,
    instrument_threads: bool = True,
    **scope_pairs: ScopeValue,
) -> Iterator[CaptureSession]:
    """Capture all instrumented SDK traffic in this context to the local store.

    Spawns (or reuses) the store daemon for ``data_dir``, instruments the
    Tinker SDK with an exporter feeding a :class:`StoreSink`, enables thread
    scope propagation (``instrument_threads=True`` by default, mirroring how
    the SDK instrumentation is automatic: forgetting a manual
    ``propagate.instrument_threads()`` call would silently yield
    unattributed rows from thread-pool code), and enters a
    ``capture(run_id=..., run_attempt=..., **scope_pairs)`` scope. On exit the
    previous instrumentation state is restored: if an enclosing context (a
    nested ``capture_to_store`` or a manual ``instrument_tinker``) was
    active, its exporter is swapped back in; otherwise the SDK is
    uninstrumented. Futures still outstanding are grace-drained: their done-callbacks hold the exporter snapshotted at call
    time, so teardown waits up to ``drain_timeout_sec`` for pending calls
    (``exporter.wait_pending``), force-flushes, then shuts the exporter down.
    Anything enqueued after shutdown is a counted-dropped no-op. If the store
    goes down mid-run, exports fail into the exporter's counters; the
    training process is never disturbed.
    """
    base_url = ensure_daemon(data_dir, idle_shutdown_minutes=idle_shutdown_minutes)
    exporter = CaptureExporter(StoreSink(base_url), flush_interval_sec=flush_interval_sec)
    # Nesting-aware: if instrumentation is already active (an enclosing
    # capture_to_store, or manual instrument_tinker), remember its exporter
    # and swap it back on exit instead of tearing the wrappers down out from
    # under the enclosing context.
    previous_exporter = current_exporter()
    instrument_tinker(exporter)
    # Thread propagation is nesting-aware the same way: only uninstrument on
    # exit if WE turned it on and no outer session (or independent caller)
    # already had it on.
    threads_were_instrumented = propagate.threads_instrumented()
    if instrument_threads:
        propagate.instrument_threads()
    try:
        with capture(run_id=run_id, run_attempt=run_attempt, **scope_pairs):
            yield CaptureSession(base_url=base_url, exporter=exporter)
    finally:
        # Stop routing new records to THIS session's exporter first;
        # in-flight calls keep their snapshotted exporter reference.
        if previous_exporter is not None:
            instrument_tinker(previous_exporter)
        else:
            uninstrument_tinker()
        if instrument_threads and not threads_were_instrumented:
            propagate.uninstrument_threads()
        exporter.wait_pending(timeout=drain_timeout_sec)
        exporter.force_flush(timeout=drain_timeout_sec)
        exporter.shutdown()
