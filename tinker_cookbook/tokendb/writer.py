"""Buffered parquet segment writer for the token DB.

Layout under the run's ``log_path`` (all I/O through the ``Storage``
protocol, so S3/GCS log dirs work)::

    {log_path}/tokens/
      run.json                                   # run identity, coordinator-owned
      segments/seg-{writer_id}-{seq:06d}.parquet # immutable, one file per flush
      manifest-{writer_id}.jsonl                 # 1 line per segment (< 4KB appends)
      labels.jsonl                               # annotations (written by readers/agents)

Writer discipline: a segment file is fully written **before** its manifest
line is appended, so a crash between the two leaves an orphan segment, never
a dangling manifest entry. Segments are immutable per-writer files and each
manifest has exactly one appender, so multiple writers (other processes or
hosts) can share one store without coordination; readers glob
``manifest-*.jsonl`` and treat manifests as a liveness hint (the directory
listing is the source of truth).
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import io
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinker_cookbook.stores.storage import Storage, storage_from_uri
from tinker_cookbook.tokendb.schema import SCHEMA_VERSION, TokenRow, arrow_schema, row_to_record

logger = logging.getLogger(__name__)

TOKENS_DIR = "tokens"
SEGMENTS_DIR = f"{TOKENS_DIR}/segments"
RUN_JSON_PATH = f"{TOKENS_DIR}/run.json"

DEFAULT_BUFFER_ROWS = 2048
DEFAULT_FLUSH_INTERVAL_S = 5.0


def make_writer_id() -> str:
    """Return a writer ID unique across processes and hosts.

    ``hostname-pid-suffix``: hostname + pid disambiguate concurrent writers
    across hosts, and the random suffix guards against pid reuse (e.g. a
    restarted worker on the same host).
    """
    host = re.sub(r"[^A-Za-z0-9]+", "-", socket.gethostname()).strip("-")
    return f"{host}-{os.getpid()}-{secrets.token_hex(3)}"


def _encode_parquet(rows: Sequence[TokenRow]) -> bytes:
    """Encode rows as a zstd-compressed parquet file in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([row_to_record(row) for row in rows], schema=arrow_schema())
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


class TokenDbWriter:
    """Buffered token DB writer: rows in, immutable parquet segments out.

    Buffered rows are flushed to a **new** segment when the buffer reaches
    ``buffer_rows``, every ``flush_interval_s`` seconds (via a daemon
    thread), or on :meth:`close`. Thread-safe: appends may come from
    multiple threads or event-loop tasks; the flush swaps the buffer under a
    lock so encoding never touches a live buffer, and flushes are serialized
    so segment sequence numbers stay monotonic.

    Identity: the coordinator constructs the writer without ``run_id`` /
    ``run_attempt`` in *context*; it then owns ``run.json``, incrementing
    ``run_attempt`` each construction (a run resumed from a checkpoint
    re-runs iterations, so rows are stamped with the attempt that produced
    them). Workers on other processes/hosts are handed explicit ``run_id``
    and ``run_attempt`` in *context* and never touch ``run.json``.

    Args:
        storage_or_log_path: A ``Storage`` backend, or a path/URI for
            ``storage_from_uri`` (local path, ``s3://``, ``gs://``, ...).
        writer_id: Override the auto-generated writer ID.
        context: Run metadata recorded in ``run.json`` (e.g. ``model_name``).
            If it contains ``run_id`` and ``run_attempt``, this writer acts
            as a worker and does not touch ``run.json``.
        buffer_rows: Flush when the buffer reaches this many rows.
        flush_interval_s: Background flush period in seconds.
    """

    def __init__(
        self,
        storage_or_log_path: Storage | str | Path,
        *,
        writer_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        buffer_rows: int = DEFAULT_BUFFER_ROWS,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        if isinstance(storage_or_log_path, (str, Path)):
            self._storage: Storage = storage_from_uri(str(storage_or_log_path))
        else:
            self._storage = storage_or_log_path
        self.writer_id = writer_id if writer_id is not None else make_writer_id()
        self._context: dict[str, Any] = dict(context or {})
        self._buffer_rows = buffer_rows
        self._flush_interval_s = flush_interval_s

        self._buffer: list[TokenRow] = []
        self._buffer_lock = threading.Lock()  # guards _buffer
        self._flush_lock = threading.Lock()  # serializes flushes (monotonic seq)
        self._seq = 0
        self._closed = False

        if "run_id" in self._context and "run_attempt" in self._context:
            # Worker: identity passed in explicitly; run.json stays
            # coordinator-owned.
            self.run_id = str(self._context["run_id"])
            self.run_attempt = int(self._context["run_attempt"])
        else:
            self.run_id, self.run_attempt = self._init_run_json()

        self._manifest_path = f"{TOKENS_DIR}/manifest-{self.writer_id}.jsonl"

        atexit.register(self._atexit_close)
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name=f"tokendb-flush-{self.writer_id}", daemon=True
        )
        self._flush_thread.start()

    def _init_run_json(self) -> tuple[str, int]:
        """Read/create ``run.json``, incrementing ``run_attempt`` on resume."""
        run_id: str | None = None
        run_attempt = 1
        if self._storage.exists(RUN_JSON_PATH):
            try:
                existing = json.loads(self._storage.read(RUN_JSON_PATH).decode())
                run_id = existing.get("run_id")
                run_attempt = int(existing.get("run_attempt", 0)) + 1
            except Exception:
                logger.exception(
                    "Failed to read existing %s; starting a new run identity", RUN_JSON_PATH
                )
        if not run_id:
            seed = f"{self._storage.url('')}:{time.time_ns()}"
            run_id = hashlib.sha1(seed.encode()).hexdigest()[:16]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "writer_id": self.writer_id,
            "context": self._context,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._storage.write(RUN_JSON_PATH, (json.dumps(payload, default=str) + "\n").encode())
        return run_id, run_attempt

    def append_rows(self, rows: Sequence[TokenRow]) -> None:
        """Stamp writer identity on *rows* and buffer them for writing."""
        if self._closed:
            raise RuntimeError("append_rows() called on a closed TokenDbWriter")
        for row in rows:
            row.run_id = self.run_id
            row.run_attempt = self.run_attempt
            row.writer_id = self.writer_id
        with self._buffer_lock:
            self._buffer.extend(rows)
            should_flush = len(self._buffer) >= self._buffer_rows
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows to a new segment, then append its manifest line.

        No-op when the buffer is empty. Safe to call from any thread; when
        called from async code, prefer :meth:`aflush` to keep the parquet
        encode off the event loop.
        """
        with self._flush_lock:
            with self._buffer_lock:
                rows, self._buffer = self._buffer, []
            if not rows:
                return
            seq = self._seq
            self._seq += 1
            segment_name = f"seg-{self.writer_id}-{seq:06d}.parquet"
            data = _encode_parquet(rows)
            # Segment fully written BEFORE its manifest line: a crash between
            # the two leaves an orphan segment, never a dangling manifest entry.
            self._storage.write(f"{SEGMENTS_DIR}/{segment_name}", data)
            manifest_line = {
                "path": f"segments/{segment_name}",
                "n_rows": len(rows),
                "min_iteration": min(row.iteration for row in rows),
                "max_iteration": max(row.iteration for row in rows),
                "min_ts": min(row.ts for row in rows).isoformat(),
                "max_ts": max(row.ts for row in rows).isoformat(),
                "run_attempt": self.run_attempt,
                "writer_id": self.writer_id,
                "schema_version": SCHEMA_VERSION,
            }
            self._storage.append(self._manifest_path, (json.dumps(manifest_line) + "\n").encode())

    async def aflush(self) -> None:
        """Async :meth:`flush` — runs the parquet encode + write in a thread."""
        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        """Flush remaining rows and stop the background flusher. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._flush_thread.join(timeout=self._flush_interval_s + 1.0)
        self.flush()
        self._storage.flush()
        atexit.unregister(self._atexit_close)

    def _atexit_close(self) -> None:
        """Best-effort flush at interpreter exit (crash paths that skip close())."""
        try:
            self.close()
        except Exception:
            logger.exception("Best-effort token DB flush at exit failed")

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval_s):
            try:
                self.flush()
            except Exception:
                logger.exception("Background token DB flush failed")

    def __enter__(self) -> TokenDbWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ParquetSegmentBackend:
    """Parquet-segments-through-``Storage`` implementation of ``TokenStoreBackend``.

    The default (v1) backend: the writer is a buffered segment flusher
    (:class:`TokenDbWriter`); the read half (DuckDB over the segment glob)
    arrives in a later phase.
    """

    def __init__(
        self,
        storage_or_log_path: Storage | str | Path,
        *,
        buffer_rows: int = DEFAULT_BUFFER_ROWS,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        if isinstance(storage_or_log_path, (str, Path)):
            self._storage: Storage = storage_from_uri(str(storage_or_log_path))
        else:
            self._storage = storage_or_log_path
        self._buffer_rows = buffer_rows
        self._flush_interval_s = flush_interval_s

    def open_writer(self, run_context: Mapping[str, Any]) -> TokenDbWriter:
        """See :meth:`TokenStoreBackend.open_writer`."""
        return TokenDbWriter(
            self._storage,
            context=run_context,
            buffer_rows=self._buffer_rows,
            flush_interval_s=self._flush_interval_s,
        )

    # --- Read half: implemented in a later phase. ---

    def query(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.query`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")

    def get_rollout(self, split: str, iteration: int, group_idx: int, traj_idx: int) -> Any:
        """See :meth:`TokenStoreBackend.get_rollout`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")

    def search(self, pattern: str, *, text_field: str = "ac_text") -> Any:
        """See :meth:`TokenStoreBackend.search`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")

    def subscribe(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.subscribe`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        """See :meth:`TokenStoreBackend.add_label`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")

    def labels(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.labels`."""
        raise NotImplementedError("ParquetSegmentBackend read path is not implemented yet")
