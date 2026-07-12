"""Buffered parquet segment writer for the token DB.

Layout under the run's ``log_path`` (all I/O through the ``Storage``
protocol, so S3/GCS log dirs work)::

    {log_path}/tokens/
      run.json                                   # run identity, coordinator-owned (latest attempt)
      run-attempts.jsonl                         # 1 line per attempt (appended, never rewritten)
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
RUN_ATTEMPTS_PATH = f"{TOKENS_DIR}/run-attempts.jsonl"

DEFAULT_BUFFER_ROWS = 2048
DEFAULT_FLUSH_INTERVAL_S = 5.0

# Observed-keys manifest lists are capped so a manifest line stays well under
# the 4KB append-atomicity budget even for pathological key cardinalities.
MANIFEST_MAX_KEYS = 200


def _observed_keys(values: set[str]) -> tuple[list[str], bool]:
    """Sorted observed-keys list for a manifest line, capped at
    :data:`MANIFEST_MAX_KEYS` (returns ``(keys, truncated)``)."""
    keys = sorted(values)
    if len(keys) > MANIFEST_MAX_KEYS:
        return keys[:MANIFEST_MAX_KEYS], True
    return keys, False


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
        registry_dir: Run registry directory for the multi-run viewer
            (coordinator mode only). ``None`` resolves via the
            ``TINKER_TOKENDB_REGISTRY`` env var, falling back to
            ``~/.cache/tinker-cookbook/tokendb/runs``; an explicit ``""``
            disables registration. Registration is best-effort and never
            breaks training.
    """

    def __init__(
        self,
        storage_or_log_path: Storage | str | Path,
        *,
        writer_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        buffer_rows: int = DEFAULT_BUFFER_ROWS,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        registry_dir: str | None = None,
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
            # Coordinator-only: record this run in the local run registry so
            # the multi-run viewer can find it. Best-effort by design.
            self._register_run(registry_dir)

        self._manifest_path = f"{TOKENS_DIR}/manifest-{self.writer_id}.jsonl"

        atexit.register(self._atexit_close)
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name=f"tokendb-flush-{self.writer_id}", daemon=True
        )
        self._flush_thread.start()

    def _init_run_json(self) -> tuple[str, int]:
        """Read/create ``run.json``, incrementing ``run_attempt`` on resume.

        Run identity is recorded twice, by design:

        - ``run.json`` is overwritten each attempt and always holds the
          **latest** attempt (kept for compatibility: readers use it for
          quick identity/tokenizer lookup).
        - ``run-attempts.jsonl`` gets the same payload **appended** as one
          line per attempt, so per-attempt context (which can change on a
          resume, e.g. a different learning rate) is never lost. The
          reader's ``runs`` view is materialized from this file.
        """
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
        encoded = (json.dumps(payload, default=str) + "\n").encode()
        self._storage.write(RUN_JSON_PATH, encoded)
        # Append-per-attempt record (see the docstring). Best-effort: identity
        # is already durable in run.json, so a failed append must not break
        # writer construction.
        try:
            self._storage.append(RUN_ATTEMPTS_PATH, encoded)
            # Upload the staged append now (no-op on LocalStorage) so the
            # attempt record is visible on cloud stores from the start of the
            # run, not only after the first segment flush.
            self._storage.flush()
        except Exception:
            logger.warning("Failed to append %s (training continues)", RUN_ATTEMPTS_PATH)
        return run_id, run_attempt

    def _register_run(self, registry_dir: str | None) -> None:
        """Best-effort record of this run in the multi-run viewer registry."""
        from tinker_cookbook.tokendb.registry import register_run

        register_run(
            log_path=self._storage.url(""),
            run_id=self.run_id,
            run_attempt=self.run_attempt,
            model_name=self._context.get("model_name"),
            recipe_name=self._context.get("recipe_name"),
            writer_id=self.writer_id,
            registry_dir=registry_dir,
        )

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

    def record_sample(
        self,
        model_input: Any,
        sequence: Any,
        *,
        split: str = "sample",
        iteration: int = -1,
        group_idx: int = 0,
        traj_idx: int = 0,
        tags: Sequence[str] = (),
        tokenizer: Any = None,
        store_text: bool = True,
        attrs: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
        **metadata: Any,
    ) -> TokenRow:
        """Record one directly sampled sequence as a ``source="sample"`` row.

        Explicit capture API for code that calls ``sample_async`` itself
        (bypassing the completers, and therefore the
        ``capture_samples`` sink). Thin wrapper: builds the row with
        :func:`~tinker_cookbook.tokendb.capture.sample_to_row` and buffers it.

        Args:
            model_input: The ``tinker.ModelInput`` prompt.
            sequence: The sampled sequence (``tinker.SampledSequence`` or
                anything with ``tokens`` / ``logprobs`` / ``stop_reason``).
            split: Row split; defaults to ``"sample"``.
            iteration: Row iteration; defaults to ``-1`` (no training step).
            group_idx: Caller-chosen grouping index (e.g. prompt index).
            traj_idx: Index of this sequence within its sample call.
            tags: Logging tags for the row.
            tokenizer: Optional ``decode(list[int]) -> str`` for text columns.
            store_text: Store decoded text when *tokenizer* is given.
            attrs: Categorical dimensions (teacher model, source dataset, ...)
                for the typed ``attrs`` string-map column. The reserved key
                ``"row_id"`` promotes to the ``env_row_id`` column.
            metrics: Numeric per-row values for the typed ``metrics``
                float-map column.
            **metadata: Recorded in the row's free-form ``extra`` JSON column.

        Returns:
            The buffered :class:`TokenRow`.
        """
        from tinker_cookbook.tokendb.capture import sample_to_row

        row = sample_to_row(
            model_input,
            sequence,
            split=split,
            iteration=iteration,
            group_idx=group_idx,
            traj_idx=traj_idx,
            tags=tags,
            tokenizer=tokenizer,
            store_text=store_text,
            attrs=attrs,
            metrics=metrics,
            extra=metadata,
        )
        self.append_rows([row])
        return row

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
            # Observed-keys manifest: the sorted unique metrics/attrs/
            # token_metrics keys and tags seen in this flush, so readers can
            # build a per-run schema card without scanning segment bytes.
            metrics_keys, metrics_truncated = _observed_keys(
                {key for row in rows for key in row.metrics}
            )
            attrs_keys, attrs_truncated = _observed_keys({key for row in rows for key in row.attrs})
            token_metrics_keys, token_metrics_truncated = _observed_keys(
                {key for row in rows for key in row.token_metrics}
            )
            tags, tags_truncated = _observed_keys({tag for row in rows for tag in row.tags})
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
                "metrics_keys": metrics_keys,
                "attrs_keys": attrs_keys,
                "token_metrics_keys": token_metrics_keys,
                "tags": tags,
                "keys_truncated": metrics_truncated
                or attrs_truncated
                or token_metrics_truncated
                or tags_truncated,
            }
            self._storage.append(self._manifest_path, (json.dumps(manifest_line) + "\n").encode())
            # Push staged appends to the backing store. FsspecStorage stages
            # append()s locally and only uploads on flush(), so without this
            # a gs:///s3:// store would have segments but no visible manifest
            # lines until close() — readers treating manifests as a liveness
            # hint would see the run as dead all through training.
            # LocalStorage.flush() is a no-op; FsspecStorage re-uploads only
            # files with staged appends (here: this writer's manifest, which
            # stays small — one short JSON line per segment).
            self._storage.flush()

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
    (:class:`TokenDbWriter`); the read half is a DuckDB reader over the
    segment files (:class:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader`,
    constructed lazily so the write path never needs duckdb).
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
        self._reader_instance: Any = None

    def open_writer(self, run_context: Mapping[str, Any]) -> TokenDbWriter:
        """See :meth:`TokenStoreBackend.open_writer`."""
        return TokenDbWriter(
            self._storage,
            context=run_context,
            buffer_rows=self._buffer_rows,
            flush_interval_s=self._flush_interval_s,
        )

    # --- Read half: delegates to a lazily constructed DuckDB reader. ---

    def _reader(self) -> Any:
        """Return the (lazily constructed) DuckDB reader for this store."""
        if self._reader_instance is None:
            from tinker_cookbook.tokendb.reader import ParquetSegmentReader

            self._reader_instance = ParquetSegmentReader(self._storage)
        return self._reader_instance

    def query(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.query` and
        :meth:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader.query`."""
        return self._reader().query(**filters)

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
    ) -> Any:
        """See :meth:`TokenStoreBackend.get_rollout`."""
        return self._reader().get_rollout(split, iteration, group_idx, traj_idx, run_attempt)

    def search(self, **kwargs: Any) -> Any:
        """See :meth:`TokenStoreBackend.search` and
        :meth:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader.search`."""
        return self._reader().search(**kwargs)

    def search_hit_counts(self, **kwargs: Any) -> Any:
        """Grouped-by-iteration hit counts for a :meth:`search` call."""
        return self._reader().search_hit_counts(**kwargs)

    def refresh(self) -> Any:
        """See :meth:`TokenStoreBackend.refresh`; returns newly registered
        segment file names."""
        return self._reader().refresh()

    def trajectories(
        self,
        *,
        latest_only: bool = False,
        limit: int = 500,
        offset: int = 0,
        **filters: Any,
    ) -> Any:
        """See :meth:`TokenStoreBackend.trajectories` and
        :meth:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader.trajectories`."""
        return self._reader().trajectories(
            latest_only=latest_only, limit=limit, offset=offset, **filters
        )

    def dashboard_stats(self, *, recent_k: int = 5, series_len: int = 50) -> Any:
        """See :meth:`TokenStoreBackend.dashboard_stats`."""
        return self._reader().dashboard_stats(recent_k=recent_k, series_len=series_len)

    def group_traj_idxs(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        run_attempt: int | None = None,
    ) -> Any:
        """See :meth:`TokenStoreBackend.group_traj_idxs`."""
        return self._reader().group_traj_idxs(split, iteration, group_idx, run_attempt)

    def runs(self) -> Any:
        """One row per (run_id, run_attempt); see
        :meth:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader.runs`."""
        return self._reader().runs()

    def schema_card(self) -> Any:
        """Observed-keys card for this store; see
        :meth:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader.schema_card`."""
        return self._reader().schema_card()

    def subscribe(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.subscribe`."""
        return self._reader().subscribe(**filters)

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
        self._reader().add_label(key, label_key, label_value, author=author, note=note)

    def labels(self, **filters: Any) -> Any:
        """See :meth:`TokenStoreBackend.labels`."""
        return self._reader().labels(**filters)

    # Backend-specific escape hatch, deliberately NOT part of TokenStoreBackend
    # (SQL dialects differ across backends; the portable surface is above).

    def sql(self, query: str, params: Sequence[Any] | None = None) -> Any:
        """Read-only (SELECT-only) DuckDB SQL over this store's views."""
        return self._reader().sql(query, params)
