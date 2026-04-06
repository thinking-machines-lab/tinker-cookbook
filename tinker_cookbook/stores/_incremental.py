"""Incremental JSONL reader with byte-offset tracking.

Used by TrainingRunStore for metrics and timing — both are append-only
files that grow during training.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from tinker_cookbook.stores.storage import Storage

logger = logging.getLogger(__name__)

_MAX_RECORDS = 50_000


class IncrementalReader:
    """Reads a JSONL file incrementally, tracking the byte offset.

    Only new bytes since the last ``read()`` are parsed. Memory is bounded
    to ``max_records`` — oldest records are dropped when the limit is hit.

    Uses ``Storage.read_range()`` for efficient partial reads.

    Thread-safe: a ``threading.Lock`` protects all mutations so concurrent
    SSE connections sharing the same ``TrainingRunStore`` don't race.

    Created lazily by stores to preserve pickle-serializability of the
    parent store object.
    """

    def __init__(self, storage: Storage, path: str, max_records: int = _MAX_RECORDS) -> None:
        self._storage = storage
        self._path = path
        self._max_records = max_records
        self._offset: int = 0
        self._records: list[dict[str, Any]] = []
        self._known_keys: set[str] = set()
        self._total_read: int = 0
        self._lock = threading.Lock()

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    @property
    def total_read(self) -> int:
        return self._total_read

    @property
    def known_keys(self) -> set[str]:
        """All metric keys seen (excluding 'step')."""
        with self._lock:
            return set(self._known_keys)

    def read(self) -> list[dict[str, Any]]:
        """Read new lines since last call. Returns only new records."""
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> list[dict[str, Any]]:
        stat = self._storage.stat(self._path)
        if stat is None or stat.size <= self._offset:
            return []

        try:
            new_bytes = self._storage.read_range(self._path, self._offset)
        except FileNotFoundError:
            return []

        if not new_bytes.endswith(b"\n"):
            last_nl = new_bytes.rfind(b"\n")
            if last_nl == -1:
                return []
            new_bytes = new_bytes[: last_nl + 1]

        self._offset += len(new_bytes)
        raw = new_bytes.decode("utf-8", errors="replace")

        new_records: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                new_records.append(record)
                self._known_keys.update(k for k in record if k != "step")
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line: %s", line[:100])

        self._records.extend(new_records)
        self._total_read += len(new_records)

        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        return new_records

    async def aread(self) -> list[dict[str, Any]]:
        """Async version of ``read()``."""
        return await asyncio.to_thread(self.read)

    def has_data(self) -> bool:
        with self._lock:
            if self._records:
                return True
        stat = self._storage.stat(self._path)
        return stat is not None and stat.size > 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
