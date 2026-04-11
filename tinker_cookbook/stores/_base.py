"""BaseStore — shared I/O helpers for all typed stores.

Provides path resolution and JSON/JSONL read/write methods on top of
the ``Storage`` protocol.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tinker_cookbook.stores.storage import Storage, storage_join

logger = logging.getLogger(__name__)


class BaseStore:
    """Common storage helpers shared by TrainingRunStore, EvalStore, etc.

    All file I/O goes through the ``Storage`` protocol — no direct
    ``Path``/``open()`` usage.
    """

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        self.storage = storage
        self.prefix = prefix

    def _path(self, *parts: str) -> str:
        return storage_join(self.prefix, *parts)

    # ── JSON/JSONL helpers ────────────────────────────────────────────

    def _read_json(self, *parts: str) -> dict[str, Any] | None:
        try:
            data = self.storage.read(self._path(*parts))
            return json.loads(data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read JSON %s: %s", self._path(*parts), e)
            return None

    def _write_json(self, data: dict[str, Any], *parts: str) -> None:
        self.storage.write(self._path(*parts), json.dumps(data, indent=2).encode("utf-8"))

    def _read_jsonl(self, *parts: str) -> list[dict[str, Any]]:
        try:
            data = self.storage.read(self._path(*parts))
        except FileNotFoundError:
            return []
        except OSError as e:
            logger.warning("Failed to read JSONL %s: %s", self._path(*parts), e)
            return []

        records: list[dict[str, Any]] = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", self._path(*parts))
        return records

    def _append_jsonl(self, record: dict[str, Any], *parts: str) -> None:
        self.storage.append(self._path(*parts), (json.dumps(record) + "\n").encode("utf-8"))
