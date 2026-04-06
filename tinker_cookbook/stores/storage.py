"""Storage protocols and LocalStorage implementation.

Pure byte-level I/O — no JSON, no application logic. The ``Storage`` protocol
is the extension point for adding S3, GCS, or other backends.

Design intent: ALL file I/O in tinker-cookbook should go through ``Storage``
so that cloud backends (S3, GCS) work end-to-end for both reads and writes.

**Cloud backend contract:**

- ``append()`` — Must be supported by all backends. Cloud backends may
  implement this via read-modify-write, append blobs (Azure), or sharding.
  Callers should keep individual appends small (< 4KB) and tolerate
  last-append loss on crash.
- ``list_dir()`` — For flat key-spaces (S3/GCS), list objects sharing a
  prefix up to the next ``/`` delimiter. Return names only (not full keys).
- ``stat()`` — Maps to HEAD requests on cloud. ``mtime`` may have
  second-level granularity on some backends.
- ``read_range()`` — Maps to Range GET on cloud.
- ``remove_dir()`` — No-op on backends without real directories (S3/GCS).
- ``url()`` — Returns a human-readable URI (``file:///``, ``s3://``, ``gs://``).

**Migration status:**

- Done: read paths (TrainingRunStore, EvalStore, RunRegistry)
- TODO phase 1: ``ml_log.JsonLogger``, ``utils/trace.py``, ``checkpoint_utils``
- TODO phase 2: cloud backends (``FsspecStorage`` wrapping fsspec)
- TODO phase 3: eval runner (``_runner.py`` — see docstring there for details)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageStat:
    """File metadata."""

    size: int
    mtime: float  # seconds since epoch


@runtime_checkable
class Storage(Protocol):
    """Sync byte-level file I/O.

    All paths are relative strings (e.g., ``"runs/001/metrics.jsonl"``).
    The backend resolves them against its root.

    Implementations must be **pickle-serializable** (for Ray/multiprocessing)
    — store only config (bucket, prefix, credentials path), not connections.
    """

    def url(self, path: str = "") -> str:
        """Return a human-readable URI for a path.

        Examples::

            LocalStorage("/data").url("metrics.jsonl")  # "file:///data/metrics.jsonl"
            S3Storage("bucket", "pfx").url("metrics.jsonl")  # "s3://bucket/pfx/metrics.jsonl"

        Used for logging, display, and interop with tools that accept URIs.
        """
        ...

    def read(self, path: str) -> bytes:
        """Read entire file. Raises ``FileNotFoundError`` if missing."""
        ...

    def write(self, path: str, data: bytes) -> None:
        """Write data, creating parent dirs. Overwrites if exists."""
        ...

    def append(self, path: str, data: bytes) -> None:
        """Append data to a file, creating if needed.

        All backends must support this. Callers should keep individual
        appends small (< 4KB for POSIX atomicity). Cloud backends may
        implement via read-modify-write, append blobs (Azure), or
        internal buffering — the choice is transparent to callers.

        On crash, the last append may be lost. Callers must tolerate this.
        """
        ...

    def exists(self, path: str) -> bool:
        """Return ``True`` if the file exists."""
        ...

    def stat(self, path: str) -> StorageStat | None:
        """Get file size and mtime, or ``None`` if missing."""
        ...

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Read bytes from offset. If length is None, read to end.

        Raises ``FileNotFoundError`` if missing.
        Maps to Range GET on cloud backends.
        """
        ...

    def list_dir(self, prefix: str) -> list[str]:
        """List immediate children under prefix. Returns names only.

        For flat key-spaces (S3/GCS), this lists keys sharing the prefix
        up to the next ``/`` delimiter.
        """
        ...

    def remove(self, path: str) -> None:
        """Delete a file. No error if missing."""
        ...

    def remove_dir(self, path: str) -> None:
        """Remove an empty directory. No error if missing or non-empty.

        Cloud backends (S3/GCS) can treat this as a no-op since they
        don't have real directories.
        """
        ...

    def __enter__(self) -> Storage: ...

    def __exit__(self, *exc: object) -> None: ...


@runtime_checkable
class AsyncStorage(Protocol):
    """Async byte-level file I/O.

    Cloud backends (S3, GCS) implement this natively. LocalStorage
    wraps sync methods with ``asyncio.to_thread``.
    """

    async def aread(self, path: str) -> bytes:
        """Async version of :meth:`Storage.read`."""
        ...

    async def awrite(self, path: str, data: bytes) -> None:
        """Async version of :meth:`Storage.write`."""
        ...

    async def aappend(self, path: str, data: bytes) -> None:
        """Async version of :meth:`Storage.append`."""
        ...

    async def aexists(self, path: str) -> bool:
        """Async version of :meth:`Storage.exists`."""
        ...

    async def astat(self, path: str) -> StorageStat | None:
        """Async version of :meth:`Storage.stat`."""
        ...

    async def aread_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Async version of :meth:`Storage.read_range`."""
        ...

    async def alist_dir(self, prefix: str) -> list[str]:
        """Async version of :meth:`Storage.list_dir`."""
        ...

    async def aremove(self, path: str) -> None:
        """Async version of :meth:`Storage.remove`."""
        ...

    async def aremove_dir(self, path: str) -> None:
        """Async version of :meth:`Storage.remove_dir`."""
        ...

    async def __aenter__(self) -> AsyncStorage: ...

    async def __aexit__(self, *exc: object) -> None: ...


class LocalStorage:
    """File-based storage rooted at a local directory.

    Implements both ``Storage`` (sync) and ``AsyncStorage`` (via ``to_thread``).
    Pickle-serializable (stores only a ``Path``).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        """The resolved local directory root."""
        return self._root

    def url(self, path: str = "") -> str:
        """Return a ``file:///`` URI for the given path."""
        resolved = self._root / path if path else self._root
        return resolved.as_uri()

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes storage root: {path}")
        return resolved

    # --- Sync (Storage protocol) --- see Storage for full docstrings

    def read(self, path: str) -> bytes:
        """See :meth:`Storage.read`."""
        return self._resolve(path).read_bytes()

    def write(self, path: str, data: bytes) -> None:
        """See :meth:`Storage.write`."""
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def append(self, path: str, data: bytes) -> None:
        """See :meth:`Storage.append`."""
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "ab") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        """See :meth:`Storage.exists`."""
        return self._resolve(path).exists()

    def stat(self, path: str) -> StorageStat | None:
        """See :meth:`Storage.stat`."""
        try:
            st = self._resolve(path).stat()
            return StorageStat(size=st.st_size, mtime=st.st_mtime)
        except FileNotFoundError:
            return None

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """See :meth:`Storage.read_range`."""
        with open(self._resolve(path), "rb") as f:
            f.seek(offset)
            return f.read(length) if length is not None else f.read()

    def list_dir(self, prefix: str) -> list[str]:
        """See :meth:`Storage.list_dir`. Returns sorted names."""
        full = self._resolve(prefix)
        if not full.is_dir():
            return []
        return sorted(child.name for child in full.iterdir())

    def remove(self, path: str) -> None:
        """See :meth:`Storage.remove`."""
        full = self._resolve(path)
        full.unlink(missing_ok=True)

    def remove_dir(self, path: str) -> None:
        """See :meth:`Storage.remove_dir`."""
        full = self._resolve(path)
        with contextlib.suppress(FileNotFoundError, OSError):
            full.rmdir()

    def __enter__(self) -> LocalStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    # --- Async (AsyncStorage protocol, via to_thread) ---

    async def aread(self, path: str) -> bytes:
        """Async version of :meth:`read`."""
        return await asyncio.to_thread(self.read, path)

    async def awrite(self, path: str, data: bytes) -> None:
        """Async version of :meth:`write`."""
        await asyncio.to_thread(self.write, path, data)

    async def aappend(self, path: str, data: bytes) -> None:
        """Async version of :meth:`append`."""
        await asyncio.to_thread(self.append, path, data)

    async def aexists(self, path: str) -> bool:
        """Async version of :meth:`exists`."""
        return await asyncio.to_thread(self.exists, path)

    async def astat(self, path: str) -> StorageStat | None:
        """Async version of :meth:`stat`."""
        return await asyncio.to_thread(self.stat, path)

    async def aread_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """Async version of :meth:`read_range`."""
        return await asyncio.to_thread(self.read_range, path, offset, length)

    async def alist_dir(self, prefix: str) -> list[str]:
        """Async version of :meth:`list_dir`."""
        return await asyncio.to_thread(self.list_dir, prefix)

    async def aremove(self, path: str) -> None:
        """Async version of :meth:`remove`."""
        await asyncio.to_thread(self.remove, path)

    async def aremove_dir(self, path: str) -> None:
        """Async version of :meth:`remove_dir`."""
        await asyncio.to_thread(self.remove_dir, path)

    async def __aenter__(self) -> LocalStorage:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


def storage_from_uri(uri: str) -> Storage:
    """Create a Storage backend from a URI string.

    Supported schemes::

        /path/to/dir          → LocalStorage("/path/to/dir")
        file:///path/to/dir   → LocalStorage("/path/to/dir")
        s3://bucket/prefix    → raises (not yet implemented)
        gs://bucket/prefix    → raises (not yet implemented)

    This is the recommended way to create a Storage from user config
    (e.g., ``log_path`` in training scripts).
    """
    if uri.startswith("file://"):
        return LocalStorage(uri[len("file://") :])
    if uri.startswith("s3://"):
        raise NotImplementedError(
            "S3 storage not yet implemented. Install a cloud storage extra when available."
        )
    if uri.startswith("gs://"):
        raise NotImplementedError(
            "GCS storage not yet implemented. Install a cloud storage extra when available."
        )
    if uri.startswith("az://") or uri.startswith("abfs://"):
        raise NotImplementedError(
            "Azure Blob storage not yet implemented. Install a cloud storage extra when available."
        )
    # Default: treat as local path
    return LocalStorage(uri)


def storage_join(*parts: str) -> str:
    """Join storage path segments, normalizing the result.

    Strips leading/trailing slashes from each segment, filters empty
    strings, and applies ``posixpath.normpath`` to collapse redundant
    separators and ``.``/``..`` components. Returns ``""`` for empty
    input (not ``"."`` which normpath would produce).
    """
    stripped = [p.strip("/") for p in parts]
    joined = "/".join(p for p in stripped if p)
    if not joined:
        return ""
    return posixpath.normpath(joined)
