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
from typing import Any, Protocol, runtime_checkable

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


class FsspecStorage:
    """Storage backend wrapping any ``fsspec.AbstractFileSystem``.

    Supports S3 (via ``s3fs``), GCS (via ``gcsfs``), Azure (via ``adlfs``),
    and any other filesystem that fsspec supports.

    **Append strategy:** Cloud backends don't support native append. This
    class stages append-only files locally using POSIX atomic writes, then
    uploads them to cloud on :meth:`flush`. Reads check the local stage
    first, so ``IncrementalReader`` sees appended data immediately.

    Call :meth:`flush` at checkpoints or training end to persist staged
    data to cloud. The context manager calls ``flush`` on exit.

    Pickle-serializable — stores protocol, root, and kwargs. Local staged
    data is NOT included in pickle (each process starts with an empty stage).

    Examples::

        storage = storage_from_uri("s3://my-bucket/experiments/run1")

        # Appends are staged locally (fast, atomic)
        storage.append("metrics.jsonl", b'{...}\\n')

        # Reads see staged data immediately
        data = storage.read("metrics.jsonl")

        # Upload staged files to cloud
        storage.flush()
    """

    def __init__(self, fs: Any, root: str = "", **fs_kwargs: Any) -> None:
        import tempfile

        self._fs = fs
        self._root = root.rstrip("/")
        self._protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        self._fs_kwargs = fs_kwargs
        # Local staging for append-only files
        self._stage_dir = Path(tempfile.mkdtemp(prefix="fsspec_stage_"))
        self._staged: set[str] = set()

    def _full(self, path: str) -> str:
        if not path:
            return self._root
        return f"{self._root}/{path}" if self._root else path

    def _stage_path(self, path: str) -> Path:
        return self._stage_dir / path

    def url(self, path: str = "") -> str:
        """Return a URI like ``s3://bucket/prefix/path``."""
        return f"{self._protocol}://{self._full(path)}"

    # --- Sync (Storage protocol) ---

    def read(self, path: str) -> bytes:
        """See :meth:`Storage.read`. Reads from local stage if available."""
        if path in self._staged:
            return self._stage_path(path).read_bytes()
        return self._fs.cat_file(self._full(path))

    def write(self, path: str, data: bytes) -> None:
        """See :meth:`Storage.write`. Writes directly to cloud."""
        full = self._full(path)
        self._fs.mkdirs(posixpath.dirname(full), exist_ok=True)
        self._fs.pipe_file(full, data)
        # If this path was staged, update the local copy too
        if path in self._staged:
            sp = self._stage_path(path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_bytes(data)

    def append(self, path: str, data: bytes) -> None:
        """See :meth:`Storage.append`.

        Stages appends locally using POSIX atomic writes. The first append
        for a given path pulls existing content from cloud (if any), then
        all subsequent appends are local. Call :meth:`flush` to upload
        staged data to cloud.
        """
        if path not in self._staged:
            # First append: pull existing content from cloud
            sp = self._stage_path(path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            try:
                existing = self._fs.cat_file(self._full(path))
                sp.write_bytes(existing)
            except FileNotFoundError:
                pass  # File doesn't exist yet — will be created by append
            self._staged.add(path)
        with open(self._stage_path(path), "ab") as f:
            f.write(data)

    def exists(self, path: str) -> bool:
        """See :meth:`Storage.exists`."""
        if path in self._staged:
            return self._stage_path(path).exists()
        return self._fs.exists(self._full(path))

    def stat(self, path: str) -> StorageStat | None:
        """See :meth:`Storage.stat`."""
        if path in self._staged:
            sp = self._stage_path(path)
            if not sp.exists():
                return None
            st = sp.stat()
            return StorageStat(size=st.st_size, mtime=st.st_mtime)
        try:
            info = self._fs.info(self._full(path))
            return StorageStat(
                size=info.get("size", 0),
                mtime=info.get("mtime", 0.0) or info.get("LastModified", 0.0) or 0.0,
            )
        except FileNotFoundError:
            return None

    def read_range(self, path: str, offset: int, length: int | None = None) -> bytes:
        """See :meth:`Storage.read_range`."""
        if path in self._staged:
            with open(self._stage_path(path), "rb") as f:
                f.seek(offset)
                return f.read(length) if length is not None else f.read()
        full = self._full(path)
        if length is not None:
            return self._fs.read_block(full, offset, length)
        with self._fs.open(full, "rb") as f:
            f.seek(offset)
            return f.read()

    def list_dir(self, prefix: str) -> list[str]:
        """See :meth:`Storage.list_dir`. Returns immediate children names only."""
        full = self._full(prefix)
        try:
            entries = self._fs.ls(full, detail=False)
        except FileNotFoundError:
            entries = []
        cloud_names = {
            e.split("/")[-1] for e in entries if e != full and e.rstrip("/") != full.rstrip("/")
        }
        # Merge with staged files at this prefix level
        stage_prefix = self._stage_dir / prefix if prefix else self._stage_dir
        if stage_prefix.is_dir():
            cloud_names.update(child.name for child in stage_prefix.iterdir())
        return sorted(cloud_names)

    def remove(self, path: str) -> None:
        """See :meth:`Storage.remove`."""
        with contextlib.suppress(FileNotFoundError):
            self._fs.rm_file(self._full(path))
        # Also remove from stage
        if path in self._staged:
            self._stage_path(path).unlink(missing_ok=True)
            self._staged.discard(path)

    def remove_dir(self, path: str) -> None:
        """See :meth:`Storage.remove_dir`."""
        with contextlib.suppress(FileNotFoundError, OSError):
            self._fs.rmdir(self._full(path))

    def flush(self) -> None:
        """Upload all locally staged files to cloud.

        Call this at checkpoints or training end. The context manager
        calls this automatically on exit.
        """
        for path in list(self._staged):
            sp = self._stage_path(path)
            if sp.exists():
                data = sp.read_bytes()
                full = self._full(path)
                self._fs.mkdirs(posixpath.dirname(full), exist_ok=True)
                self._fs.pipe_file(full, data)

    def close(self) -> None:
        """Flush staged data and clean up the local staging directory."""
        self.flush()
        import shutil

        shutil.rmtree(self._stage_dir, ignore_errors=True)
        self._staged.clear()

    def __enter__(self) -> FsspecStorage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup — don't rely on this
        import shutil

        shutil.rmtree(self._stage_dir, ignore_errors=True)

    # --- Async (via to_thread) ---

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

    async def __aenter__(self) -> FsspecStorage:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.close)

    # --- Pickle support ---

    def __getstate__(self) -> dict[str, Any]:
        # Staged data is NOT pickled — each process starts with empty stage
        return {"protocol": self._protocol, "root": self._root, "fs_kwargs": self._fs_kwargs}

    def __setstate__(self, state: dict[str, Any]) -> None:
        import tempfile

        import fsspec

        self._protocol = state["protocol"]
        self._root = state["root"]
        self._fs_kwargs = state["fs_kwargs"]
        self._fs = fsspec.filesystem(self._protocol, **self._fs_kwargs)
        self._stage_dir = Path(tempfile.mkdtemp(prefix="fsspec_stage_"))
        self._staged = set()


def storage_from_uri(uri: str, **kwargs: Any) -> Storage:
    """Create a Storage backend from a URI string.

    Supported schemes::

        /path/to/dir          → LocalStorage("/path/to/dir")
        file:///path/to/dir   → LocalStorage("/path/to/dir")
        s3://bucket/prefix    → FsspecStorage (requires s3fs)
        gs://bucket/prefix    → FsspecStorage (requires gcsfs)
        az://container/prefix → FsspecStorage (requires adlfs)

    Extra keyword arguments are passed to the fsspec filesystem constructor
    (e.g., ``anon=True`` for public S3 buckets).

    This is the recommended way to create a Storage from user config
    (e.g., ``log_path`` in training scripts).
    """
    if uri.startswith("file://"):
        return LocalStorage(uri[len("file://") :])
    if "://" in uri:
        # Cloud URI — delegate to fsspec
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                f"fsspec is required for cloud storage URIs ({uri}). "
                "Install it with: uv pip install fsspec"
            ) from None
        fs, path = fsspec.core.url_to_fs(uri, **kwargs)
        return FsspecStorage(fs, path, **kwargs)
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
