"""Central registry of token DB runs for the multi-run viewer.

Every coordinator :class:`~tinker_cookbook.tokendb.writer.TokenDbWriter`
registers its run as **one JSON file per run** under a registry directory
(default ``~/.cache/tinker-cookbook/tokendb/runs/``), so concurrent training
jobs never share an append target. The viewer server, started with no
``log_path``, lists this directory to show all runs (especially
currently-running ones) in one dashboard.

Resolution order for the registry directory: explicit argument, then the
``TINKER_TOKENDB_REGISTRY`` environment variable, then the default. An empty
string at any level disables the registry entirely.

Registration is best-effort by design: a broken registry (unwritable
directory, weird ``HOME``) must never break training, so
:func:`register_run` logs a warning and returns ``None`` instead of raising.

Liveness is inferred from the run's own store, not from the registry: a run
is "live" if any of its ``manifest-*.jsonl`` files was modified within the
last :data:`DEFAULT_LIVE_WINDOW_S` seconds (writers flush at least every few
seconds while training). :func:`run_status` computes this cheaply from
manifest stats and last lines, never loading segment data.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.tokendb.writer import TOKENS_DIR

logger = logging.getLogger(__name__)

REGISTRY_ENV_VAR = "TINKER_TOKENDB_REGISTRY"
DEFAULT_REGISTRY_DIR = "~/.cache/tinker-cookbook/tokendb/runs"
DEFAULT_LIVE_WINDOW_S = 120.0


def resolve_registry_dir(registry_dir: str | None = None) -> Path | None:
    """Resolve the registry directory, or ``None`` when the registry is disabled.

    Precedence: explicit *registry_dir*, then the ``TINKER_TOKENDB_REGISTRY``
    environment variable, then :data:`DEFAULT_REGISTRY_DIR`. An empty string
    (at any level) disables the registry.
    """
    if registry_dir is None:
        registry_dir = os.environ.get(REGISTRY_ENV_VAR)
    if registry_dir is None:
        registry_dir = DEFAULT_REGISTRY_DIR
    if registry_dir == "":
        return None
    return Path(registry_dir).expanduser()


def normalize_log_path(log_path: str) -> str:
    """Return a resolved form of *log_path* usable to reopen the store later.

    ``file://`` URIs and plain local paths become absolute local paths; cloud
    URIs (``gs://``, ``s3://``, ...) pass through unchanged.
    """
    if log_path.startswith("file://"):
        return str(Path(unquote(urlparse(log_path).path)))
    if "://" in log_path:
        return log_path
    return str(Path(log_path).expanduser().resolve())


def register_run(
    *,
    log_path: str,
    run_id: str,
    run_attempt: int,
    model_name: str | None = None,
    recipe_name: str | None = None,
    writer_id: str | None = None,
    registry_dir: str | None = None,
) -> Path | None:
    """Write (or overwrite) this run's registry record ``{run_id}.json``.

    Best-effort: returns the record path on success, or ``None`` when the
    registry is disabled or the write failed (logged, never raised), so a
    broken registry cannot break training.
    """
    try:
        directory = resolve_registry_dir(registry_dir)
        if directory is None:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "run_attempt": run_attempt,
            "log_path": normalize_log_path(log_path),
            "model_name": model_name,
            "recipe_name": recipe_name,
            "started_at": datetime.now(UTC).isoformat(),
            "writer_id": writer_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        target = directory / f"{run_id}.json"
        # Write-then-rename so a concurrent reader never sees a partial record.
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return target
    except Exception:
        logger.warning(
            "Failed to register token DB run %s in the run registry (training continues)",
            run_id,
            exc_info=True,
        )
        return None


def load_run_record(registry_dir: str | None, run_id: str) -> dict[str, Any] | None:
    """Read one run's registry record, or ``None`` if missing/unreadable."""
    directory = resolve_registry_dir(registry_dir)
    if directory is None:
        return None
    path = directory / f"{run_id}.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def run_status(log_path: str, *, live_window_s: float = DEFAULT_LIVE_WINDOW_S) -> dict[str, Any]:
    """Cheap liveness/progress probe for one run's token store.

    Stats the run's ``manifest-*.jsonl`` files and reads their last lines;
    never loads segment data. Returns::

        {
            "live": bool,               # any manifest modified within live_window_s
            "last_activity_ts": float | None,  # newest manifest mtime (epoch s)
            "n_segments": int,          # total manifest lines across writers
            "latest_iteration": int | None,    # max last-line max_iteration >= 0
        }
    """
    status: dict[str, Any] = {
        "live": False,
        "last_activity_ts": None,
        "n_segments": 0,
        "latest_iteration": None,
    }
    try:
        storage = storage_from_uri(str(log_path))
        names = [
            name
            for name in storage.list_dir(TOKENS_DIR)
            if name.startswith("manifest-") and name.endswith(".jsonl")
        ]
    except Exception:
        return status
    last_mtime: float | None = None
    n_segments = 0
    latest_iteration: int | None = None
    for name in names:
        path = f"{TOKENS_DIR}/{name}"
        stat = storage.stat(path)
        if stat is not None:
            last_mtime = stat.mtime if last_mtime is None else max(last_mtime, stat.mtime)
        try:
            lines = [line for line in storage.read(path).decode().splitlines() if line.strip()]
        except FileNotFoundError:
            continue
        n_segments += len(lines)
        if not lines:
            continue
        try:
            entry = json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        max_iteration = entry.get("max_iteration")
        if isinstance(max_iteration, int) and max_iteration >= 0:
            latest_iteration = (
                max_iteration if latest_iteration is None else max(latest_iteration, max_iteration)
            )
    status["last_activity_ts"] = last_mtime
    status["n_segments"] = n_segments
    status["latest_iteration"] = latest_iteration
    status["live"] = last_mtime is not None and (time.time() - last_mtime) <= live_window_s
    return status


def list_runs(
    registry_dir: str | None = None, *, live_window_s: float = DEFAULT_LIVE_WINDOW_S
) -> list[dict[str, Any]]:
    """List all registered runs, newest first, each with a ``status`` probe.

    Returns registry records (see :func:`register_run`) with a ``"status"``
    key holding :func:`run_status` output. Unreadable records are skipped
    with a warning.
    """
    directory = resolve_registry_dir(registry_dir)
    if directory is None or not directory.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable registry record %s", path)
            continue
        if not isinstance(record, dict) or "run_id" not in record:
            logger.warning("Skipping malformed registry record %s", path)
            continue
        record["status"] = run_status(str(record.get("log_path", "")), live_window_s=live_window_s)
        runs.append(record)
    runs.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
    return runs
