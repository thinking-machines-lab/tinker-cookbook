"""RunRegistry — discovers training runs and routes to stores."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from tinker_cookbook.stores.storage import Storage, storage_join
from tinker_cookbook.stores.training_store import TrainingRunStore

_ITERATION_RE = re.compile(r"^iteration_(\d+)$")

Status = Literal["running", "completed", "idle"]
TrainingType = Literal["rl", "sl", "dpo"]
_ACTIVE_THRESHOLD_SECONDS = 120


class _EvalUnset:
    """Sentinel for eval cache — survives pickle (unlike bare object())."""

    _instance: _EvalUnset | None = None

    def __new__(cls) -> _EvalUnset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_DEFAULT_EVAL_PREFIXES: tuple[str, ...] = ("eval", "eval_store", "")


@dataclass(frozen=True)
class RunInfo:
    """Metadata about a discovered training run.

    Created by :meth:`RunRegistry.refresh` during directory scanning.
    Serialized to JSON by the ``/api/runs`` endpoint.
    """

    run_id: str
    """Directory name used as the run identifier."""
    prefix: str
    """Storage path prefix (e.g., local path or ``s3://bucket/path``)."""
    has_config: bool
    """Whether ``config.json`` exists in the run directory."""
    has_metrics: bool
    """Whether ``metrics.jsonl`` exists."""
    has_checkpoints: bool
    """Whether ``checkpoints.jsonl`` exists."""
    has_timing: bool
    """Whether ``timing_spans.jsonl`` exists."""
    iteration_count: int
    """Number of ``iteration_NNNNNN/`` directories found."""
    status: Status
    """``'running'`` if files modified within 2 minutes, ``'completed'`` if
    a final checkpoint exists, ``'idle'`` otherwise."""
    last_updated: float | None
    """Unix timestamp of the most recently modified file, or ``None``."""
    training_type: TrainingType | None
    """``'rl'`` if config has ``loss_fn``, ``'dpo'`` if has ``dpo_beta``,
    ``'sl'`` otherwise. ``None`` if no config."""


class RunRegistry:
    """Discovers training runs and provides stores for each.

    Supports multiple storage backends. Each run is tracked with its
    source storage so stores route to the correct backend.
    """

    _EVAL_UNSET = _EvalUnset()

    def __init__(
        self,
        storages: list[Storage],
        eval_prefixes: tuple[str, ...] = _DEFAULT_EVAL_PREFIXES,
    ) -> None:
        self._storages = storages
        self._eval_prefixes = eval_prefixes
        self._runs: dict[str, RunInfo] | None = None
        self._run_storage: dict[str, Storage] = {}
        self._training_stores: dict[str, TrainingRunStore] = {}
        self._eval_store: Any = RunRegistry._EVAL_UNSET

    @property
    def storage_count(self) -> int:
        """Number of storage backends registered."""
        return len(self._storages)

    @property
    def primary_storage(self) -> Storage:
        """The first (default) storage backend."""
        return self._storages[0]

    def storage_for(self, run_id: str) -> Storage:
        """Return the storage backend that owns a run (falls back to primary)."""
        return self._run_storage.get(run_id) or self._storages[0]

    def refresh(self) -> list[RunInfo]:
        """Re-scan all storages. Clears cached stores."""
        all_runs: dict[str, RunInfo] = {}
        self._run_storage.clear()
        self._training_stores.clear()
        self._eval_store = RunRegistry._EVAL_UNSET

        for storage in self._storages:
            source = storage.url().rstrip("/").rsplit("/", 1)[-1]

            for run in _discover_runs(storage):
                uid = f"{source}--{run.run_id}" if run.run_id in all_runs else run.run_id
                all_runs[uid] = run
                self._run_storage[uid] = storage

        self._runs = all_runs
        return list(all_runs.values())

    def get_runs(self) -> list[RunInfo]:
        """Return all discovered runs (lazy — calls :meth:`refresh` on first access)."""
        if self._runs is None:
            self.refresh()
        assert self._runs is not None
        return list(self._runs.values())

    def get_run(self, run_id: str) -> RunInfo | None:
        """Look up a single run by ID, or ``None`` if not found."""
        if self._runs is None:
            self.refresh()
        assert self._runs is not None
        return self._runs.get(run_id)

    def get_training_store(self, run_id: str) -> TrainingRunStore:
        """Get a TrainingRunStore for a specific run."""
        if run_id not in self._training_stores:
            run = self.get_run(run_id)
            storage = self.storage_for(run_id)
            prefix = run.prefix if run else ""
            self._training_stores[run_id] = TrainingRunStore(storage, prefix)
        return self._training_stores[run_id]

    def get_eval_store(self) -> Any | None:
        """Get the EvalStore if eval data exists (cached after first call).

        Returns None if no eval data found. Import is deferred to avoid
        circular deps with eval/__init__.py. Probes ``eval_prefixes``
        (configurable via constructor) for a ``runs.jsonl`` file.
        """
        if self._eval_store is not RunRegistry._EVAL_UNSET:
            return self._eval_store
        for storage in self._storages:
            for prefix in self._eval_prefixes:
                path = storage_join(prefix, "runs.jsonl")
                if storage.exists(path):
                    try:
                        from tinker_cookbook.stores.eval_store import EvalStore

                        self._eval_store = EvalStore(storage, prefix)
                        return self._eval_store
                    except ImportError:
                        break
        self._eval_store = None
        return None


def _discover_runs(storage: Storage, root_prefix: str = "") -> list[RunInfo]:
    """Scan storage for directories containing metrics.jsonl or config.json.

    Checks the root prefix and one level of children. Does NOT recurse
    deeper — this is intentional to avoid expensive ``list_dir`` calls on
    cloud backends. Users needing deeper nesting should pass multiple
    storages or flatten their directory structure.
    """
    runs: list[RunInfo] = []

    if _is_run_dir(storage, root_prefix):
        name = root_prefix.rstrip("/").rsplit("/", 1)[-1] if root_prefix else "root"
        runs.append(_build_run_info(storage, name, root_prefix))
        return runs

    for child in sorted(storage.list_dir(root_prefix)):
        child_prefix = storage_join(root_prefix, child) if root_prefix else child
        if _is_run_dir(storage, child_prefix):
            runs.append(_build_run_info(storage, child, child_prefix))

    return runs


def _is_run_dir(storage: Storage, prefix: str) -> bool:
    return storage.exists(storage_join(prefix, "metrics.jsonl")) or storage.exists(
        storage_join(prefix, "config.json")
    )


def _detect_status(store: TrainingRunStore) -> tuple[Status, float | None]:
    """Infer run status from metrics mtime and checkpoint records."""
    stat = store.storage.stat(storage_join(store.prefix, "metrics.jsonl"))
    if stat is None:
        return "idle", None
    age = time.time() - stat.mtime
    if age < _ACTIVE_THRESHOLD_SECONDS:
        return "running", stat.mtime
    for ckpt in reversed(store.read_checkpoints()):
        if ckpt.get("final"):
            return "completed", stat.mtime
    return "idle", stat.mtime


def _infer_training_type(store: TrainingRunStore) -> TrainingType | None:
    """Infer training type from config keys."""
    config = store.read_config()
    if config is None:
        return None
    if "dpo_beta" in config:
        return "dpo"
    if "loss_fn" in config:
        return "rl"
    if "num_epochs" in config:
        return "sl"
    db = config.get("dataset_builder")
    if isinstance(db, dict):
        dt = db.get("__type__", "")
        if "RL" in dt:
            return "rl"
        if "Supervised" in dt or "SL" in dt:
            return "sl"
    return None


def _build_run_info(storage: Storage, run_id: str, prefix: str) -> RunInfo:
    store = TrainingRunStore(storage, prefix)
    status, last_updated = _detect_status(store)
    training_type = _infer_training_type(store)
    iteration_count = sum(1 for child in storage.list_dir(prefix) if _ITERATION_RE.match(child))

    return RunInfo(
        run_id=run_id,
        prefix=prefix,
        has_config=storage.exists(storage_join(prefix, "config.json")),
        has_metrics=storage.exists(storage_join(prefix, "metrics.jsonl")),
        has_checkpoints=storage.exists(storage_join(prefix, "checkpoints.jsonl")),
        has_timing=storage.exists(storage_join(prefix, "timing_spans.jsonl")),
        iteration_count=iteration_count,
        status=status,
        last_updated=last_updated,
        training_type=training_type,
    )
