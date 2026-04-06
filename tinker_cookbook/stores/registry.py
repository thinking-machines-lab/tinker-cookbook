"""RunRegistry — discovers training runs and routes to stores."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tinker_cookbook.stores.storage import Storage, storage_join
from tinker_cookbook.stores.training_store import (
    Status,
    TrainingRunStore,
    TrainingType,
)

_ITERATION_RE = re.compile(r"^iteration_(\d+)$")


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
    """Metadata about a discovered training run."""

    run_id: str
    prefix: str
    has_config: bool
    has_metrics: bool
    has_checkpoints: bool
    has_timing: bool
    iteration_count: int
    status: Status
    last_updated: float | None
    training_type: TrainingType | None


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
        return len(self._storages)

    @property
    def primary_storage(self) -> Storage:
        return self._storages[0]

    def storage_for(self, run_id: str) -> Storage:
        return self._run_storage.get(run_id) or self._storages[0]

    def refresh(self) -> list[RunInfo]:
        """Re-scan all storages. Clears cached stores."""
        all_runs: dict[str, RunInfo] = {}
        self._run_storage.clear()
        self._training_stores.clear()
        self._eval_store = RunRegistry._EVAL_UNSET

        for storage in self._storages:
            source = ""
            if hasattr(storage, "root"):
                source = getattr(storage, "root").name  # noqa: B009

            for run in _discover_runs(storage):
                uid = f"{source}--{run.run_id}" if run.run_id in all_runs else run.run_id
                all_runs[uid] = run
                self._run_storage[uid] = storage

        self._runs = all_runs
        return list(all_runs.values())

    def get_runs(self) -> list[RunInfo]:
        if self._runs is None:
            self.refresh()
        assert self._runs is not None
        return list(self._runs.values())

    def get_run(self, run_id: str) -> RunInfo | None:
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


def _build_run_info(storage: Storage, run_id: str, prefix: str) -> RunInfo:
    store = TrainingRunStore(storage, prefix)
    status, last_updated = store.detect_status()
    training_type = store.infer_training_type()
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
