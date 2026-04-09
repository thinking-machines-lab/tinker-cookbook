"""Unified data access layer for tinker-cookbook.

Provides Storage protocols for byte-level I/O (local, S3, GCS),
typed stores for training and eval data, and a registry for
discovering runs across multiple backends.

Note: ``EvalStore`` is not imported here to avoid circular imports
with ``tinker_cookbook.eval``. Import it directly::

    from tinker_cookbook.stores.eval_store import EvalStore
"""

from tinker_cookbook.stores._base import BaseStore
from tinker_cookbook.stores._incremental import IncrementalReader
from tinker_cookbook.stores.registry import RunInfo, RunRegistry
from tinker_cookbook.stores.storage import (
    AsyncStorage,
    FsspecStorage,
    LocalStorage,
    Storage,
    StorageStat,
    storage_from_uri,
    storage_join,
)
from tinker_cookbook.stores.training_store import TrainingRunStore

__all__ = [
    "AsyncStorage",
    "BaseStore",
    "FsspecStorage",
    "IncrementalReader",
    "LocalStorage",
    "RunInfo",
    "RunRegistry",
    "Storage",
    "StorageStat",
    "TrainingRunStore",
    "storage_from_uri",
    "storage_join",
]
