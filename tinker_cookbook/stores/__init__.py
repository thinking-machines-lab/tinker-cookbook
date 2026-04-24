"""Unified data access layer for tinker-cookbook.

Provides Storage protocols for byte-level I/O (local, S3, GCS),
typed stores for training and eval data, and a registry for
discovering runs across multiple backends.

On-Disk Layout
--------------

Training runs and eval data follow this structure::

    {root}/
    ├── {run_id}/                        # One per training run
    │   ├── config.json                  # Training config (chz-serialized)
    │   ├── metrics.jsonl                # Per-step metrics (append-only)
    │   ├── checkpoints.jsonl            # Checkpoint records (append-only)
    │   ├── timing_spans.jsonl           # Timing spans (append-only)
    │   ├── code.diff                    # Code diff at training start
    │   ├── chat_sessions/               # Interactive chat sessions
    │   │   └── {session_id}.json        # ChatSession (Pydantic model)
    │   └── iteration_{NNNNNN}/          # Per-iteration artifacts
    │       ├── train_rollout_summaries.jsonl   # One record per trajectory
    │       ├── train_logtree.json       # HTML visualization data
    │       ├── eval_{label}_rollout_summaries.jsonl
    │       └── eval_{label}_logtree.json
    └── eval/                            # Evaluation data (optional)
        ├── runs.jsonl                   # Eval run index (append-only)
        └── runs/
            └── {eval_run_id}/
                ├── metadata.json        # RunMetadata
                └── {benchmark}/
                    ├── result.json      # BenchmarkResult
                    └── trajectories.jsonl  # StoredTrajectory per line

File Formats
~~~~~~~~~~~~

- **metrics.jsonl**: ``{"step": int, "env/all/reward/total": float, ...}``
- **checkpoints.jsonl**: ``{"name": str, "batch": int, "state_path": str, ...}``
- **timing_spans.jsonl**: ``{"step": int, "spans": [{"name": str, "duration": float, ...}]}``
- **rollout_summaries.jsonl**: See :mod:`tinker_cookbook.rl.rollout_logging` for full schema.

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
