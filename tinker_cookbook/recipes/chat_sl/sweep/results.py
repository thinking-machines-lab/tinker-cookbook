"""Result collection for completed sweep runs."""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from tinker_cookbook.stores import TrainingRunStore, storage_from_uri

logger = logging.getLogger(__name__)


def _read_final_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Read the last metrics.jsonl record (final training metrics) for a run."""
    store = TrainingRunStore(storage_from_uri(str(run_dir), mkdir=False))
    records = store.read_metrics()
    return records[-1] if records else {}


def _read_config(run_dir: str | Path) -> dict[str, Any]:
    """Read config.json from a run directory."""
    store = TrainingRunStore(storage_from_uri(str(run_dir), mkdir=False))
    return store.read_config() or {}


def _extract_config_value(config: dict[str, Any], key: str) -> Any:
    """Extract a value from a config dict, supporting dot-separated paths.

    Example: ``_extract_config_value(config, "dataset_builder.common_config.batch_size")``
    """
    parts = key.split(".")
    current: Any = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def collect(
    sweep_dir: str | Path,
    *,
    config_keys: list[str] | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Collect results from all completed runs in a sweep directory.

    Scans subdirectories for ``metrics.jsonl`` and ``config.json``. Returns a
    DataFrame with one row per completed run.

    Args:
        sweep_dir: Directory containing sweep run subdirectories.
        config_keys: Config fields to include as columns. Supports dot-separated
            paths for nested configs (e.g. ``"dataset_builder.common_config.batch_size"``).
            If ``None``, includes all top-level scalar config fields.
        require_complete: If True, skip runs where ``progress < 0.98``.

    Returns:
        pandas DataFrame with config columns, metric columns, and ``log_path``.
    """
    sweep_uri = str(sweep_dir)
    sweep_storage = storage_from_uri(sweep_uri, mkdir=False)
    if not sweep_storage.exists_tree(""):
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")

    rows: list[dict[str, Any]] = []
    for run_name in sweep_storage.list_dir(""):
        # Join directly (not via url(), which percent-encodes run names like "lr=1e-4").
        run_path = os.path.join(sweep_uri, run_name)
        # Skip non-run children (e.g. stray files): a run must have metrics.jsonl.
        if not storage_from_uri(run_path, mkdir=False).exists("metrics.jsonl"):
            continue

        metrics = _read_final_metrics(run_path)
        if not metrics:
            logger.warning(f"Skipping {run_name}: no metrics.jsonl or empty")
            continue

        if require_complete:
            progress = metrics.get("progress", 0)
            if isinstance(progress, (int, float)) and progress < 0.98:
                logger.warning(f"Skipping {run_name}: incomplete (progress={progress:.2f})")
                continue

        config = _read_config(run_path)

        row: dict[str, Any] = {"log_path": run_path}

        # Extract config values
        if config_keys is not None:
            for key in config_keys:
                row[key.split(".")[-1]] = _extract_config_value(config, key)
        elif config:
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    row[key] = value

        # Add all scalar metrics (don't overwrite config values)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in row:
                row[key] = value

        rows.append(row)

    if not rows:
        logger.warning(f"No completed runs found in {sweep_dir}")
        return pd.DataFrame()

    return pd.DataFrame(rows)
