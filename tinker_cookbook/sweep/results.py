"""Result collection for completed sweep runs."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _read_final_metrics(run_dir: Path) -> dict[str, Any]:
    """Read the last line of metrics.jsonl to get final training metrics."""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return {}
    last_line = None
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return {}
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse last line of {metrics_file}")
        return {}


def _read_config(run_dir: Path) -> dict[str, Any]:
    """Read config.json from a run directory."""
    config_file = run_dir / "config.json"
    if not config_file.exists():
        return {}
    try:
        with open(config_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse {config_file}")
        return {}


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
    sweep_path = Path(sweep_dir)
    if not sweep_path.is_dir():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(sweep_path.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics = _read_final_metrics(run_dir)
        if not metrics:
            logger.warning(f"Skipping {run_dir.name}: no metrics.jsonl or empty")
            continue

        if require_complete:
            progress = metrics.get("progress", 0)
            if isinstance(progress, (int, float)) and progress < 0.98:
                logger.warning(f"Skipping {run_dir.name}: incomplete (progress={progress:.2f})")
                continue

        config = _read_config(run_dir)

        row: dict[str, Any] = {"log_path": str(run_dir)}

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
