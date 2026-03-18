"""Sweep runner — execute experiments across a parameter grid."""

import logging
import os
import typing
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from datetime import datetime
from typing import Any, TypeVar

import chz
import pandas as pd

from tinker_cookbook.sweep.grid import default_run_name
from tinker_cookbook.sweep.grid import grid as make_grid
from tinker_cookbook.sweep.results import collect

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_SWEEP_ROOT = "/tmp/tinker-sweeps"


def _validate_axes(config_type: type, sweep_axes: dict[str, list[Any]]) -> None:
    """Validate that all sweep axis names are fields on the config type."""
    try:
        hints = typing.get_type_hints(config_type)
    except Exception:
        # If we can't get type hints, skip validation
        return

    for axis_name in sweep_axes:
        if axis_name not in hints:
            available = sorted(hints.keys())
            raise TypeError(
                f"Sweep axis '{axis_name}' is not a field on {config_type.__name__}. "
                f"Available fields: {available}"
            )


def _validate_config_has_log_path(config_type: type) -> None:
    """Validate that the config type has a log_path field."""
    try:
        hints = typing.get_type_hints(config_type)
    except Exception:
        return
    if "log_path" not in hints:
        raise TypeError(
            f"{config_type.__name__} does not have a 'log_path' field. "
            f"sweep.run needs to set log_path for each run automatically."
        )


def _make_sweep_dir() -> str:
    """Generate a default sweep directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_DEFAULT_SWEEP_ROOT, timestamp)


def _run_single(
    main_fn: Callable[[Any], None],
    base_config: Any,
    overrides: dict[str, Any],
    log_path: str,
) -> None:
    """Run a single experiment with the given overrides."""
    config = chz.replace(base_config, log_path=log_path, **overrides)
    main_fn(config)


def run(
    main_fn: Callable[[T], None],
    base_config: T,
    *,
    sweep_dir: str | None = None,
    max_parallel: int = 1,
    executor: Executor | None = None,
    name_fn: Callable[[dict[str, Any]], str] | None = None,
    skip_existing: bool = False,
    **sweep_axes: list[Any],
) -> pd.DataFrame:
    """Run experiments across a parameter grid and collect results.

    Wraps any ``@chz.chz`` config with sweep axes. For each combination,
    creates a modified config via ``chz.replace`` and calls ``main_fn``.

    Example::

        from tinker_cookbook import sweep
        from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main

        results = sweep.run(
            cli_main,
            CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
            learning_rate=[1e-4, 3e-4, 1e-3],
            lora_rank=[32, 128],
        )

        # Parallel execution:
        results = sweep.run(cli_main, base, max_parallel=4,
                            learning_rate=[1e-4, 3e-4])

        # Custom executor (e.g. Ray):
        results = sweep.run(cli_main, base, executor=ray_pool,
                            learning_rate=[1e-4, 3e-4])

    Args:
        main_fn: Recipe entry function that accepts a single config argument.
            Must be importable (not defined inline in ``__main__``) when using
            ``max_parallel > 1`` or ``executor``.
        base_config: A ``@chz.chz`` config with defaults for non-swept params.
            Must have a ``log_path`` field.
        sweep_dir: Directory for run outputs. Default: ``/tmp/tinker-sweeps/{timestamp}/``.
        max_parallel: Number of parallel workers. 1 = sequential (default),
            >1 = ``ProcessPoolExecutor`` managed internally. Ignored if
            ``executor`` is provided.
        executor: Custom ``concurrent.futures.Executor`` for parallel runs
            (e.g. Ray Pool). Caller is responsible for executor lifecycle.
            Takes precedence over ``max_parallel``.
        name_fn: Maps overrides dict to a subdirectory name. Default generates
            names like ``learning_rate=0.0003_lora_rank=32``.
        skip_existing: If True, skip runs whose log_path already has metrics.jsonl.
        **sweep_axes: Parameter names mapped to lists of values. Each name must
            be a field on ``base_config``.

    Returns:
        pandas DataFrame with one row per completed run (config + metric columns).
    """
    if not sweep_axes:
        raise ValueError("At least one sweep axis must be provided as a keyword argument.")

    # Validate config and axes
    _validate_config_has_log_path(type(base_config))
    _validate_axes(type(base_config), sweep_axes)

    # Validate axis values are lists
    for axis_name, values in sweep_axes.items():
        if not isinstance(values, list):
            raise TypeError(
                f"Sweep axis '{axis_name}' must be a list of values, got {type(values).__name__}. "
                f"Did you mean {axis_name}=[{values!r}]?"
            )

    # Generate grid
    points = make_grid(sweep_axes)
    namer = name_fn or default_run_name

    # Set up sweep directory
    if sweep_dir is None:
        sweep_dir = _make_sweep_dir()
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Sweep: {len(points)} configurations → {sweep_dir}")

    # Run experiments
    if executor is not None:
        _run_parallel(main_fn, base_config, points, sweep_dir, skip_existing, namer, executor)
    elif max_parallel > 1:
        with ProcessPoolExecutor(max_workers=max_parallel) as pool:
            _run_parallel(main_fn, base_config, points, sweep_dir, skip_existing, namer, pool)
    else:
        _run_sequential(main_fn, base_config, points, sweep_dir, skip_existing, namer)

    # Collect and return results
    return collect(sweep_dir, require_complete=False)


def _run_sequential(
    main_fn: Callable[[Any], None],
    base_config: Any,
    points: list[dict[str, Any]],
    sweep_dir: str,
    skip_existing: bool,
    namer: Callable[[dict[str, Any]], str],
) -> None:
    """Run experiments sequentially."""
    for i, overrides in enumerate(points):
        run_name = namer(overrides)
        log_path = os.path.join(sweep_dir, run_name)

        if skip_existing and os.path.exists(os.path.join(log_path, "metrics.jsonl")):
            print(f"  [{i + 1}/{len(points)}] Skipping {run_name} (already exists)")
            continue

        print(f"  [{i + 1}/{len(points)}] {run_name}")
        try:
            _run_single(main_fn, base_config, overrides, log_path)
        except Exception:
            logger.exception(f"Run failed: {run_name}")


def _run_parallel(
    main_fn: Callable[[Any], None],
    base_config: Any,
    points: list[dict[str, Any]],
    sweep_dir: str,
    skip_existing: bool,
    namer: Callable[[dict[str, Any]], str],
    executor: Executor,
) -> None:
    """Run experiments in parallel using the provided executor."""
    futures: list[tuple[str, Future[None]]] = []

    for overrides in points:
        run_name = namer(overrides)
        log_path = os.path.join(sweep_dir, run_name)

        if skip_existing and os.path.exists(os.path.join(log_path, "metrics.jsonl")):
            print(f"  Skipping {run_name} (already exists)")
            continue

        future = executor.submit(_run_single, main_fn, base_config, overrides, log_path)
        futures.append((run_name, future))

    print(f"  Submitted {len(futures)} jobs to executor")

    # Wait for all futures to complete
    for run_name, future in futures:
        try:
            future.result()
        except Exception:
            logger.exception(f"Run failed: {run_name}")
