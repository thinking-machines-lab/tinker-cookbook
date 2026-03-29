"""
Small utilities requiring only basic python libraries.
"""

import importlib
import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """Context manager that measures wall-clock time and stores it in a metrics dict.

    Logs the start and duration of the timed block at INFO level, and writes
    the elapsed seconds to ``metrics[f"time/{key}"]``.

    Args:
        key (str): Label for the timed section. Used in log messages and as
            the suffix in the metrics key ``time/{key}``.
        metrics (dict[str, Any]): Mutable dictionary where the elapsed time
            (in seconds) will be stored.

    Example::

        metrics: dict[str, Any] = {}
        with timed("forward_pass", metrics):
            output = model(inputs)
        print(metrics["time/forward_pass"])  # e.g. 1.23
    """
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))
"""A strict version of ``zip`` that raises ``ValueError`` when iterables have different lengths.

Example::

    safezip([1, 2], ["a", "b"])  # OK
    safezip([1, 2], ["a"])       # raises ValueError
"""


def dict_mean(list_of_dicts: list[dict[str, float | int]]) -> dict[str, float]:
    """Compute the element-wise mean across a list of dictionaries.

    Keys that appear in only some dictionaries are averaged over only those
    entries (i.e., missing keys are not treated as zero).

    Args:
        list_of_dicts (list[dict[str, float | int]]): List of metric
            dictionaries to average.

    Returns:
        dict[str, float]: Dictionary mapping each key to its mean value.
    """
    key2values = {}
    for d in list_of_dicts:
        for k, v in d.items():
            key2values.setdefault(k, []).append(v)
    return {k: float(np.mean(values)) for k, values in key2values.items()}


def all_same(xs: list[Any]) -> bool:
    """Check whether all elements in a list are equal.

    Args:
        xs (list[Any]): List of elements to compare.

    Returns:
        bool: ``True`` if every element equals the first, or if the list is empty.
    """
    return all(x == xs[0] for x in xs)


def lookup_func(path_to_func: str, default_module: str | None = None):
    """Import and return a callable by its dotted module path.

    Accepts either ``"path.to.module:func_name"`` or a bare ``"func_name"``
    (in which case *default_module* is used as the module).

    Args:
        path_to_func (str): Either ``"module.path:name"`` or a bare name.
        default_module (str | None): Module to import from when no colon is
            present.  Required if *path_to_func* has no colon.

    Returns:
        Any: The attribute retrieved from the module.

    Raises:
        ValueError: If *path_to_func* contains more than one colon or has
            no colon and *default_module* is ``None``.
    """
    colon_count = path_to_func.count(":")
    if colon_count == 0 and default_module is not None:
        module_name = default_module
        func_name = path_to_func
    elif colon_count == 1:
        module_name, func_name = path_to_func.rsplit(":", 1)
    else:
        raise ValueError(f"Invalid path: {path_to_func}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    """Split a sequence into approximately equal-sized sublists.

    The sizes of the resulting sublists differ by at most 1, and longer
    and shorter sublists are distributed as uniformly as possible.

    Args:
        lst (Sequence[T]): The sequence to split.
        num_splits (int): Number of sublists to create.

    Returns:
        list[list[T]]: A list of sublists with sizes differing by at most 1.

    Raises:
        ValueError: If *num_splits* > ``len(lst)`` or *num_splits* <= 0.

    Example::

        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2, 3], [4, 5]]
        >>> split_list([1, 2, 3, 4, 5], 3)
        [[1, 2], [3, 4], [5]]
    """
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_splits > len(lst):
        raise ValueError(f"Cannot split list of length {len(lst)} into {num_splits} parts")

    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]


def concat_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    """Flatten a list of lists into a single list.

    Args:
        list_of_lists (list[list[Any]]): Nested list to flatten (one level).

    Returns:
        list[Any]: Concatenated elements from all sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]


def not_none(x: T | None) -> T:
    """Assert that a value is not ``None`` and return it with a narrowed type.

    Args:
        x (T | None): Value to check.

    Returns:
        T: The same value, guaranteed non-``None``.

    Raises:
        AssertionError: If *x* is ``None``.
    """
    assert x is not None, f"{x=} must not be None"
    return x


def iteration_dir(log_path: str | Path | None, step: int) -> Path | None:
    """Return the per-iteration subdirectory path for one training step.

    Output files for each training step are grouped under
    ``log_path/iteration_NNNNNN/`` to keep the top-level directory manageable.

    Args:
        log_path (str | Path | None): Root log directory, or ``None`` to
            disable iteration directories.
        step (int): Training step number (zero-padded to 6 digits in the
            directory name).

    Returns:
        Path | None: The iteration subdirectory path, or ``None`` when
            *log_path* is ``None`` or empty.
    """
    if not log_path:
        return None
    return Path(log_path) / f"iteration_{step:06d}"
