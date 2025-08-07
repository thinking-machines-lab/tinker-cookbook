"""
Small utilities requiring only basic python libraries.
"""

import importlib
import time
from contextlib import contextmanager
from typing import Any

import numpy as np


@contextmanager
def timed(key: str, metrics: dict[str, float]):
    print(f"Starting {key}")
    tstart = time.time()
    yield
    print(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


def safezip(*args, **kwargs):  # type: ignore
    return zip(*args, **kwargs, strict=True)


def dict_mean(list_of_dicts: list[dict[str, float | int]]) -> dict[str, float]:
    assert all_same([d.keys() for d in list_of_dicts])
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in list_of_dicts[0]}


def all_same(xs: list[Any]) -> bool:
    return all(x == xs[0] for x in xs)


def lookup_func(path_to_func: str, default_module: str | None = None):
    """
    path.to.module:func_name or func_name (assumes default_module)
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


def concat_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]
