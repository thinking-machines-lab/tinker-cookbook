"""Utility helpers for tinker-cookbook.

Provides logging, tracing, and general-purpose utilities used across
training loops and evaluation code.

Submodules:
    - ``ml_log``: Metrics logging (JSON, W&B, console)
    - ``logtree``: Structured HTML report generation
    - ``trace``: Performance tracing and Gantt chart visualization
    - ``misc_utils``: Small helpers (safezip, timed, dict_mean, etc.)
    - ``lr_scheduling``: Learning rate schedule utilities
"""

from tinker_cookbook.utils import logtree, ml_log, trace
from tinker_cookbook.utils.misc_utils import (
    all_same,
    concat_lists,
    dict_mean,
    not_none,
    safezip,
    split_list,
    timed,
)

__all__ = [
    # Submodules (commonly imported as `from tinker_cookbook.utils import ml_log`)
    "ml_log",
    "logtree",
    "trace",
    # Misc utilities
    "safezip",
    "timed",
    "dict_mean",
    "all_same",
    "split_list",
    "concat_lists",
    "not_none",
]
