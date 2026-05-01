"""Singleton timer for implicit per-step timing collection.

Training code wraps operations with ``timer("name")`` -- no metrics dict
needed.  At step end, ``flush_timing()`` returns ``{"perf/X_time": Y, ...}``
and resets the singleton for the next step.

Usage::

    with timer("ref_forward"):
        reference.forward(...)
    with timer("fwd_bwd"):
        policy.forward_backward_custom(...)

    perf = flush_timing()
    # perf == {"perf/ref_forward_time": 1.2, "perf/fwd_bwd_time": 3.4}

The ``timer`` function doubles as a decorator::

    @timer
    def save_model(...):
        ...
"""

from __future__ import annotations

import time
import logging
from copy import deepcopy
from typing import Any
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """Global singleton that accumulates wall-clock durations by name."""

    _instance: Timer | None = None

    def __new__(cls) -> Timer:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst.timers: dict[str, float] = {}
            inst._start_time: dict[str, float] = {}
            cls._instance = inst
        return cls._instance

    def start(self, name: str) -> None:
        if name in self._start_time:
            logger.warning("Timer '%s' already running -- restarting", name)
        self._start_time[name] = time.time()

    def end(self, name: str) -> None:
        if name not in self._start_time:
            logger.warning("Timer '%s' was never started -- ignoring end", name)
            return
        elapsed = time.time() - self._start_time.pop(name)
        self.timers[name] = self.timers.get(name, 0.0) + elapsed

    def add(self, name: str, elapsed: float) -> None:
        self.timers[name] = self.timers.get(name, 0.0) + elapsed

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self.timers.clear()
        else:
            self.timers.pop(name, None)

    def log_dict(self) -> dict[str, float]:
        return dict(self.timers)

    @contextmanager
    def context(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


def timer(name_or_func):
    """Context manager or decorator -- writes to the global ``Timer`` singleton.

    As a context manager::

        with timer("fwd_bwd"):
            ...

    As a decorator::

        @timer
        def save_model(...):
            ...
    """
    if isinstance(name_or_func, str):
        return Timer().context(name_or_func)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def inverse_timer(name: str):
    """Measure the gap *between* operations (e.g. wait time).

    Ends the named timer on entry, re-starts it on exit.  Pair with
    ``Timer().start("train_wait")`` at init time so the wait clock runs
    whenever the training code is *not* inside an ``inverse_timer`` block.
    """
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)


def flush_timing() -> dict[str, Any]:
    """Return ``{"perf/{name}_time": seconds, ...}`` and reset the singleton."""
    t = Timer()
    raw = deepcopy(t.log_dict())
    t.reset()
    return {f"perf/{k}_time": v for k, v in raw.items()}


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """Backward-compatible wrapper: writes to *metrics* **and** the singleton.

    Existing DPO/SFT code that still passes a dict keeps working, while the
    singleton also records the duration so ``flush_timing()`` picks it up.
    """
    t = Timer()
    t.start(key)
    try:
        yield
    finally:
        t.end(key)
    metrics[f"perf/{key}_time"] = t.timers.get(key, 0.0)
