"""
Profiler for training loop functions.

Provides :class:`Profiler` and :func:`profiled` for automatically recording
wall-clock time and CPU time of decorated functions. The profiler produces
a separate dict of profiling metrics that the user merges into their training
metrics explicitly.

Usage::

    from tinker_cookbook.utils.profiling import Profiler, profiled

    prof = Profiler()

    @profiled("rollout", agg=["mean", "max"])
    async def do_rollout(...):
        ...

    @profiled("train")
    async def train_step(...):
        ...

    for i_batch in range(n_batches):
        metrics = {"progress/batch": i_batch}

        with prof.measure() as profile:
            await do_rollout(...)
            await train_step(...)

        metrics.update(profile)
        ml_logger.log_metrics(metrics, step=i_batch)

The profiling dict (``profile`` above) contains only profiler-produced keys:
    - ``time/{key}`` — wall-clock duration (seconds)
    - ``cpu/{key}`` — CPU time (seconds, excludes I/O wait and sleep)
    - ``time/total`` — wall-clock duration of the entire ``prof.measure()`` block
    - ``cpu/total`` — CPU time of the entire ``prof.measure()`` block

The difference between ``time/`` and ``cpu/`` reveals how much time is spent
waiting on I/O (e.g., Tinker API calls) vs doing local compute.

Concurrency safety:
    - Single async task: safe (cooperative scheduling, no concurrent writes).
    - asyncio.gather() of bare coroutines: safe (shared context, cooperative).
    - asyncio.create_task() children: safe (inherit same accumulator, single-
      threaded event loop). The parent must await children before exiting
      prof.measure(), otherwise the children's data is lost.
    - threading.Thread / ThreadPoolExecutor: isolated by design. Child threads
      do NOT inherit the parent's ContextVar state, so @profiled silently
      logs to Python logging only (no metrics recorded). This is intentional.
    - asyncio.to_thread(): copies ContextVar context to the thread, so the
      background thread shares the same accumulator as the caller. A per-scope
      threading.Lock protects all writes and the final flattening, making this
      safe even under free-threaded Python (no-GIL, PEP 703).
    - Multi-process: no shared memory, not applicable.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, overload

logger = logging.getLogger(__name__)

__all__ = [
    "Profiler",
    "profiled",
    "AggFn",
    "AggSpec",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AggFn = Callable[[list[float]], float]
"""A function that reduces a list of durations to a single float."""

AggSpec = str | tuple[str, AggFn]
"""Aggregation spec: a built-in name ("mean", "max", "min") or a (name, func) tuple."""

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Built-in aggregations
# ---------------------------------------------------------------------------

_BUILTIN_AGGS: dict[str, AggFn] = {
    "mean": lambda d: sum(d) / len(d),
    "max": lambda d: max(d),
    "min": lambda d: min(d),
}


def _validate_agg_specs(agg: list[AggSpec]) -> None:
    """Validate aggregation specs eagerly at decoration time."""
    for spec in agg:
        if isinstance(spec, str):
            if spec not in _BUILTIN_AGGS:
                raise ValueError(
                    f"Unknown built-in aggregation '{spec}'. "
                    f"Available: {list(_BUILTIN_AGGS.keys())}"
                )
        elif isinstance(spec, tuple):
            if len(spec) != 2 or not isinstance(spec[0], str) or not callable(spec[1]):
                raise TypeError(
                    f"Custom aggregation must be a (name, callable) tuple, got {spec!r}"
                )
        else:
            raise TypeError(
                f"Aggregation spec must be a string or (name, callable) tuple, "
                f"got {type(spec).__name__}"
            )


# ---------------------------------------------------------------------------
# Internal: measurement record for a single function call
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Measurement:
    """A single measurement of a profiled function call."""

    wall: float  # wall-clock seconds
    cpu: float  # CPU seconds


# ---------------------------------------------------------------------------
# Internal: scope state (bundled into a single ContextVar)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ScopeState:
    """Internal state for a single prof.measure() scope."""

    measurements: dict[str, list[_Measurement]]
    agg_config: dict[str, list[AggSpec]]
    lock: threading.Lock


_scope_state: ContextVar[_ScopeState | None] = ContextVar("profiler_scope_state", default=None)


# ---------------------------------------------------------------------------
# Internal: flattening measurements into a profile dict
# ---------------------------------------------------------------------------


def _flatten_measurements(
    profile: dict[str, Any],
    measurements: dict[str, list[_Measurement]],
    agg_config: dict[str, list[AggSpec]],
) -> None:
    """Flatten accumulated measurements into the profile dict.

    For each key:
    - Single call: time/{key}, cpu/{key}
    - Multiple calls: time/{key}/total, time/{key}/count, cpu/{key}/total, cpu/{key}/count,
      plus user-requested aggs applied to both time/ and cpu/
    """
    for key, ms in measurements.items():
        wall_durs = [m.wall for m in ms]
        cpu_durs = [m.cpu for m in ms]

        if len(ms) == 1:
            profile[f"time/{key}"] = wall_durs[0]
            profile[f"cpu/{key}"] = cpu_durs[0]
        else:
            profile[f"time/{key}/total"] = sum(wall_durs)
            profile[f"time/{key}/count"] = len(wall_durs)
            profile[f"cpu/{key}/total"] = sum(cpu_durs)
            profile[f"cpu/{key}/count"] = len(cpu_durs)

            for agg_spec in agg_config.get(key, []):
                if isinstance(agg_spec, str):
                    profile[f"time/{key}/{agg_spec}"] = _BUILTIN_AGGS[agg_spec](wall_durs)
                    profile[f"cpu/{key}/{agg_spec}"] = _BUILTIN_AGGS[agg_spec](cpu_durs)
                else:
                    name, fn = agg_spec
                    profile[f"time/{key}/{name}"] = fn(wall_durs)
                    profile[f"cpu/{key}/{name}"] = fn(cpu_durs)


# ---------------------------------------------------------------------------
# Public API: Profiler
# ---------------------------------------------------------------------------


class Profiler:
    """Profiler for training loop functions.

    Create a ``Profiler`` instance and use :meth:`measure` to define profiling
    windows. Decorated functions (via :func:`profiled`) record their wall-clock
    and CPU durations into the active window.

    Example::

        prof = Profiler()

        @profiled("rollout", agg=["mean", "max"])
        async def do_rollout(...): ...

        for i_batch in range(n_batches):
            metrics = {"progress/batch": i_batch}

            with prof.measure() as profile:
                await do_rollout(...)

            metrics.update(profile)
            ml_logger.log_metrics(metrics, step=i_batch)
    """

    @contextmanager
    def measure(self) -> Iterator[dict[str, Any]]:
        """Open a profiling window.

        All :func:`profiled` function calls within this window have their
        wall-clock and CPU durations recorded. When the window closes, the
        accumulated measurements are flattened into the yielded dict.

        The yielded dict contains only profiler-produced keys (``time/...``,
        ``cpu/...``). It is separate from the user's training metrics dict —
        merge with ``metrics.update(profile)`` when ready.

        Automatically records ``time/total`` and ``cpu/total`` for the
        entire window.
        """
        profile: dict[str, Any] = {}
        state = _ScopeState(
            measurements={},
            agg_config={},
            lock=threading.Lock(),
        )
        token = _scope_state.set(state)
        t_wall = time.monotonic()
        t_cpu = time.process_time()
        try:
            yield profile
        finally:
            profile["time/total"] = time.monotonic() - t_wall
            profile["cpu/total"] = time.process_time() - t_cpu
            try:
                with state.lock:
                    _flatten_measurements(profile, state.measurements, state.agg_config)
            finally:
                _scope_state.reset(token)


# ---------------------------------------------------------------------------
# Public API: profiled
# ---------------------------------------------------------------------------


@overload
def profiled(key: F) -> F: ...  # @profiled (bare, no parens)


@overload
def profiled(
    key: str | None = ...,
    agg: list[AggSpec] | None = ...,
) -> Callable[[F], F]: ...  # @profiled("key") or @profiled(key="key", agg=[...])


def profiled(
    key: str | F | None = None,
    agg: list[AggSpec] | None = None,
) -> F | Callable[[F], F]:
    """Decorator that records wall-clock and CPU duration of a function.

    Args:
        key: Profile key. If omitted, uses ``func.__qualname__``.
        agg: Additional aggregations when called multiple times per
             measurement window. Built-in: ``"mean"``, ``"max"``, ``"min"``.
             Custom: ``("name", lambda durs: ...)`` for arbitrary reducers.
             ``"total"`` and ``"count"`` are always emitted for multi-call
             functions. Aggregations are applied to both ``time/`` and
             ``cpu/`` independently.

    Works on both sync and async functions.

    When no :meth:`Profiler.measure` window is active, durations are still
    logged to Python logging (at DEBUG level) but not recorded.

    Examples::

        @profiled("save_checkpoint")
        async def save_checkpoint_and_get_sampling_client(...):
            ...

        @profiled
        async def prepare_minibatch(...):
            ...

        @profiled("rollout", agg=["mean", "max"])
        async def do_rollout(...):
            ...

        @profiled("rollout", agg=[("p95", lambda durs: sorted(durs)[int(len(durs) * 0.95)])])
        async def do_rollout(...):
            ...
    """

    def decorator(func: F) -> F:
        profile_key = key if isinstance(key, str) else func.__qualname__
        effective_agg: list[AggSpec] = agg or []

        if effective_agg:
            _validate_agg_specs(effective_agg)

        def _record(wall: float, cpu: float) -> None:
            logger.debug(f"{profile_key} took {wall:.2f}s wall, {cpu:.2f}s cpu")
            state = _scope_state.get()
            if state is None:
                return
            with state.lock:
                state.measurements.setdefault(profile_key, []).append(
                    _Measurement(wall=wall, cpu=cpu)
                )
                if effective_agg:
                    state.agg_config[profile_key] = effective_agg

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t_wall = time.monotonic()
                t_cpu = time.process_time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    _record(time.monotonic() - t_wall, time.process_time() - t_cpu)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                t_wall = time.monotonic()
                t_cpu = time.process_time()
                try:
                    return func(*args, **kwargs)
                finally:
                    _record(time.monotonic() - t_wall, time.process_time() - t_cpu)

            return sync_wrapper  # type: ignore[return-value]

    # Support both @profiled and @profiled("key") and @profiled(key="key")
    if callable(key):
        func = key
        key = None
        return decorator(func)  # type: ignore[arg-type]
    return decorator  # type: ignore[return-value]
