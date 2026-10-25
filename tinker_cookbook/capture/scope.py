"""Ambient capture scopes backed by a single ContextVar.

A capture scope is an immutable mapping of JSON-serializable key/value pairs
that ambiently tags any SDK calls made while the scope is active. Scopes nest:
entering a scope merges its pairs over the enclosing scope's pairs, and exiting
restores the previous mapping exactly (ContextVar token reset). Because the
mapping lives in a ContextVar, each asyncio task and each
``contextvars.copy_context()``-propagated thread sees its own isolated value.

Reserved keys (conventions used by the capture pipeline; arbitrary extra keys
are accepted):

- ``run_id``: stable identifier for the training run
- ``run_attempt``: integer attempt/restart counter for the run
- ``split``: e.g. ``"train"`` or ``"test"``
- ``iteration``: training iteration / step index
- ``group_idx``: trajectory-group index within the iteration
- ``traj_idx``: trajectory index within the group
- ``purpose``: free-form purpose tag, e.g. ``"rollout"`` or ``"eval"``

Values must be JSON-serializable scalars (``str | int | float | bool | None``).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Coroutine, Iterator, Mapping
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import ParamSpec, TypeVar, cast

#: The value type of a capture scope entry: JSON-serializable scalars only,
#: enforced at scope entry by ``_validate_pairs``.
ScopeValue = str | int | float | bool | None

_P = ParamSpec("_P")
_R = TypeVar("_R")

RESERVED_KEYS: frozenset[str] = frozenset(
    {"run_id", "run_attempt", "split", "iteration", "group_idx", "traj_idx", "purpose"}
)

_EMPTY_SCOPE: Mapping[str, ScopeValue] = MappingProxyType({})

_scope_var: ContextVar[Mapping[str, ScopeValue]] = ContextVar("capture_scope", default=_EMPTY_SCOPE)

_SCALAR_TYPES = (str, int, float, bool, type(None))


def current_scope() -> Mapping[str, ScopeValue]:
    """Return the currently active capture scope as an immutable mapping.

    Returns an empty mapping when no scope is active. The returned mapping is
    a live view of the current scope; snapshot it with ``dict(...)`` if you
    need a value that survives scope exit (see the snapshot-at-call rule in
    the README).
    """
    return _scope_var.get()


def _validate_pairs(pairs: dict[str, ScopeValue]) -> None:
    for key, value in pairs.items():
        if not isinstance(value, _SCALAR_TYPES):
            raise TypeError(
                f"capture scope value for {key!r} must be a JSON-serializable scalar "
                f"(str, int, float, bool, or None), got {type(value).__name__}"
            )


class capture:
    """Push pairs onto the ambient capture scope for the duration of a block.

    Merges over the enclosing scope (inner keys win) and restores the previous
    scope exactly on exit. Usable as a context manager or as a decorator on
    both sync and async functions::

        with capture(run_id="run-1", iteration=3):
            ...  # SDK calls here are tagged

        @capture(purpose="eval")
        def run_eval() -> None:
            ...

        @capture(purpose="eval")
        async def run_eval_async() -> None:
            ...  # the scope is active while the BODY runs, even if the
                 # coroutine is created in one place and awaited elsewhere

    The async-decorator case matters: a plain ``ContextDecorator`` would
    enter/exit around coroutine *creation*, leaving the body unattributed, so
    the decorator path enters the scope inside the coroutine instead.

    Args:
        **pairs: JSON-serializable scalar values to add to the scope. Reserved
            keys (:data:`RESERVED_KEYS`) have conventional meanings but any
            key is accepted.
    """

    def __init__(self, **pairs: ScopeValue) -> None:
        _validate_pairs(pairs)
        self._pairs = pairs
        # Reset tokens are kept in a per-instance ContextVar (a tuple used as
        # a stack) rather than instance state: one instance may be entered by
        # overlapping asyncio tasks or threads, and an instance-level list
        # would mix tokens across execution contexts (ContextVar.reset with
        # a foreign token raises "Token was created in a different Context").
        # A ContextVar keeps each task's/thread's entries isolated while
        # still supporting same-context re-entrancy.
        self._entries: ContextVar[tuple[Token[Mapping[str, ScopeValue]], ...]] = ContextVar(
            "capture_entries", default=()
        )

    def __enter__(self) -> Mapping[str, ScopeValue]:
        merged = MappingProxyType({**_scope_var.get(), **self._pairs})
        token = _scope_var.set(merged)
        self._entries.set((*self._entries.get(), token))
        return merged

    def __exit__(self, *exc: object) -> None:
        stack = self._entries.get()
        self._entries.set(stack[:-1])
        _scope_var.reset(stack[-1])

    def __call__(self, func: Callable[_P, _R]) -> Callable[_P, _R]:
        if inspect.iscoroutinefunction(func):
            # func: Callable[_P, Coroutine[..., ..., X]]; the wrapper has the
            # same shape, but pyright cannot relate X back to _R through the
            # iscoroutinefunction narrowing, hence the cast.
            async_func = cast(Callable[_P, Coroutine[object, object, object]], func)

            @functools.wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> object:
                with capture(**self._pairs):
                    return await async_func(*args, **kwargs)

            return cast(Callable[_P, _R], async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with capture(**self._pairs):
                return func(*args, **kwargs)

        return sync_wrapper


@contextlib.contextmanager
def replace_scope(pairs: Mapping[str, ScopeValue]) -> Iterator[Mapping[str, ScopeValue]]:
    """Set the ambient scope to exactly ``pairs``, ignoring any enclosing scope.

    Unlike :func:`capture`, this does NOT merge. It exists for re-entering a
    transmitted scope snapshot in a context that may already carry unrelated
    (or stale) scope state. The main user is thread propagation (see
    ``propagate.py``): a worker thread reused across tasks (e.g. in a
    ``ThreadPoolExecutor``) may still carry scope from a previous task, so
    entering the submitting context's snapshot must replace, not merge. The
    same applies to a scope snapshot transported to another process (e.g.
    pickled to a spawn-context worker). The previous scope is restored on
    exit.
    """
    replaced = dict(pairs)
    _validate_pairs(replaced)
    token = _scope_var.set(MappingProxyType(replaced))
    try:
        yield _scope_var.get()
    finally:
        _scope_var.reset(token)
