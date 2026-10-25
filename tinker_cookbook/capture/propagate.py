"""Automatic capture scope propagation across thread boundaries.

``instrument_threads()`` monkeypatches the stdlib threading and thread-pool
entry points (OTel ``ThreadingInstrumentor``-shaped) so ambient ``capture(...)``
scopes reach worker threads automatically.
"""

from __future__ import annotations

import functools
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tinker_cookbook.capture.scope import ScopeValue, current_scope, replace_scope

_ORIGINAL_ATTR = "_capture_propagate_original"

# (class, method_name) -> original callable
_patched: dict[tuple[type, str], Callable[..., Any]] = {}


def _patch(
    cls: type, name: str, wrapper: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> None:
    key = (cls, name)
    if key in _patched:
        return
    original = getattr(cls, name)
    if getattr(original, _ORIGINAL_ATTR, None) is not None:
        return
    wrapped = wrapper(original)
    setattr(wrapped, _ORIGINAL_ATTR, original)
    _patched[key] = original
    setattr(cls, name, wrapped)


# Context captured at Thread.start, consumed in Thread.run (WeakKeyDictionary
# avoids mutating Thread instances and keeps typing clean).
_thread_contexts: weakref.WeakKeyDictionary[threading.Thread, dict[str, ScopeValue]] = (
    weakref.WeakKeyDictionary()
)
_thread_contexts_lock = threading.Lock()


def _wrap_thread_start(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: threading.Thread, *args: object, **kwargs: object) -> Any:
        with _thread_contexts_lock:
            _thread_contexts[self] = dict(current_scope())
        return original(self, *args, **kwargs)

    return wrapper


def _wrap_thread_run(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(self: threading.Thread, *args: object, **kwargs: object) -> Any:
        with _thread_contexts_lock:
            scope = _thread_contexts.pop(self, None)
        if scope is not None:
            with replace_scope(scope):
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    return wrapper


def _wrap_pool_submit(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(
        self: ThreadPoolExecutor,
        fn: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> Any:
        scope = dict(current_scope())

        def wrapped_func(*func_args: object, **func_kwargs: object) -> Any:
            with replace_scope(scope):
                return fn(*func_args, **func_kwargs)

        return original(self, wrapped_func, *args, **kwargs)

    return wrapper


def instrument_threads() -> None:
    """Patch stdlib threading paths to propagate capture scope.

    Idempotent: repeated calls are no-ops once patched.
    """
    _patch(threading.Thread, "start", _wrap_thread_start)
    _patch(threading.Thread, "run", _wrap_thread_run)
    # Timer overrides Thread.run, so patch its override separately. Its
    # inherited Thread.start already snapshots the context.
    _patch(threading.Timer, "run", _wrap_thread_run)
    _patch(ThreadPoolExecutor, "submit", _wrap_pool_submit)


def threads_instrumented() -> bool:
    """True while :func:`instrument_threads` patches are applied.

    Nesting-aware callers (``capture_to_store``) save this on enter and only
    uninstrument on exit if they were the ones who turned propagation on, so
    an outer session's (or an independent caller's) patches survive an inner
    exit.
    """
    return bool(_patched)


def uninstrument_threads() -> None:
    """Restore patched threading methods. Idempotent."""
    for (cls, name), original in list(_patched.items()):
        setattr(cls, name, original)
        del _patched[(cls, name)]
    with _thread_contexts_lock:
        _thread_contexts.clear()
