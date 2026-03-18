"""
Deprecation utilities for managing API evolution in tinker-cookbook.

This module provides tools for deprecating functions, classes, parameters,
and module-level attributes with clear migration guidance and automatic
enforcement when the removal version is reached.

Usage examples::

    from tinker_cookbook.utils.deprecation import deprecated, warn_deprecated

    # Deprecate an entire function or class
    @deprecated(message="Use new_func() instead.", removal_version="0.20.0")
    def old_func(x):
        return new_func(x)

    # Deprecate inside a function body (e.g., a parameter)
    def train(*, lr, learning_rate=None):
        if learning_rate is not None:
            warn_deprecated(
                "learning_rate",
                removal_version="0.20.0",
                message="Use the 'lr' parameter instead.",
            )
            lr = learning_rate
        ...

    # Deprecate a module-level attribute (put in the module's __init__.py)
    __getattr__ = make_deprecated_module_getattr(
        __name__,
        {"OldClass": ("new_module.NewClass", "0.20.0")},
    )
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of ints.

    Strips any pre-release/dev suffixes (e.g. ``"0.15.0.dev3+g1234"`` becomes
    ``(0, 15, 0)``).  This avoids a dependency on ``packaging``.
    """
    parts: list[int] = []
    for segment in v.split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) if parts else (0,)


def _current_version() -> tuple[int, ...]:
    """Return the current package version as a comparable tuple."""
    try:
        raw = importlib.metadata.version("tinker_cookbook")
        return _parse_version(raw)
    except Exception:
        return (0, 0, 0)


def _check_past_removal(removal_version: str | None) -> bool:
    """Return True if the current version is at or past the removal version."""
    if removal_version is None:
        return False
    try:
        return _current_version() >= _parse_version(removal_version)
    except Exception:
        return False


def warn_deprecated(
    name: str,
    *,
    removal_version: str | None = None,
    message: str = "",
    stacklevel: int = 2,
) -> None:
    """Emit a DeprecationWarning for a deprecated feature.

    If the current package version is at or past *removal_version*, raises
    a ``RuntimeError`` instead so that stale deprecated code paths are not
    silently used after their intended removal date.

    Args:
        name: Short identifier for the deprecated feature (e.g. function name,
            parameter name).
        removal_version: The version in which this feature will be removed.
            When the running version reaches this value the warning becomes
            a hard error.  Pass ``None`` to warn without a scheduled removal.
        message: Additional guidance, typically a migration path such as
            "Use X instead."
        stacklevel: Passed through to ``warnings.warn``. The default of 2
            points at the caller of the function that calls ``warn_deprecated``.
    """
    parts: list[str] = [f"'{name}' is deprecated."]
    if removal_version is not None:
        parts.append(f"It will be removed in version {removal_version}.")
    if message:
        parts.append(message)
    full_message = " ".join(parts)

    if _check_past_removal(removal_version):
        raise RuntimeError(
            f"{full_message} (Current version is "
            f"{'.'.join(str(x) for x in _current_version())}; "
            f"this should have been removed by {removal_version}.)"
        )

    warnings.warn(full_message, DeprecationWarning, stacklevel=stacklevel)


@overload
def deprecated(__func: F) -> F: ...


@overload
def deprecated(
    *,
    message: str = ...,
    removal_version: str | None = ...,
) -> Callable[[F], F]: ...


def deprecated(
    _func: Callable[..., Any] | None = None,
    *,
    message: str = "",
    removal_version: str | None = None,
) -> Any:
    """Decorator to mark a function or class as deprecated.

    Can be used with or without arguments::

        @deprecated
        def old(): ...

        @deprecated(message="Use new_func instead.", removal_version="0.20.0")
        def old(): ...

    When applied to a class, the warning is emitted at instantiation time.
    """

    def decorator(obj: F) -> F:
        obj_name: str = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        kind = "Class" if isinstance(obj, type) else "Function"

        if isinstance(obj, type):
            original_init: Callable[..., None] = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(
                    f"{kind} {obj_name}",
                    removal_version=removal_version,
                    message=message,
                    stacklevel=2,
                )
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[misc]
            return obj  # type: ignore[return-value]
        else:

            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warn_deprecated(
                    f"{kind} {obj_name}",
                    removal_version=removal_version,
                    message=message,
                    stacklevel=2,
                )
                return obj(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

    if _func is not None:
        return decorator(_func)

    return decorator


def make_deprecated_module_getattr(
    module_name: str,
    attrs: dict[str, tuple[str, str | None]],
) -> Callable[[str], Any]:
    """Create a ``__getattr__`` function for deprecating module-level attributes.

    Returns a function suitable for assigning to ``__getattr__`` at module scope.
    When an old attribute name is accessed, it emits a deprecation warning and
    transparently returns the new object.

    Args:
        module_name: ``__name__`` of the module defining ``__getattr__``.
        attrs: Mapping of ``{old_name: (dotted_path_to_new, removal_version)}``.
            *dotted_path_to_new* is ``"package.module.NewName"`` and will be
            imported and returned.  *removal_version* may be ``None``.

    Returns:
        A ``__getattr__`` function.

    Example::

        # In mymodule/__init__.py
        __getattr__ = make_deprecated_module_getattr(
            __name__,
            {"OldThing": ("mymodule.new_place.NewThing", "0.20.0")},
        )
    """

    def __getattr__(name: str) -> Any:
        if name not in attrs:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

        new_path, removal_version = attrs[name]

        module_path, _, attr_name = new_path.rpartition(".")
        if not module_path:
            raise ValueError(
                f"make_deprecated_module_getattr: new path {new_path!r} must be a "
                f"dotted path (e.g. 'package.module.Name')"
            )

        mod = importlib.import_module(module_path)
        replacement = getattr(mod, attr_name)

        warn_deprecated(
            f"{module_name}.{name}",
            removal_version=removal_version,
            message=f"Use {new_path} instead.",
            stacklevel=2,
        )
        return replacement

    return __getattr__
