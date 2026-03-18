"""
Deprecation utilities for managing API evolution in tinker-cookbook.

This module provides tools for deprecating functions, classes, parameters,
and module-level attributes with clear migration guidance and automatic
enforcement when the removal version is reached.

Usage examples::

    from tinker_cookbook.utils.deprecation import deprecated, warn_deprecated

    # Deprecate an entire function or class
    @deprecated("Use new_func() instead.", removal_version="0.20.0")
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
    def __getattr__(name):
        return deprecated_module_attr(
            name,
            module_name=__name__,
            attrs={"OldClass": ("new_module.NewClass", "0.20.0")},
        )
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from packaging.version import Version

F = TypeVar("F", bound=Callable[..., Any])


def _current_version() -> Version:
    """Return the current package version, or 0.0.0 if unavailable."""
    try:
        from tinker_cookbook import __version__

        return Version(__version__)
    except Exception:
        return Version("0.0.0")


def _check_past_removal(removal_version: str | None) -> bool:
    """Return True if the current version is at or past the removal version."""
    if removal_version is None:
        return False
    try:
        return _current_version() >= Version(removal_version)
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
            f"{full_message} (Current version is {_current_version()}; "
            f"this should have been removed by {removal_version}.)"
        )

    warnings.warn(full_message, DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    _func_or_message: Callable[..., Any] | str | None = None,
    *,
    message: str = "",
    removal_version: str | None = None,
) -> Any:
    """Decorator to mark a function or class as deprecated.

    Can be used with or without arguments::

        @deprecated
        def old(): ...

        @deprecated("Use new_func instead.", removal_version="0.20.0")
        def old(): ...

        @deprecated(message="Use new_func instead.", removal_version="0.20.0")
        def old(): ...

    When applied to a class, the warning is emitted at instantiation time.
    """
    # Resolve the actual message when the first positional arg is a string
    effective_message = message
    if isinstance(_func_or_message, str):
        effective_message = _func_or_message
        _func_or_message = None

    def decorator(obj: Any) -> Any:
        obj_name: str = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        kind = "Class" if isinstance(obj, type) else "Function"

        if isinstance(obj, type):
            original_init: Callable[..., None] = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(
                    f"{kind} {obj_name}",
                    removal_version=removal_version,
                    message=effective_message,
                    stacklevel=2,
                )
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[misc]
            return obj
        else:

            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warn_deprecated(
                    f"{kind} {obj_name}",
                    removal_version=removal_version,
                    message=effective_message,
                    stacklevel=2,
                )
                return obj(*args, **kwargs)

            return wrapper

    # @deprecated (no parentheses) — _func_or_message is the decorated callable
    if callable(_func_or_message):
        return decorator(_func_or_message)

    # @deprecated(), @deprecated("msg"), @deprecated(message="msg", ...)
    return decorator


def deprecated_module_attr(
    name: str,
    *,
    module_name: str,
    attrs: dict[str, tuple[str, str | None]],
) -> Any:
    """Helper for deprecating module-level attributes via ``__getattr__``.

    Place this in a module's ``__getattr__`` to redirect old attribute names
    to their new locations with a deprecation warning.

    Args:
        name: The attribute name being looked up.
        module_name: ``__name__`` of the module defining ``__getattr__``.
        attrs: Mapping of ``{old_name: (dotted_path_to_new, removal_version)}``.
            *dotted_path_to_new* is ``"package.module.NewName"`` and will be
            imported and returned.  *removal_version* may be ``None``.

    Returns:
        The resolved new attribute.

    Raises:
        AttributeError: If *name* is not in *attrs*.
        RuntimeError: If the removal version has passed.

    Example::

        # In mymodule/__init__.py
        def __getattr__(name):
            return deprecated_module_attr(
                name,
                module_name=__name__,
                attrs={
                    "OldThing": ("mymodule.new_place.NewThing", "0.20.0"),
                },
            )
    """
    if name not in attrs:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    new_path, removal_version = attrs[name]

    # Import the replacement
    module_path, _, attr_name = new_path.rpartition(".")
    if module_path:
        import importlib

        mod = importlib.import_module(module_path)
        replacement = getattr(mod, attr_name)
    else:
        raise ValueError(
            f"deprecated_module_attr: new path {new_path!r} must be a dotted path "
            f"(e.g. 'package.module.Name')"
        )

    warn_deprecated(
        f"{module_name}.{name}",
        removal_version=removal_version,
        message=f"Use {new_path} instead.",
        stacklevel=3,
    )
    return replacement
