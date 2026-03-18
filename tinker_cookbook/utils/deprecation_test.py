"""Tests for the deprecation utilities."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from tinker_cookbook.utils.deprecation import (
    _parse_version,
    deprecated,
    make_deprecated_module_getattr,
    warn_deprecated,
)

# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_simple(self):
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_dev_suffix(self):
        assert _parse_version("0.15.0.dev3+g1234") == (0, 15, 0)

    def test_prerelease(self):
        assert _parse_version("1.0.0rc1") == (1, 0, 0)

    def test_single_number(self):
        assert _parse_version("42") == (42,)

    def test_empty_fallback(self):
        assert _parse_version("") == (0,)


# ---------------------------------------------------------------------------
# warn_deprecated
# ---------------------------------------------------------------------------


class TestWarnDeprecated:
    def test_basic_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "'old_func' is deprecated." in str(w[0].message)

    def test_warning_with_removal_version(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func", removal_version="99.0.0")
        assert "removed in version 99.0.0" in str(w[0].message)

    def test_warning_with_message(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func", message="Use new_func() instead.")
        assert "Use new_func() instead." in str(w[0].message)

    def test_full_warning_message(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated(
                "my_feature",
                removal_version="99.0.0",
                message="Use better_feature instead.",
            )
        msg = str(w[0].message)
        assert "'my_feature' is deprecated." in msg
        assert "removed in version 99.0.0" in msg
        assert "Use better_feature instead." in msg

    def test_past_removal_version_raises(self):
        with patch("tinker_cookbook.utils.deprecation._current_version") as mock_ver:
            mock_ver.return_value = (1, 0, 0)
            with pytest.raises(RuntimeError, match="should have been removed"):
                warn_deprecated("old_func", removal_version="0.5.0")

    def test_no_removal_version_never_raises(self):
        """When removal_version is None, it always warns, never raises."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated("old_func", removal_version=None)
        assert len(w) == 1


# ---------------------------------------------------------------------------
# @deprecated decorator
# ---------------------------------------------------------------------------


class TestDeprecatedDecorator:
    def test_decorate_function_no_args(self):
        @deprecated
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

        assert result == "result"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old_func" in str(w[0].message)

    def test_decorate_function_with_message(self):
        @deprecated(message="Use new_func instead.", removal_version="99.0.0")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

        assert result == "result"
        assert "Use new_func instead." in str(w[0].message)
        assert "99.0.0" in str(w[0].message)

    def test_decorate_function_empty_parens(self):
        @deprecated()
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

        assert result == "result"
        assert len(w) == 1

    def test_preserves_function_metadata(self):
        @deprecated(message="msg", removal_version="99.0.0")
        def old_func() -> str:
            """Original docstring."""
            return "result"

        assert old_func.__name__ == "old_func"
        assert old_func.__doc__ == "Original docstring."

    def test_decorate_class(self):
        @deprecated(message="Use NewClass instead.", removal_version="99.0.0")
        class OldClass:
            def __init__(self, x: int):
                self.x = x

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass(42)

        assert obj.x == 42
        assert len(w) == 1
        assert "OldClass" in str(w[0].message)
        assert "Use NewClass instead." in str(w[0].message)

    def test_class_preserves_isinstance(self):
        @deprecated(message="Use NewClass instead.")
        class OldClass:
            pass

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = OldClass()

        assert isinstance(obj, OldClass)

    def test_function_with_args_and_kwargs(self):
        @deprecated(message="msg")
        def add(a: int, b: int, *, extra: int = 0) -> int:
            return a + b + extra

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add(1, 2, extra=3)

        assert result == 6
        assert len(w) == 1

    def test_past_removal_raises_on_call(self):
        @deprecated(message="Use new.", removal_version="0.1.0")
        def old_func() -> str:
            return "result"

        with patch("tinker_cookbook.utils.deprecation._current_version") as mock_ver:
            mock_ver.return_value = (1, 0, 0)
            with pytest.raises(RuntimeError, match="should have been removed"):
                old_func()


# ---------------------------------------------------------------------------
# make_deprecated_module_getattr
# ---------------------------------------------------------------------------


class TestMakeDeprecatedModuleGetattr:
    def test_unknown_attr_raises_attribute_error(self):
        getattr_fn = make_deprecated_module_getattr("mymod", {})
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr_fn("nonexistent")

    def test_redirects_with_warning(self):
        getattr_fn = make_deprecated_module_getattr(
            "mymod",
            {"OldPath": ("os.path.join", "99.0.0")},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = getattr_fn("OldPath")

        import os.path

        assert result is os.path.join
        assert len(w) == 1
        assert "mymod.OldPath" in str(w[0].message)
        assert "os.path.join" in str(w[0].message)

    def test_bad_path_raises(self):
        getattr_fn = make_deprecated_module_getattr(
            "mymod",
            {"Bad": ("NoDots", "99.0.0")},
        )
        with pytest.raises(ValueError, match="dotted path"):
            getattr_fn("Bad")
