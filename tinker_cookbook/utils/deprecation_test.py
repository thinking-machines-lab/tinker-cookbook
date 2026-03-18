"""Tests for the deprecation utilities."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from tinker_cookbook.utils.deprecation import (
    _check_past_removal,
    deprecated,
    deprecated_module_attr,
    warn_deprecated,
)

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
            from packaging.version import Version

            mock_ver.return_value = Version("1.0.0")
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
        @deprecated("Use new_func instead.", removal_version="99.0.0")
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
        @deprecated("msg", removal_version="99.0.0")
        def old_func() -> str:
            """Original docstring."""
            return "result"

        assert old_func.__name__ == "old_func"
        assert old_func.__doc__ == "Original docstring."

    def test_decorate_class(self):
        @deprecated("Use NewClass instead.", removal_version="99.0.0")
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

    def test_function_with_args_and_kwargs(self):
        @deprecated("msg")
        def add(a: int, b: int, *, extra: int = 0) -> int:
            return a + b + extra

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = add(1, 2, extra=3)

        assert result == 6
        assert len(w) == 1

    def test_past_removal_raises_on_call(self):
        @deprecated("Use new.", removal_version="0.1.0")
        def old_func() -> str:
            return "result"

        with patch("tinker_cookbook.utils.deprecation._current_version") as mock_ver:
            from packaging.version import Version

            mock_ver.return_value = Version("1.0.0")
            with pytest.raises(RuntimeError, match="should have been removed"):
                old_func()


# ---------------------------------------------------------------------------
# _check_past_removal
# ---------------------------------------------------------------------------


class TestCheckPastRemoval:
    def test_none_removal_returns_false(self):
        assert _check_past_removal(None) is False

    def test_future_version_returns_false(self):
        assert _check_past_removal("999.0.0") is False

    def test_past_version(self):
        with patch("tinker_cookbook.utils.deprecation._current_version") as mock_ver:
            from packaging.version import Version

            mock_ver.return_value = Version("2.0.0")
            assert _check_past_removal("1.0.0") is True

    def test_equal_version(self):
        with patch("tinker_cookbook.utils.deprecation._current_version") as mock_ver:
            from packaging.version import Version

            mock_ver.return_value = Version("1.0.0")
            assert _check_past_removal("1.0.0") is True


# ---------------------------------------------------------------------------
# deprecated_module_attr
# ---------------------------------------------------------------------------


class TestDeprecatedModuleAttr:
    def test_unknown_attr_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            deprecated_module_attr(
                "nonexistent",
                module_name="mymod",
                attrs={},
            )

    def test_redirects_with_warning(self):
        # Use a known importable object as the redirect target
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecated_module_attr(
                "OldVersion",
                module_name="mymod",
                attrs={
                    "OldVersion": ("packaging.version.Version", "99.0.0"),
                },
            )

        from packaging.version import Version

        assert result is Version
        assert len(w) == 1
        assert "mymod.OldVersion" in str(w[0].message)
        assert "packaging.version.Version" in str(w[0].message)

    def test_bad_path_raises(self):
        with pytest.raises(ValueError, match="dotted path"):
            deprecated_module_attr(
                "Bad",
                module_name="mymod",
                attrs={"Bad": ("NoDots", "99.0.0")},
            )
