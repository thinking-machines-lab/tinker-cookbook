"""Tests for the exception hierarchy in tinker_cookbook.exceptions.

Verifies inheritance contracts so that future changes don't accidentally
break backward compatibility (stdlib bases) or the catch-all
TinkerCookbookError base.
"""

import pytest

from tinker_cookbook.exceptions import (
    CheckpointError,
    ConfigurationError,
    DataError,
    DataFormatError,
    DataValidationError,
    RendererError,
    SandboxError,
    TinkerCookbookError,
    TrainingError,
    WeightsDownloadError,
    WeightsError,
    WeightsMergeError,
)

# ---------------------------------------------------------------------------
# Every custom exception must be a TinkerCookbookError
# ---------------------------------------------------------------------------

ALL_EXCEPTIONS = [
    ConfigurationError,
    DataError,
    DataFormatError,
    DataValidationError,
    RendererError,
    TrainingError,
    CheckpointError,
    WeightsError,
    WeightsDownloadError,
    WeightsMergeError,
    SandboxError,
]


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS, ids=lambda c: c.__name__)
def test_all_exceptions_are_tinker_cookbook_errors(exc_cls: type[Exception]):
    assert issubclass(exc_cls, TinkerCookbookError)
    assert isinstance(exc_cls("test"), TinkerCookbookError)


# ---------------------------------------------------------------------------
# Backward-compatible stdlib bases
# ---------------------------------------------------------------------------

STDLIB_COMPAT = [
    (ConfigurationError, ValueError),
    (DataError, ValueError),
    (DataFormatError, ValueError),
    (DataValidationError, ValueError),
    (RendererError, ValueError),
    (TrainingError, RuntimeError),
    (CheckpointError, RuntimeError),
    (WeightsError, RuntimeError),
    (WeightsDownloadError, RuntimeError),
    (WeightsMergeError, ValueError),
    (SandboxError, RuntimeError),
]


@pytest.mark.parametrize(
    "exc_cls, stdlib_base",
    STDLIB_COMPAT,
    ids=lambda x: x.__name__ if isinstance(x, type) else "",
)
def test_stdlib_backward_compatibility(exc_cls: type[Exception], stdlib_base: type[Exception]):
    """Existing `except ValueError:` / `except RuntimeError:` handlers must keep working."""
    assert issubclass(exc_cls, stdlib_base)
    assert isinstance(exc_cls("test"), stdlib_base)


# ---------------------------------------------------------------------------
# Subclass relationships
# ---------------------------------------------------------------------------


def test_data_subtypes():
    assert issubclass(DataFormatError, DataError)
    assert issubclass(DataValidationError, DataError)


def test_training_subtypes():
    assert issubclass(CheckpointError, TrainingError)


def test_weights_subtypes():
    assert issubclass(WeightsDownloadError, WeightsError)
    assert issubclass(WeightsMergeError, WeightsError)


# ---------------------------------------------------------------------------
# SandboxTerminatedError integration
# ---------------------------------------------------------------------------


def test_sandbox_terminated_error_is_sandbox_error():
    from tinker_cookbook.sandbox.sandbox_interface import SandboxTerminatedError

    assert issubclass(SandboxTerminatedError, SandboxError)
    assert issubclass(SandboxTerminatedError, TinkerCookbookError)
    assert issubclass(SandboxTerminatedError, RuntimeError)


# ---------------------------------------------------------------------------
# __all__ is in sync
# ---------------------------------------------------------------------------


def test_exceptions_all_is_complete():
    """__all__ in exceptions.py must list every public exception class."""
    import tinker_cookbook.exceptions as mod

    public_exc_classes = {
        name
        for name, obj in vars(mod).items()
        if isinstance(obj, type)
        and issubclass(obj, TinkerCookbookError)
        and not name.startswith("_")
    }
    assert public_exc_classes == set(mod.__all__)
