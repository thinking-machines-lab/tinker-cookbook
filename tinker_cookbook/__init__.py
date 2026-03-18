"""Tinker Cookbook: post-training algorithms using the Tinker API."""

try:
    from tinker_cookbook._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("tinker_cookbook")
    except Exception:
        __version__ = "0.0.0.dev0+unknown"

from tinker_cookbook.exceptions import (
    CheckpointError,
    ConfigurationError,
    DataError,
    DataFormatError,
    DataValidationError,
    EvalError,
    RendererError,
    SandboxError,
    TinkerCookbookError,
    TrainingError,
    WeightsDownloadError,
    WeightsError,
    WeightsMergeError,
)

__all__ = [
    "__version__",
    "CheckpointError",
    "ConfigurationError",
    "DataError",
    "DataFormatError",
    "DataValidationError",
    "EvalError",
    "RendererError",
    "SandboxError",
    "TinkerCookbookError",
    "TrainingError",
    "WeightsDownloadError",
    "WeightsError",
    "WeightsMergeError",
]
