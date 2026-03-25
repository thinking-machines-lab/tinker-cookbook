"""Centralized exception hierarchy for tinker-cookbook.

All custom exceptions inherit from :class:`TinkerCookbookError`, making it easy
for downstream consumers to catch *any* cookbook-specific error with a single
``except TinkerCookbookError`` clause while still allowing fine-grained handling
of specific error categories.

This module does **not** replace the Tinker SDK's own exception hierarchy
(``tinker.TinkerError``, ``tinker.APIError``, etc.).  Those exceptions are
raised by the SDK when communicating with the Tinker service; the exceptions
here cover errors that originate in the cookbook's own logic — configuration
validation, data loading, rendering, weight management, and so on.

Typical usage::

    from tinker_cookbook.exceptions import ConfigurationError, DataError

    if model_name not in KNOWN_MODELS:
        raise ConfigurationError(f"Unknown model: {model_name}")

Adding a new exception
~~~~~~~~~~~~~~~~~~~~~~

1. Subclass :class:`TinkerCookbookError` (or a category subclass like
   :class:`DataError`).
2. Also inherit from the stdlib exception it replaces (e.g. ``ValueError``,
   ``RuntimeError``) so that existing ``except`` clauses keep working.
3. Add it to :data:`__all__` below **and** to ``tinker_cookbook/__init__.py``.
4. Keep exceptions picklable — do **not** add custom ``__init__`` parameters
   without implementing ``__reduce__``.  Picklability is required for
   ``multiprocessing`` and distributed task frameworks.
"""

__all__ = [
    "TinkerCookbookError",
    "ConfigurationError",
    "DataError",
    "DataFormatError",
    "DataValidationError",
    "RendererError",
    "TrainingError",
    "CheckpointError",
    "AllTrajectoriesFailedError",
    "WeightsError",
    "WeightsDownloadError",
    "WeightsMergeError",
    "WeightsAdapterError",
    "SandboxError",
]


class TinkerCookbookError(Exception):
    """Base exception for all tinker-cookbook errors.

    Catch this to handle any error raised by cookbook code (as opposed to
    errors from the Tinker SDK or third-party libraries).
    """


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(TinkerCookbookError, ValueError):
    """A configuration parameter is invalid or missing.

    Raised when user-supplied configuration (model names, hyperparameters,
    renderer names, required fields, etc.) fails validation.  Inherits from
    :class:`ValueError` for backward compatibility with code that already
    catches ``ValueError`` for configuration problems.

    Examples:
        - Unknown model name
        - Missing required config key (e.g. ``kl_reference_config``)
        - Invalid hyperparameter combination
    """


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------


class DataError(TinkerCookbookError, ValueError):
    """An error related to training or evaluation data.

    Base class for data-related errors.  Inherits from :class:`ValueError`
    for backward compatibility.
    """


class DataFormatError(DataError):
    """Data is not in the expected format.

    Raised when input data (JSONL files, HuggingFace datasets, conversation
    dicts, etc.) is structurally malformed — e.g. a missing ``messages``
    field in a JSONL line, or a conversation with too few tokens.
    """


class DataValidationError(DataError):
    """Data fails a semantic validation check.

    Raised when data is structurally correct but violates a logical
    constraint — e.g. streaming datasets cannot seek backward, or
    there are not enough tokens for an input/target split.
    """


# ---------------------------------------------------------------------------
# Renderer errors
# ---------------------------------------------------------------------------


class RendererError(TinkerCookbookError, ValueError):
    """An error related to renderer configuration or rendering.

    Raised when a renderer cannot be found, messages cannot be rendered
    into a model prompt, or a response cannot be parsed back into messages.
    Inherits from :class:`ValueError` for backward compatibility.
    """


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------


class TrainingError(TinkerCookbookError, RuntimeError):
    """An error during a training loop.

    Base class for errors that occur while executing SL, RL, DPO, or
    distillation training loops.  Inherits from :class:`RuntimeError`
    for backward compatibility.
    """


class CheckpointError(TrainingError):
    """An error related to saving, loading, or resuming checkpoints.

    Raised when a checkpoint file is missing, corrupted, or when the
    save/load operation fails.
    """


class AllTrajectoriesFailedError(TrainingError):
    """All trajectories in a rollout group failed.

    Caught internally by the rollout pipeline to skip the affected group
    rather than crash the training run.
    """


# ---------------------------------------------------------------------------
# Weights errors
# ---------------------------------------------------------------------------


class WeightsError(TinkerCookbookError):
    """An error related to weight download, merge, or export.

    Grouping base for weights-related errors.  Does not inherit from a
    stdlib exception — use the specific subclasses which each carry
    exactly one stdlib base appropriate to their failure mode.
    """


class WeightsDownloadError(WeightsError, RuntimeError):
    """Failed to download weights from Tinker storage.

    Raised when the Tinker service cannot be reached, the checkpoint
    path is invalid, or the download archive is corrupt.  Inherits from
    :class:`RuntimeError` because these are operational failures.
    """


class WeightsMergeError(WeightsError, ValueError):
    """Failed to merge LoRA adapter weights into a base model.

    Raised when adapter weights are incompatible with the base model
    (shape mismatches, missing keys, etc.).  Inherits from
    :class:`ValueError` because merge errors are validation failures
    (wrong shapes, missing config keys).
    """


class WeightsAdapterError(WeightsError, ValueError):
    """Failed to convert a Tinker LoRA adapter to PEFT format.

    Raised when the adapter cannot be converted for serving — e.g. the
    model family is not supported for adapter serving, adapter key names
    cannot be remapped, or tensor shapes are inconsistent.  Inherits
    from :class:`ValueError` because adapter conversion errors are
    validation failures.
    """


# ---------------------------------------------------------------------------
# Sandbox errors
# ---------------------------------------------------------------------------


class SandboxError(TinkerCookbookError, RuntimeError):
    """An error related to code-execution sandboxes.

    Base class for sandbox errors — e.g. sandbox termination, timeouts,
    or unexpected sandbox failures.
    """
