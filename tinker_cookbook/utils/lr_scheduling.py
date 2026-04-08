import logging
import math
from typing import Literal

from tinker_cookbook.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


LRSchedule = Literal["linear", "cosine", "constant"]
"""Type alias for supported learning-rate schedule names."""


def compute_schedule_lr_multiplier(lr_schedule: LRSchedule, step: int, total_steps: int) -> float:
    """Compute the learning-rate decay multiplier for a given step.

    The returned value should be multiplied by the base learning rate to
    obtain the effective LR at this step.

    Args:
        lr_schedule (LRSchedule): One of ``"linear"``, ``"cosine"``, or
            ``"constant"``.
        step (int): Current training step (0-indexed).
        total_steps (int): Total number of training steps.

    Returns:
        float: Multiplier in ``[0, 1]`` to apply to the base learning rate.

    Raises:
        ConfigurationError: If *lr_schedule* is not a recognized schedule.

    Example::

        base_lr = 1e-4
        effective_lr = base_lr * compute_schedule_lr_multiplier("cosine", step=50, total_steps=100)
    """
    if total_steps <= 0:
        raise ConfigurationError(f"total_steps must be positive, got {total_steps}")
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif lr_schedule == "constant":
        return 1
    else:
        raise ConfigurationError(f"Unknown learning rate schedule: {lr_schedule}")
