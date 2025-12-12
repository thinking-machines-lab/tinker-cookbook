import logging
import math

from typing import Literal

logger = logging.getLogger(__name__)


LRSchedule = Literal["linear", "cosine", "constant"]


def compute_schedule_lr_multiplier(lr_schedule: LRSchedule, step: int, total_steps: int) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule
    """
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif lr_schedule == "constant":
        return 1
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
