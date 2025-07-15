"""
Training Utilities Module

This module provides common utilities for training scripts including:
- Learning rate computation
- Training client creation
- Checkpoint saving
- Logging setup

These functions are designed to be compatible with different training algorithms
(SFT, DPO, etc.)
"""

import logging

import tinker_public

logger = logging.getLogger(__name__)


def save_checkpoint(training_client: tinker_public.TrainingClient, name: str) -> str:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
    Returns:
        Path where checkpoint was saved
    """
    # XXX currently saving both sampler and state
    save_weights_future = training_client.save_weights_for_sampler(name)
    save_state_future = training_client.save_state(name)
    save_weights_result = save_weights_future.result()
    save_state_result = save_state_future.result()
    logger.info(f"Saved weights for sampler to: {save_weights_result.path}")
    logger.info(f"Saved state to: {save_state_result.path}")
    return save_weights_result.path


def compute_schedule_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule
    """
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "constant":
        return 1
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
