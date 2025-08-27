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

import tinker

logger = logging.getLogger(__name__)


def save_checkpoint(training_client: tinker.TrainingClient, name: str) -> str:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
    Returns:
        Dictionary with 'state_path' key
    """
    save_state_future = training_client.save_state(name)
    save_state_result = save_state_future.result()
    logger.info(f"Saved state to: {save_state_result.path}")
    return save_state_result.path


def save_sampling_checkpoint(training_client: tinker.TrainingClient, name: str) -> str:
    """Save model checkpoint for sampling.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
    Returns:
        Dictionary with 'weights_path' key
    """
    save_weights_future = training_client.save_weights_for_sampler(name)
    save_weights_result = save_weights_future.result()
    logger.info(f"Saved weights for sampler to: {save_weights_result.path}")
    return save_weights_result.path


async def save_checkpoint_async(training_client: tinker.TrainingClient, name: str) -> str:
    """Save current weights and return the path."""
    save_state_future = await training_client.save_state_async(name)
    save_state_result = await save_state_future.result_async()
    logger.info(f"Saved state to {save_state_result.path}")
    return save_state_result.path


async def save_sampling_checkpoint_async(training_client: tinker.TrainingClient, name: str) -> str:
    """Save current weights and return the path."""
    save_sampler_future = await training_client.save_weights_for_sampler_async(name)
    save_sampler_result = await save_sampler_future.result_async()
    logger.info(f"Saved sampler weights to {save_sampler_result.path}")
    return save_sampler_result.path


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
