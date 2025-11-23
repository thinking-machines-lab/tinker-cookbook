import asyncio
import json
import logging
import os
from typing import Any, Literal

import tinker

from tinker_cookbook.utils.file_utils import read_jsonl
from tinker_cookbook.utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)


@scope
def load_checkpoints_file(log_dir: str) -> list[dict[str, Any]]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return read_jsonl(checkpoint_path)


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """
    Get the last checkpoint from the checkpoints.jsonl file in the specified log directory.

    Args:
        log_dir: The directory to check.
        required_key: The key to check for in the checkpoint.
            We might save partial checkpoints (e.g. sampler) in the same file,
            so we need to filter to the rows that have a fully-resumable checkpoint.

    Returns:
        The last checkpoint, or None if no checkpoint is found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{required_key}' in {log_dir}"
        )
        logger.info(f"Using last checkpoint: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.info(f"No checkpoints found with key {required_key} in {log_dir}")
        return None


@scope
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


@scope
def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    return asyncio.run(
        save_checkpoint_async(
            training_client, name=name, log_path=log_path, kind=kind, loop_state=loop_state
        )
    )
