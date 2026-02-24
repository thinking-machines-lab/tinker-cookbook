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
RENDERER_NAME_METADATA_KEY = "renderer_name"


def add_renderer_name_to_user_metadata(user_metadata: dict[str, str], renderer_name: str | None) -> None:
    """Attach renderer name to training-run metadata when available."""
    if renderer_name:
        user_metadata[RENDERER_NAME_METADATA_KEY] = renderer_name


def _handle_checkpoint_renderer_check_result(
    checkpoint_path: str,
    expected_renderer_name: str,
    checkpoint_renderer_name: str | None,
) -> None:
    if checkpoint_renderer_name is None:
        logger.info("Checkpoint %s has no renderer metadata.", checkpoint_path)
    elif checkpoint_renderer_name != expected_renderer_name:
        logger.warning(
            "Renderer mismatch for checkpoint %s: checkpoint=%s current=%s",
            checkpoint_path,
            checkpoint_renderer_name,
            expected_renderer_name,
        )
    else:
        logger.info(
            "Renderer metadata matches for checkpoint %s: %s",
            checkpoint_path,
            expected_renderer_name,
        )
    return None


def get_renderer_name_from_checkpoint(
    service_client: tinker.ServiceClient, checkpoint_path: str
) -> str | None:
    """Read renderer_name metadata from the training run referenced by a checkpoint path."""
    try:
        rest_client = service_client.create_rest_client()
        training_run = rest_client.get_training_run_by_tinker_path(checkpoint_path).result()
        return (training_run.user_metadata or {}).get(RENDERER_NAME_METADATA_KEY)
    except (tinker.TinkerError, ValueError) as e:
        logger.warning(
            "Could not fetch renderer metadata for checkpoint %s: %s",
            checkpoint_path,
            e,
        )
        return None


async def get_renderer_name_from_checkpoint_async(
    service_client: tinker.ServiceClient, checkpoint_path: str
) -> str | None:
    """Async version of get_renderer_name_from_checkpoint."""
    try:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(checkpoint_path)
        return (training_run.user_metadata or {}).get(RENDERER_NAME_METADATA_KEY)
    except (tinker.TinkerError, ValueError) as e:
        logger.warning(
            "Could not fetch renderer metadata for checkpoint %s: %s",
            checkpoint_path,
            e,
        )
        return None


def check_renderer_name_for_checkpoint(
    service_client: tinker.ServiceClient,
    checkpoint_path: str,
    expected_renderer_name: str | None,
) -> None:
    """
    Inspect a checkpoint's originating training run metadata and compare renderer name.

    """
    if expected_renderer_name is None:
        return None

    checkpoint_renderer_name = get_renderer_name_from_checkpoint(service_client, checkpoint_path)

    _handle_checkpoint_renderer_check_result(
        checkpoint_path, expected_renderer_name, checkpoint_renderer_name
    )
    return None


async def check_renderer_name_for_checkpoint_async(
    service_client: tinker.ServiceClient,
    checkpoint_path: str,
    expected_renderer_name: str | None,
) -> None:
    """
    Compare an expected renderer with renderer metadata attached to a checkpoint's training run.

    Behavior:
    - If ``expected_renderer_name`` is None, returns None and does no check.
    - Otherwise fetches ``renderer_name`` from the run referenced by ``checkpoint_path``.
    - Logs info if metadata is missing or matches.
    - Logs warning if the checkpoint renderer differs from the expected renderer.

    """
    if expected_renderer_name is None:
        return None

    checkpoint_renderer_name = await get_renderer_name_from_checkpoint_async(
        service_client, checkpoint_path
    )

    _handle_checkpoint_renderer_check_result(
        checkpoint_path, expected_renderer_name, checkpoint_renderer_name
    )
    return None


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
    ttl_seconds: int | None = None,
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
        futures["state"] = await training_client.save_state_async(name, ttl_seconds=ttl_seconds)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(
            name, ttl_seconds=ttl_seconds
        )

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
    ttl_seconds: int | None = None,
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
            training_client,
            name=name,
            log_path=log_path,
            kind=kind,
            loop_state=loop_state,
            ttl_seconds=ttl_seconds,
        )
    )
