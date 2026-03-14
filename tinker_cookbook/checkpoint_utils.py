import asyncio
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import tinker

from tinker_cookbook import model_info
from tinker_cookbook.utils.file_utils import read_jsonl
from tinker_cookbook.utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)
RENDERER_NAME_METADATA_KEY = "renderer_name"


@dataclass
class CheckpointRecord:
    """A single checkpoint record stored in ``checkpoints.jsonl``.

    Known fields are exposed as typed attributes. Any additional user-supplied
    metadata from ``loop_state`` is preserved in :attr:`extra` so that custom
    keys round-trip through save/load without loss.
    """

    name: str
    batch: int
    epoch: int | None = None
    final: bool | None = None
    state_path: str | None = None
    sampler_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage. Omits ``None`` optional fields."""
        d: dict[str, Any] = {"name": self.name, "batch": self.batch}
        if self.epoch is not None:
            d["epoch"] = self.epoch
        if self.final is not None:
            d["final"] = self.final
        if self.state_path is not None:
            d["state_path"] = self.state_path
        if self.sampler_path is not None:
            d["sampler_path"] = self.sampler_path
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointRecord":
        """Deserialize from a JSON-parsed dict."""
        return cls(
            name=d["name"],
            batch=d["batch"],
            epoch=d.get("epoch"),
            final=d.get("final"),
            state_path=d.get("state_path"),
            sampler_path=d.get("sampler_path"),
            extra={k: v for k, v in d.items() if k not in _CHECKPOINT_RECORD_KNOWN_KEYS},
        )

    def has(self, key: str) -> bool:
        """Check whether a field is present (not None), including extra keys."""
        if key in _CHECKPOINT_RECORD_KNOWN_KEYS:
            return getattr(self, key) is not None
        return key in self.extra


# Derived from the dataclass fields so it stays in sync automatically.
# Excludes "extra" since that's the catch-all, not a serialized key.
_CHECKPOINT_RECORD_KNOWN_KEYS = frozenset(
    f.name for f in dataclasses.fields(CheckpointRecord) if f.name != "extra"
)


def add_renderer_name_to_user_metadata(
    user_metadata: dict[str, str], renderer_name: str | None
) -> None:
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


def resolve_renderer_name_from_checkpoint_or_default(
    *,
    model_name: str,
    explicit_renderer_name: str | None,
    load_checkpoint_path: str | None,
    base_url: str | None = None,
) -> str:
    """
    Resolve renderer name for training/eval setup.

    Precedence:
    1) explicit renderer name, if provided
    2) renderer metadata from load checkpoint path, if available
    3) recommended renderer for model_name
    """
    if explicit_renderer_name is not None:
        return explicit_renderer_name

    if load_checkpoint_path is not None:
        service_client = tinker.ServiceClient(base_url=base_url)
        renderer_name = get_renderer_name_from_checkpoint(service_client, load_checkpoint_path)
        if renderer_name is not None:
            logger.info(
                "Using renderer from checkpoint metadata for %s: %s",
                load_checkpoint_path,
                renderer_name,
            )
            return renderer_name

    return model_info.get_recommended_renderer_name(model_name)


async def resolve_renderer_name_from_checkpoint_or_default_async(
    *,
    model_name: str,
    explicit_renderer_name: str | None,
    load_checkpoint_path: str | None,
    base_url: str | None = None,
) -> str:
    """
    Async version of resolve_renderer_name_from_checkpoint_or_default.
    """
    if explicit_renderer_name is not None:
        return explicit_renderer_name

    if load_checkpoint_path is not None:
        service_client = tinker.ServiceClient(base_url=base_url)
        renderer_name = await get_renderer_name_from_checkpoint_async(
            service_client, load_checkpoint_path
        )
        if renderer_name is not None:
            logger.info(
                "Using renderer from checkpoint metadata for %s: %s",
                load_checkpoint_path,
                renderer_name,
            )
            return renderer_name

    return model_info.get_recommended_renderer_name(model_name)


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
def load_checkpoints_file(log_dir: str) -> list[CheckpointRecord]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return [CheckpointRecord.from_dict(d) for d in read_jsonl(checkpoint_path)]


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> CheckpointRecord | None:
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
    checkpoints_with_key = [c for c in checkpoints if c.has(required_key)]
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
    """Save model checkpoint and append a record to ``checkpoints.jsonl``.

    Args:
        training_client: Training client to save from.
        name: Name for the checkpoint (used in the tinker:// path).
        log_path: Directory containing ``checkpoints.jsonl``.
        loop_state: Training loop state (must include ``batch``; may include
            ``epoch``, ``final``, and any additional user metadata).
        kind: Which checkpoint types to save.
        ttl_seconds: Server-side retention. ``None`` keeps the checkpoint indefinitely.

    Returns:
        Dict mapping ``"state_path"`` and/or ``"sampler_path"`` to tinker:// paths.
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

    record = CheckpointRecord.from_dict({"name": name, **loop_state, **paths})
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")

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
