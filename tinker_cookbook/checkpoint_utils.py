import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import tinker
from fireworks.training.sdk import FiretitanServiceClient, FiretitanTrainingClient

from tinker_cookbook import model_info
from tinker_cookbook.utils import trace
from tinker_cookbook.utils.file_utils import read_jsonl

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)
RENDERER_NAME_METADATA_KEY = "renderer_name"


_MISSING = object()  # sentinel for distinguishing "not set" from None


@dataclass
class CheckpointRecord:
    """A single checkpoint record stored in ``checkpoints.jsonl``.

    Known fields are exposed as typed attributes.  ``batch`` is optional so
    that checkpoint files written by older code (or external tools that use
    different progress counters) can still be loaded.

    Any additional user-supplied metadata from ``loop_state`` is preserved in
    :attr:`extra` so that custom keys round-trip through save/load without
    loss.
    """

    name: str
    batch: int | None = None
    epoch: int | None = None
    final: bool | None = None
    state_path: str | None = None
    sampler_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive: if extra accidentally contains a known key (e.g. via
        # direct construction), drop it so to_dict() never double-writes.
        overlap = set(self.extra) & _CHECKPOINT_RECORD_KNOWN_KEYS
        if overlap:
            logger.warning("CheckpointRecord: dropping known keys from extra: %s", overlap)
            self.extra = {k: v for k, v in self.extra.items() if k not in overlap}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage.

        Omits ``None`` optional fields. Extra metadata keys are merged
        into the top-level dict.

        Returns:
            dict[str, Any]: JSON-serializable dict with known fields and
                any extra metadata.
        """
        d: dict[str, Any] = {"name": self.name}
        if self.batch is not None:
            d["batch"] = self.batch
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
        """Deserialize from a JSON-parsed dict.

        Unknown keys are preserved in :attr:`extra` so that downstream
        metadata (e.g. ``step``) round-trips without loss.

        Args:
            d (dict[str, Any]): Dict with at least a ``"name"`` key.

        Returns:
            CheckpointRecord: Reconstructed record.
        """
        return cls(
            name=d["name"],
            batch=d.get("batch"),
            epoch=d.get("epoch"),
            final=d.get("final"),
            state_path=d.get("state_path"),
            sampler_path=d.get("sampler_path"),
            extra={k: v for k, v in d.items() if k not in _CHECKPOINT_RECORD_KNOWN_KEYS},
        )

    def has(self, key: str) -> bool:
        """Check whether a field is present (not None), including extra keys.

        Args:
            key (str): Field name to check (known attribute or extra key).

        Returns:
            bool: True if the field exists and is not ``None``.
        """
        if key in _CHECKPOINT_RECORD_KNOWN_KEYS:
            return getattr(self, key) is not None
        return key in self.extra

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Get a field value by name, falling back to extra, then *default*.

        This provides uniform access regardless of whether a key is a known
        attribute or user-supplied metadata stored in :attr:`extra`.

        For known fields, returns the attribute value (which may be ``None``
        if the field is optional and unset).  Returns *default* only when the
        key is not a known field **and** is absent from :attr:`extra`.

        Args:
            key (str): Field name to look up (known attribute or extra key).
            default (Any): Value to return if ``key`` is not a known field
                and is absent from :attr:`extra`. If omitted, returns
                ``None`` for missing extra keys.

        Returns:
            Any: The field value, extra value, or *default*.
        """
        if key in _CHECKPOINT_RECORD_KNOWN_KEYS:
            return getattr(self, key)
        if default is _MISSING:
            return self.extra.get(key)
        return self.extra.get(key, default)


# Derived from the dataclass fields so it stays in sync automatically.
# Excludes "extra" since that's the catch-all, not a serialized key.
_CHECKPOINT_RECORD_KNOWN_KEYS = frozenset(
    f.name for f in dataclasses.fields(CheckpointRecord) if f.name != "extra"
)


def add_renderer_name_to_user_metadata(
    user_metadata: dict[str, str], renderer_name: str | None
) -> None:
    """Attach renderer name to training-run metadata when available.

    Args:
        user_metadata (dict[str, str]): Mutable metadata dict to update
            in-place.
        renderer_name (str | None): Renderer name to store, or ``None``
            to skip.
    """
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
    """Read renderer_name metadata from the training run referenced by a checkpoint path.

    Args:
        service_client (tinker.ServiceClient): Tinker service client for
            API access.
        checkpoint_path (str): Tinker checkpoint path
            (e.g. ``"tinker://run-id/sampler_weights/final"``).

    Returns:
        str | None: The renderer name if present in the run metadata,
            otherwise ``None``.
    """
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
    """Async version of :func:`get_renderer_name_from_checkpoint`.

    Args:
        service_client (tinker.ServiceClient): Tinker service client for
            API access.
        checkpoint_path (str): Tinker checkpoint path.

    Returns:
        str | None: The renderer name if present, otherwise ``None``.
    """
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
    """Resolve renderer name for training/eval setup.

    Precedence:
        1. ``explicit_renderer_name``, if provided.
        2. Renderer metadata from ``load_checkpoint_path``, if available.
        3. Recommended renderer for ``model_name``.

    Args:
        model_name (str): HuggingFace model identifier.
        explicit_renderer_name (str | None): User-specified renderer, or
            ``None`` to auto-detect.
        load_checkpoint_path (str | None): Tinker checkpoint path to read
            renderer metadata from, or ``None``.
        base_url (str | None): Custom Tinker service URL. Defaults to the
            standard endpoint.

    Returns:
        str: Resolved renderer name.
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
    """Async version of :func:`resolve_renderer_name_from_checkpoint_or_default`.

    Args:
        model_name (str): HuggingFace model identifier.
        explicit_renderer_name (str | None): User-specified renderer, or
            ``None`` to auto-detect.
        load_checkpoint_path (str | None): Tinker checkpoint path to read
            renderer metadata from, or ``None``.
        base_url (str | None): Custom Tinker service URL.

    Returns:
        str: Resolved renderer name.
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
    """Inspect a checkpoint's training run metadata and compare renderer name.

    Logs info if the renderer matches or metadata is missing, and logs a
    warning on mismatch. No-ops if ``expected_renderer_name`` is ``None``.

    Args:
        service_client (tinker.ServiceClient): Tinker service client.
        checkpoint_path (str): Tinker checkpoint path.
        expected_renderer_name (str | None): Expected renderer name, or
            ``None`` to skip the check.
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
    """Async version of :func:`check_renderer_name_for_checkpoint`.

    Compares an expected renderer with renderer metadata attached to a
    checkpoint's training run.

    - If ``expected_renderer_name`` is ``None``, returns immediately.
    - Logs info if metadata is missing or matches.
    - Logs warning if the checkpoint renderer differs from expected.

    Args:
        service_client (tinker.ServiceClient): Tinker service client.
        checkpoint_path (str): Tinker checkpoint path.
        expected_renderer_name (str | None): Expected renderer name, or
            ``None`` to skip the check.
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


@trace.scope
def load_checkpoints_file(log_dir: str) -> list[CheckpointRecord]:
    """Load checkpoint records from a JSONL file.

    Args:
        log_dir: Directory containing the ``checkpoints.jsonl`` file.

    Returns:
        A list of CheckpointRecord instances, or an empty list if the file does not exist.
    """
    checkpoint_path = Path(log_dir) / CHECKPOINTS_BASE_NAME
    if not checkpoint_path.exists():
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    trace.update_scope_context({"checkpoint_path": str(checkpoint_path)})
    return [CheckpointRecord.from_dict(d) for d in read_jsonl(str(checkpoint_path))]


@trace.scope
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


@trace.scope
async def save_checkpoint_async(
    training_client: FiretitanTrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
    store: "TrainingRunStore | None" = None,
) -> dict[str, str]:
    """Save model checkpoint and append a record to ``checkpoints.jsonl``.

    Args:
        training_client: Training client to save from.
        name: Name for the checkpoint (used in the tinker:// path).
        log_path: Directory containing ``checkpoints.jsonl``.
        loop_state: Training loop state. May include ``batch``, ``step``,
            ``epoch``, ``final``, and any additional user metadata.
        kind: Which checkpoint types to save.
        ttl_seconds: Server-side retention. ``None`` keeps the checkpoint indefinitely.
        store: If provided, write the checkpoint record via Storage protocol.

    Returns:
        Dict mapping ``"state_path"`` and/or ``"sampler_path"`` to tinker:// paths.
    """
    state_name = f"{name}-state"
    sampler_name = f"{name}-sampler"
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(state_name, ttl_seconds=ttl_seconds)
    sampler_snapshot_name = None
    if kind in ["sampler", "both"]:
        sampler_save_result = training_client.save_weights_for_sampler_ext(
            sampler_name,
        )
        sampler_snapshot_name = sampler_save_result.snapshot_name


    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    if sampler_snapshot_name:
        paths["sampler_path"] = sampler_snapshot_name
    trace.update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    record = CheckpointRecord.from_dict({"name": name, **loop_state, **paths})
    if store is not None:
        store.write_checkpoint(record.to_dict())
    else:
        with open(Path(log_path) / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    return paths


@trace.scope
def save_checkpoint(
    training_client: FiretitanTrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
    store: "TrainingRunStore | None" = None,
) -> dict[str, str]:
    """Save model checkpoint (synchronous wrapper around save_checkpoint_async).

    Args:
        training_client: Training client to save from.
        name: Name for the checkpoint (used in the tinker:// path).
        log_path: Directory containing ``checkpoints.jsonl``.
        loop_state: Training loop state dict (may include ``batch``, ``epoch``, etc.).
        kind: Which checkpoint types to save (``"state"``, ``"sampler"``, or ``"both"``).
        ttl_seconds: Server-side retention. ``None`` keeps the checkpoint indefinitely.
        store: If provided, write the checkpoint record via Storage protocol.

    Returns:
        Dict mapping ``"state_path"`` and/or ``"sampler_path"`` to tinker:// paths.

    Example::

        save_checkpoint(
            training_client=training_client,
            name="step-100",
            log_path="./logs",
            loop_state={"epoch": 0, "batch": 100},
        )
    """
    return asyncio.run(
        save_checkpoint_async(
            training_client,
            name=name,
            log_path=log_path,
            kind=kind,
            loop_state=loop_state,
            ttl_seconds=ttl_seconds,
            store=store,
        )
    )


class CheckpointManager:
    """Unified checkpoint manager for periodic, rolling, and final checkpoints.

    Manages three kinds of checkpoints:

    * **Periodic** – full checkpoints (state + sampler weights) saved every
      ``save_every`` steps with a configurable TTL.  When
      ``async_periodic_saves=True``, these run as fire-and-forget background
      tasks so the training loop is not blocked; the checkpoint record is
      written to ``checkpoints.jsonl`` once the background save completes.
    * **Rolling** – cheap resume-only checkpoints (state only, no sampler
      export) saved every ``rolling_save_every`` steps.  After each save the
      previous rolling checkpoint is deleted to bound storage.  A short TTL
      acts as a safety net if deletion fails.
    * **Final** – a permanent checkpoint saved at the end of training with no
      TTL, followed by cleanup of any remaining rolling checkpoint.  The final
      checkpoint always blocks (never fire-and-forget).

    For training loops where periodic and rolling saves happen at the same
    point, use :meth:`maybe_save_async` / :meth:`maybe_save` which handles
    both in one call.  For loops where they happen at different points,
    call :meth:`should_save_periodic` + :meth:`save_periodic_async` and
    :meth:`maybe_save_rolling_async` separately.

    Usage::

        mgr = CheckpointManager(
            training_client=tc,
            service_client=sc,
            log_path="/tmp/logs",
            save_every=20,
            rolling_save_every=1,
        )
        for step in range(1, num_steps + 1):
            train_step(...)
            await mgr.maybe_save_async(step, {"batch": step})
        await mgr.save_final_async({"batch": num_steps})
    """

    def __init__(
        self,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        log_path: str,
        save_every: int = 0,
        ttl_seconds: int | None = 604800,
        rolling_save_every: int = 0,
        rolling_ttl_seconds: int = 7200,
        store: "TrainingRunStore | None" = None,
        async_periodic_saves: bool = False,
    ) -> None:
        self._training_client = training_client
        self._service_client = service_client
        self._log_path = log_path
        self._save_every = save_every
        self._ttl_seconds = ttl_seconds
        self._rolling_save_every = rolling_save_every
        self._rolling_ttl_seconds = rolling_ttl_seconds
        self._store = store
        self._async_periodic_saves = async_periodic_saves

        self._pending_rolling_task: asyncio.Task[None] | None = None
        self._pending_periodic_task: asyncio.Task[dict[str, str]] | None = None
        self._prev_state_path: str | None = None

    # ------------------------------------------------------------------
    # Periodic checkpoints
    # ------------------------------------------------------------------

    def should_save_periodic(self, step: int) -> bool:
        """Return True if *step* warrants a periodic checkpoint."""
        return self._save_every > 0 and step > 0 and step % self._save_every == 0

    async def save_periodic_async(self, step: int, loop_state: dict[str, Any]) -> dict[str, str]:
        """Save a periodic checkpoint (state + sampler) and return paths.

        Callers that need the returned ``sampler_path`` (e.g. to create a
        sampling client) should call :meth:`should_save_periodic` first, then
        this method directly.
        """
        async with trace.scope_span("save_checkpoint"):
            return await save_checkpoint_async(
                training_client=self._training_client,
                name=f"{step:06d}",
                log_path=self._log_path,
                loop_state=loop_state,
                kind="both",
                ttl_seconds=self._ttl_seconds,
                store=self._store,
            )

    def save_periodic(self, step: int, loop_state: dict[str, Any]) -> dict[str, str]:
        """Synchronous version of :meth:`save_periodic_async`."""
        with trace.scope_span_sync("save_checkpoint"):
            return save_checkpoint(
                training_client=self._training_client,
                name=f"{step:06d}",
                log_path=self._log_path,
                loop_state=loop_state,
                kind="both",
                ttl_seconds=self._ttl_seconds,
                store=self._store,
            )

    # ------------------------------------------------------------------
    # Rolling checkpoints
    # ------------------------------------------------------------------

    def _should_save_rolling(self, step: int) -> bool:
        """Return True if *step* warrants a rolling save."""
        if self._rolling_save_every <= 0 or step <= 0:
            return False
        if step % self._rolling_save_every != 0:
            return False
        # Skip when a periodic checkpoint fires on the same step.
        return not (self._save_every > 0 and step % self._save_every == 0)

    async def maybe_save_rolling_async(self, step: int, loop_state: dict[str, Any]) -> None:
        """Resolve any pending rolling save, then fire a new one if *step* warrants it.

        The save runs as a background ``asyncio.Task`` so it overlaps with the
        next training step.
        """
        await self._resolve_pending_rolling_async()

        if not self._should_save_rolling(step):
            return

        self._pending_rolling_task = asyncio.create_task(
            self._do_rolling_save_async(step, loop_state),
            name=f"rolling_checkpoint_{step:06d}",
        )

    def maybe_save_rolling(self, step: int, loop_state: dict[str, Any]) -> None:
        """Synchronous version of :meth:`maybe_save_rolling_async`.

        Blocks on the save but catches all errors so it never crashes the
        training loop.
        """
        if not self._should_save_rolling(step):
            return
        try:
            asyncio.run(self._do_rolling_save_async(step, loop_state))
        except Exception:
            logger.warning("Rolling checkpoint save failed", exc_info=True)

    # ------------------------------------------------------------------
    # Combined convenience (periodic + rolling in one call)
    # ------------------------------------------------------------------

    async def maybe_save_async(
        self, step: int, loop_state: dict[str, Any]
    ) -> dict[str, str] | None:
        """Save periodic checkpoint if due, then fire rolling save if due.

        Returns the periodic checkpoint path dict if a periodic save happened
        synchronously, otherwise ``None``.  When ``async_periodic_saves`` is
        enabled, periodic saves run as background tasks and this always returns
        ``None``.

        Suitable for training loops where periodic and rolling saves happen at
        the same point (SL, DPO).
        """
        result: dict[str, str] | None = None
        if self.should_save_periodic(step):
            if self._async_periodic_saves:
                await self._resolve_pending_periodic_async()
                self._pending_periodic_task = asyncio.create_task(
                    self.save_periodic_async(step, loop_state),
                    name=f"periodic_checkpoint_{step:06d}",
                )
            else:
                result = await self.save_periodic_async(step, loop_state)
        await self.maybe_save_rolling_async(step, loop_state)
        return result

    def maybe_save(self, step: int, loop_state: dict[str, Any]) -> dict[str, str] | None:
        """Synchronous version of :meth:`maybe_save_async`."""
        result: dict[str, str] | None = None
        if self.should_save_periodic(step):
            result = self.save_periodic(step, loop_state)
        self.maybe_save_rolling(step, loop_state)
        return result

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------

    async def save_final_async(self, loop_state: dict[str, Any]) -> dict[str, str]:
        """Save the final checkpoint (no TTL) and clean up rolling checkpoints.

        The final checkpoint is saved with ``ttl_seconds=None`` so it persists
        indefinitely.  After saving, any remaining rolling checkpoint is
        deleted so that the last entry in ``checkpoints.jsonl`` always points
        to valid server-side data.
        """
        # Drain any in-flight periodic save so its record lands in
        # checkpoints.jsonl *before* the final record.
        await self._resolve_pending_periodic_async()
        paths = await save_checkpoint_async(
            training_client=self._training_client,
            name="final",
            log_path=self._log_path,
            kind="both",
            loop_state=loop_state,
            ttl_seconds=None,
            store=self._store,
        )
        await self.finalize_async()
        return paths

    def save_final(self, loop_state: dict[str, Any]) -> dict[str, str]:
        """Synchronous version of :meth:`save_final_async`."""
        paths = save_checkpoint(
            training_client=self._training_client,
            name="final",
            log_path=self._log_path,
            kind="both",
            loop_state=loop_state,
            ttl_seconds=None,
            store=self._store,
        )
        self.finalize()
        return paths

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def finalize_async(self) -> None:
        """Await any pending saves, then delete the last rolling checkpoint.

        Called automatically by :meth:`save_final_async`.  Can also be called
        directly if the final checkpoint is saved through other means.
        """
        await self._resolve_pending_periodic_async()
        await self._resolve_pending_rolling_async()
        await self._delete_prev_async()

    def finalize(self) -> None:
        """Synchronous version of :meth:`finalize_async`."""
        try:
            asyncio.run(self._delete_prev_async())
        except Exception:
            logger.warning(
                "Failed to delete last rolling checkpoint (TTL will clean up)",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_pending_rolling_async(self) -> None:
        if self._pending_rolling_task is None:
            return
        try:
            await self._pending_rolling_task
        except Exception:
            logger.warning("Rolling checkpoint save failed", exc_info=True)
        self._pending_rolling_task = None

    async def _resolve_pending_periodic_async(self) -> None:
        if self._pending_periodic_task is None:
            return
        try:
            await self._pending_periodic_task
        except Exception:
            logger.warning("Background periodic checkpoint save failed", exc_info=True)
        self._pending_periodic_task = None

    async def _do_rolling_save_async(self, step: int, loop_state: dict[str, Any]) -> None:
        name = f"{step:06d}"
        paths = await save_checkpoint_async(
            training_client=self._training_client,
            name=name,
            log_path=self._log_path,
            loop_state={**loop_state, "rolling": True},
            kind="state",
            ttl_seconds=self._rolling_ttl_seconds,
            store=self._store,
        )
        new_state_path = paths.get("state_path")

        # Delete the previous rolling checkpoint now that the new one is saved.
        await self._delete_prev_async()

        self._prev_state_path = new_state_path

    async def _delete_prev_async(self) -> None:
        if self._prev_state_path is None:
            return
        try:
            rest_client = self._service_client.create_rest_client()
            await rest_client.delete_checkpoint_from_tinker_path_async(self._prev_state_path)
            logger.info("Deleted rolling checkpoint %s", self._prev_state_path)
        except Exception:
            logger.warning(
                "Failed to delete rolling checkpoint %s (TTL will clean up)",
                self._prev_state_path,
                exc_info=True,
            )
        self._prev_state_path = None
