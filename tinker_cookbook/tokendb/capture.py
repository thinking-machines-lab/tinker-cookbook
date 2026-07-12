"""Capture layer: convert rollout :class:`TrajectoryGroup`\\ s into token DB rows.

:func:`record_groups` is the funnel between the RL rollout pipeline and the
token DB writer. It iterates groups/trajectories/transitions with the same
``group_idx`` / ``traj_idx`` / ``step_idx`` enumeration as
``rl/rollout_logging.serialize_rollout_summaries``, so token DB keys line up
1:1 with the rollout-summary JSONL records.

Also here:

- :data:`capture_context` / :func:`set_capture_context`: a ContextVar carrying
  split/iteration identity for capture sites that don't receive it as
  arguments (filtered-group and sample sinks, wired in a later step).
- :func:`set_filtered_group_sink` / :func:`get_filtered_group_sink`: registry
  for a callback invoked when the rollout pipeline drops a group (constant
  reward, rollout error) before it reaches the main export funnel. Follows
  the ``set_rollout_executor`` pattern in ``rl/rollouts.py``.
- :func:`sample_to_row` / :func:`capture_samples`: capture of raw samples
  made through the Tinker completers (``source="sample"`` rows), via the
  ``set_sample_sink`` registry in ``completers.py``.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import tinker

from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.tokendb.interface import TokenWriter
from tinker_cookbook.tokendb.schema import TokenRow, compute_ob_delta

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsDecode(Protocol):
    """Minimal tokenizer surface needed for text denormalization."""

    def decode(self, token_ids: list[int]) -> str: ...


def extract_ob_tokens(ob: tinker.ModelInput) -> tuple[list[int], bool]:
    """Extract text-chunk token IDs from an observation; flag non-text chunks.

    ``ModelInput.to_ints()`` raises on non-text chunks (images), so this
    collects tokens from ``EncodedTextChunk``\\ s only and reports whether any
    non-text chunk was present. Never raises.

    Returns:
        ``(tokens, has_images)`` — text-chunk token IDs in order, and whether
        the observation contained any non-text (image) chunk.
    """
    tokens: list[int] = []
    has_images = False
    for chunk in ob.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            tokens.extend(chunk.tokens)
        else:
            has_images = True
    return tokens, has_images


def _decode(tokenizer: SupportsDecode, tokens: list[int]) -> str | None:
    """Best-effort decode; text columns are a convenience, never fatal."""
    try:
        return tokenizer.decode(tokens)
    except Exception:
        return None


def _route_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str], str | None]:
    """Split builder metadata into (metrics, attrs, row_id) destinations.

    Numeric values (int / float, including bool) route to the ``metrics``
    map, everything else is stringified into ``attrs``. The reserved key
    ``"row_id"`` promotes to the ``env_row_id`` column.
    """
    meta_metrics: dict[str, Any] = {}
    meta_attrs: dict[str, str] = {}
    row_id: str | None = None
    for key, value in metadata.items():
        if key == "row_id":
            row_id = str(value)
        elif isinstance(value, (int, float)):
            meta_metrics[str(key)] = value
        else:
            meta_attrs[str(key)] = str(value)
    return meta_metrics, meta_attrs, row_id


def _coerce_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize transition tool-call records for the ``tool_calls`` column.

    Fills struct defaults so a partially populated record can never fail the
    typed-column encode in the background flush thread.
    """
    return [
        {
            "name": str(tc.get("name", "")),
            "args_json": str(tc.get("args_json", "")),
            "error_type": (str(tc["error_type"]) if tc.get("error_type") is not None else None),
            "should_stop": bool(tc.get("should_stop", False)),
        }
        for tc in tool_calls
    ]


def record_groups(
    writer: TokenWriter,
    groups: Sequence[TrajectoryGroup | RolloutSummaryGroup],
    *,
    split: str,
    iteration: int,
    sampling_client_step: int | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str | int | float] | None = None,
    tokenizer: SupportsDecode | None = None,
    store_text: bool = True,
    source: str = "rollout",
    filtered_reason: str | None = None,
) -> list[TokenRow]:
    """Emit one :class:`TokenRow` per :class:`~tinker_cookbook.rl.types.Transition`.

    Enumeration mirrors ``serialize_rollout_summaries``: ``group_idx`` is the
    index into *groups*, ``traj_idx`` the index into ``trajectories_G``, and
    ``step_idx`` the index into ``transitions`` — so a token DB row and a
    rollout-summary record with equal keys describe the same turn.

    Per row:

    - ``ob_tokens`` are delta-encoded against the previous ob+ac of the same
      trajectory (:func:`~tinker_cookbook.tokendb.schema.compute_ob_delta`,
      mirroring ``rl/data_processing.trajectory_to_data``); image chunks set
      ``has_images`` and contribute no tokens.
    - ``ac_tokens`` / ``ac_logprobs`` / ``stop_reason`` come from the
      transition's ``ac``.
    - ``total_reward`` / ``final_reward`` are denormalized per trajectory.
    - ``metrics`` merges ``Transition.metrics`` with the trajectory's
      group-level metrics (``TrajectoryGroup.metrics_G``) under a ``group/``
      key prefix, denormalized onto every row of the trajectory (like
      ``final_reward``). Values are float-coerced at row construction
      (:func:`~tinker_cookbook.tokendb.schema.coerce_metrics`).
    - ``token_metrics`` (named per-token float arrays parallel to
      ``ac_tokens``: teacher logprobs, per-token KL, token-level rewards,
      per-token entropy) is passed through when a transition carries a
      ``token_metrics`` mapping; the built-in envs do not emit one yet.
    - Builder metadata (:meth:`EnvGroupBuilder.metadata`, carried on
      :class:`RolloutSummaryGroup` or the *metadata* kwarg) is written onto
      every row of the group: numeric values into ``metrics``, strings into
      ``attrs``, and the reserved ``row_id`` key into ``env_row_id``.
    - ``Transition.attrs`` merges into the row's ``attrs``; on a key
      collision the per-step value wins over the group-level one.
    - ``Transition.tool_calls`` maps into the structured ``tool_calls``
      column.
    - ``logs["env/row_id"]`` is promoted to the ``env_row_id`` column (kept
      for compatibility; an explicit metadata ``row_id`` overrides it).
    - ``ob_text`` / ``ac_text`` are decoded when *store_text* and *tokenizer*
      are given; for delta rows only the delta portion is decoded.

    Args:
        writer: Destination writer; rows are passed to ``append_rows``.
        groups: Trajectory groups, optionally pre-bundled as
            :class:`RolloutSummaryGroup` (whose ``tags`` / ``metadata`` /
            ``sampling_client_step`` then take precedence over the kwargs).
        split: Dataset split identifier (e.g. ``"train"``).
        iteration: Training iteration / batch index.
        sampling_client_step: Fallback sampling-client step for plain
            :class:`TrajectoryGroup` entries.
        tags: Fallback logging tags for plain :class:`TrajectoryGroup` entries.
        metadata: Fallback group metadata for plain :class:`TrajectoryGroup`
            entries.
        tokenizer: Anything with ``decode(list[int]) -> str``.
        store_text: Store decoded ``ob_text`` / ``ac_text`` columns.
        source: Row provenance (``"rollout"`` | ``"filtered"`` | ``"sample"``).
        filtered_reason: Why the group was dropped, for ``source="filtered"``.

    Returns:
        The rows appended to *writer* (for tests / chaining).
    """
    rows: list[TokenRow] = []
    for group_idx, entry in enumerate(groups):
        if isinstance(entry, RolloutSummaryGroup):
            trajectory_group = entry.trajectory_group
            group_tags = list(entry.tags)
            group_step = entry.sampling_client_step
            group_metadata = entry.metadata
        else:
            trajectory_group = entry
            group_tags = list(tags)
            group_step = sampling_client_step
            group_metadata = metadata or {}
        meta_metrics, meta_attrs, meta_row_id = _route_metadata(group_metadata)
        total_rewards_G = trajectory_group.get_total_rewards()

        metrics_G = trajectory_group.metrics_G

        for traj_idx, trajectory in enumerate(trajectory_group.trajectories_G):
            group_metrics = {
                f"group/{key}": value
                for key, value in (metrics_G[traj_idx] if traj_idx < len(metrics_G) else {}).items()
            }
            prev_sequence: list[int] = []
            for step_idx, transition in enumerate(trajectory.transitions):
                ob_tokens, has_images = extract_ob_tokens(transition.ob)
                stored_ob, ob_is_delta = compute_ob_delta(prev_sequence, ob_tokens)
                ac_tokens = list(transition.ac.tokens)
                prev_sequence = ob_tokens + ac_tokens
                # Third-party env adapters (e.g. verifiers) can leave the stop
                # reason unset despite the declared type; store NULL rather
                # than fail capture.
                stop_reason: Any = transition.ac.stop_reason
                # Forward-compat plumb: Transition does not declare a
                # token_metrics field yet, but a transition (or subclass)
                # carrying one gets its per-token arrays captured.
                token_metrics = getattr(transition, "token_metrics", None)

                ob_text: str | None = None
                ac_text: str | None = None
                if store_text and tokenizer is not None:
                    ob_text = _decode(tokenizer, stored_ob)
                    ac_text = _decode(tokenizer, ac_tokens)

                # Explicit metadata row_id wins over the logs["env/row_id"]
                # fallback (kept for compatibility).
                env_row_id: Any = meta_row_id
                if env_row_id is None:
                    env_row_id = transition.logs.get("env/row_id")
                transition_attrs = getattr(transition, "attrs", None) or {}
                transition_tool_calls = getattr(transition, "tool_calls", None)
                rows.append(
                    TokenRow(
                        split=split,
                        iteration=iteration,
                        group_idx=group_idx,
                        traj_idx=traj_idx,
                        step_idx=step_idx,
                        ob_tokens=stored_ob,
                        ob_is_delta=ob_is_delta,
                        ac_tokens=ac_tokens,
                        ac_logprobs=(
                            list(transition.ac.maybe_logprobs)
                            if transition.ac.maybe_logprobs is not None
                            else None
                        ),
                        stop_reason=str(stop_reason) if stop_reason is not None else None,
                        has_images=has_images,
                        reward=transition.reward,
                        episode_done=transition.episode_done,
                        total_reward=total_rewards_G[traj_idx],
                        final_reward=trajectory_group.final_rewards_G[traj_idx],
                        ob_text=ob_text,
                        ac_text=ac_text,
                        metrics={**meta_metrics, **transition.metrics, **group_metrics},
                        # Per-step attrs win over group-level metadata on collision.
                        attrs={**meta_attrs, **transition_attrs},
                        token_metrics=dict(token_metrics) if token_metrics else {},
                        tool_calls=(
                            _coerce_tool_calls(transition_tool_calls)
                            if transition_tool_calls
                            else None
                        ),
                        logs=dict(transition.logs),
                        env_row_id=str(env_row_id) if env_row_id is not None else None,
                        sampling_client_step=group_step,
                        tags=group_tags,
                        source=source,
                        filtered_reason=filtered_reason,
                    )
                )
    writer.append_rows(rows)
    return rows


# ---------------------------------------------------------------------------
# Active capture — the writer handle registered by the training loop.
# Registered in rl/train.py main() when Config.token_db is set; the export
# funnel and the filtered-group sink read it instead of threading the writer
# through every call site.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveCapture:
    """The writer (plus decode settings) registered for the current run."""

    writer: TokenWriter
    tokenizer: SupportsDecode | None = None
    store_text: bool = True


_active_capture: ContextVar[ActiveCapture | None] = ContextVar("active_capture", default=None)


def set_active_capture(capture: ActiveCapture | None) -> None:
    """Register the active capture (``None`` disables; the default)."""
    _active_capture.set(capture)


def get_active_capture() -> ActiveCapture | None:
    """Return the active capture, or ``None`` when capture is disabled."""
    return _active_capture.get()


def record_groups_to_active_capture(
    groups: Sequence[TrajectoryGroup | RolloutSummaryGroup],
    *,
    split: str,
    iteration: int,
    sampling_client_step: int | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str | int | float] | None = None,
    source: str = "rollout",
    filtered_reason: str | None = None,
) -> None:
    """:func:`record_groups` against the active capture; never raises.

    No-op when no capture is registered. Capture failures are logged and
    swallowed — token DB capture must never break training.
    """
    active = get_active_capture()
    if active is None:
        return
    try:
        record_groups(
            active.writer,
            groups,
            split=split,
            iteration=iteration,
            sampling_client_step=sampling_client_step,
            tags=tags,
            metadata=metadata,
            tokenizer=active.tokenizer,
            store_text=active.store_text,
            source=source,
            filtered_reason=filtered_reason,
        )
    except Exception:
        logger.exception("Token DB capture failed; continuing without capture")


# ---------------------------------------------------------------------------
# Capture context — identity for capture sites without explicit arguments
# (filtered-group sink, sample sink), set around the rollout calls in
# rl/train.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureContext:
    """Identity carried to capture sites via :data:`capture_context`."""

    split: str
    iteration: int
    sampling_client_step: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


capture_context: ContextVar[CaptureContext | None] = ContextVar("capture_context", default=None)


@contextmanager
def set_capture_context(context: CaptureContext) -> Iterator[CaptureContext]:
    """Set :data:`capture_context` for the duration of the ``with`` block."""
    token = capture_context.set(context)
    try:
        yield context
    finally:
        capture_context.reset(token)


def get_capture_context() -> CaptureContext | None:
    """Return the active :class:`CaptureContext`, or ``None`` when unset."""
    return capture_context.get()


# ---------------------------------------------------------------------------
# Filtered-group sink — registry only; call sites in rl/rollouts.py are wired
# in a later step. Same pattern as set_rollout_executor in rl/rollouts.py.
# ---------------------------------------------------------------------------

FilteredGroupSink = Callable[[TrajectoryGroup | None, list[str], str], None]
"""Callback ``(trajectory_group_or_None, tags, reason)`` invoked when the
rollout pipeline drops a group before export (constant reward, rollout
error). ``trajectory_group_or_None`` is ``None`` when the group failed before
any trajectory was produced."""

_filtered_group_sink: ContextVar[FilteredGroupSink | None] = ContextVar(
    "filtered_group_sink", default=None
)


def set_filtered_group_sink(sink: FilteredGroupSink | None) -> None:
    """Set the sink for dropped trajectory groups.

    Pass ``None`` to disable (the default). Follows the
    ``set_rollout_executor`` pattern in ``rl/rollouts.py``.
    """
    _filtered_group_sink.set(sink)


def get_filtered_group_sink() -> FilteredGroupSink | None:
    """Return the active filtered-group sink, or ``None`` when unset."""
    return _filtered_group_sink.get()


def active_capture_filtered_sink(
    trajectory_group: TrajectoryGroup | None, tags: list[str], reason: str
) -> None:
    """Filtered-group sink that records dropped groups to the active capture.

    Registered via :func:`set_filtered_group_sink` when the token DB is
    enabled with ``capture_filtered=True``. Split/iteration identity comes
    from :data:`capture_context` (set around the rollout calls in
    ``rl/train.py``); when unset, rows land under ``iteration=-1``. Groups
    that failed before producing any trajectory (``trajectory_group is
    None``) have no tokens to record and are skipped.
    """
    if trajectory_group is None:
        return
    ctx = get_capture_context()
    record_groups_to_active_capture(
        [trajectory_group],
        split=ctx.split if ctx is not None else "train",
        iteration=ctx.iteration if ctx is not None else -1,
        sampling_client_step=ctx.sampling_client_step if ctx is not None else None,
        tags=tags,
        source="filtered",
        filtered_reason=reason,
    )


# ---------------------------------------------------------------------------
# Sample capture — raw completer samples as source="sample" rows, via the
# set_sample_sink registry in completers.py. Purely opt-in: the RL train
# loop does NOT register this (rollout tokens already reach the DB through
# record_groups; a sample sink there would duplicate them).
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsSampledSequence(Protocol):
    """Minimal sampled-sequence surface needed by :func:`sample_to_row`.

    Matched by ``tinker.SampledSequence`` and by
    :class:`~tinker_cookbook.completers.TokensWithLogprobs` (whose
    ``maybe_logprobs`` is preferred when present, since its ``logprobs``
    property raises when logprobs are absent).
    """

    @property
    def tokens(self) -> list[int]: ...

    @property
    def stop_reason(self) -> Any: ...


def _sequence_logprobs(sequence: SupportsSampledSequence) -> list[float] | None:
    """Best-effort logprob extraction across sampled-sequence flavors."""
    logprobs = getattr(sequence, "maybe_logprobs", None)
    if logprobs is None:
        try:
            logprobs = getattr(sequence, "logprobs", None)
        except Exception:
            logprobs = None
    return list(logprobs) if logprobs is not None else None


def sample_to_row(
    model_input: tinker.ModelInput,
    sequence: SupportsSampledSequence,
    *,
    split: str = "sample",
    iteration: int = -1,
    group_idx: int = 0,
    traj_idx: int = 0,
    step_idx: int = 0,
    sampling_client_step: int | None = None,
    tags: Sequence[str] = (),
    tokenizer: SupportsDecode | None = None,
    store_text: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> TokenRow:
    """Build a ``source="sample"`` :class:`TokenRow` from one sampled sequence.

    The prompt's text-chunk tokens become ``ob_tokens`` (full, never
    delta-encoded; image chunks set ``has_images``), and the sampled tokens /
    logprobs / stop reason become the ``ac`` fields. *extra* lands in the
    row's ``extra`` JSON column.
    """
    ob_tokens, has_images = extract_ob_tokens(model_input)
    ac_tokens = list(sequence.tokens)
    ob_text: str | None = None
    ac_text: str | None = None
    if store_text and tokenizer is not None:
        ob_text = _decode(tokenizer, ob_tokens)
        ac_text = _decode(tokenizer, ac_tokens)
    stop_reason = sequence.stop_reason
    return TokenRow(
        split=split,
        iteration=iteration,
        group_idx=group_idx,
        traj_idx=traj_idx,
        step_idx=step_idx,
        ob_tokens=ob_tokens,
        ob_is_delta=False,
        ac_tokens=ac_tokens,
        ac_logprobs=_sequence_logprobs(sequence),
        stop_reason=str(stop_reason) if stop_reason is not None else None,
        has_images=has_images,
        ob_text=ob_text,
        ac_text=ac_text,
        extra=dict(extra or {}),
        sampling_client_step=sampling_client_step,
        tags=list(tags),
        source="sample",
    )


@contextmanager
def capture_samples(
    writer_or_active: TokenWriter | ActiveCapture,
    **metadata: Any,
) -> Iterator[None]:
    """Capture every completer sample in the ``with`` block to the token DB.

    Registers a sample sink (:func:`~tinker_cookbook.completers.set_sample_sink`)
    that converts each successful ``TinkerTokenCompleter`` /
    ``TinkerMessageCompleter`` sample into ``source="sample"`` rows on
    *writer_or_active* (an :class:`ActiveCapture` also supplies the tokenizer
    and ``store_text`` for text denormalization).

    Row identity: split / iteration / step / tags come from
    :data:`capture_context` when set (falling back to ``split="sample"``,
    ``iteration=-1``); ``group_idx`` is a per-``capture_samples`` counter over
    sample calls and ``traj_idx`` indexes the sequences within one call. The
    keyword *metadata* plus the completer's sampling metadata land in the
    ``extra`` column.

    The sink never raises (capture failures are logged), and the previous
    sink is restored on exit. Purely opt-in: nothing registers this
    automatically, including the RL train loop (whose rollout rows would
    otherwise be duplicated).
    """
    from tinker_cookbook import completers

    if isinstance(writer_or_active, ActiveCapture):
        writer = writer_or_active.writer
        tokenizer = writer_or_active.tokenizer
        store_text = writer_or_active.store_text
    else:
        writer = writer_or_active
        tokenizer = None
        store_text = True
    group_counter = itertools.count()

    def _sink(
        model_input: tinker.ModelInput,
        sequences: Sequence[Any],
        sink_metadata: dict[str, Any],
    ) -> None:
        try:
            ctx = get_capture_context()
            group_idx = next(group_counter)
            extra = {**metadata, **sink_metadata}
            rows = [
                sample_to_row(
                    model_input,
                    sequence,
                    split=ctx.split if ctx is not None else "sample",
                    iteration=ctx.iteration if ctx is not None else -1,
                    group_idx=group_idx,
                    traj_idx=traj_idx,
                    sampling_client_step=ctx.sampling_client_step if ctx is not None else None,
                    tags=ctx.tags if ctx is not None else (),
                    tokenizer=tokenizer,
                    store_text=store_text,
                    extra=extra,
                )
                for traj_idx, sequence in enumerate(sequences)
            ]
            writer.append_rows(rows)
        except Exception:
            logger.exception("Token DB sample capture failed; continuing without capture")

    previous = completers.get_sample_sink()
    completers.set_sample_sink(_sink)
    try:
        yield
    finally:
        completers.set_sample_sink(previous)
