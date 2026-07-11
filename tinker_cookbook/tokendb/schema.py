"""Arrow schema and row model for the token DB.

The token DB stores one row per :class:`~tinker_cookbook.rl.types.Transition`
(i.e. per turn): the raw prompt and sampled token IDs, per-token logprobs,
rewards, and identity keys (run/split/iteration/group/traj/step). Token IDs
are canonical; decoded text columns are a denormalized convenience.

pyarrow is imported lazily so that importing this module (e.g. from the
training path when the token DB is disabled) does not require it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

SCHEMA_VERSION = 1


def arrow_schema() -> pa.Schema:
    """Return the arrow schema for token DB rows (``schema_version = 1``).

    Imports pyarrow lazily; only called on the write/read paths, never at
    module import time.
    """
    import pyarrow as pa

    return pa.schema(
        [
            # Identity / writer context
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("run_attempt", pa.int32(), nullable=False),
            pa.field("writer_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("iteration", pa.int32(), nullable=False),
            pa.field("sampling_client_step", pa.int32(), nullable=True),
            pa.field("group_idx", pa.int32(), nullable=False),
            pa.field("traj_idx", pa.int32(), nullable=False),
            pa.field("step_idx", pa.int32(), nullable=False),
            pa.field("tags", pa.list_(pa.string()), nullable=False),
            pa.field("env_row_id", pa.string(), nullable=True),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            # Tokens (canonical)
            pa.field("ob_tokens", pa.list_(pa.int32()), nullable=False),
            pa.field("ob_is_delta", pa.bool_(), nullable=False),
            pa.field("ac_tokens", pa.list_(pa.int32()), nullable=False),
            pa.field("ac_logprobs", pa.list_(pa.float32()), nullable=True),
            pa.field("stop_reason", pa.string(), nullable=True),
            pa.field("has_images", pa.bool_(), nullable=False),
            # Outcomes
            pa.field("reward", pa.float32(), nullable=False),
            pa.field("episode_done", pa.bool_(), nullable=False),
            pa.field("total_reward", pa.float32(), nullable=False),
            pa.field("final_reward", pa.float32(), nullable=False),
            # Browsing (denormalized text, optional)
            pa.field("ob_text", pa.string(), nullable=True),
            pa.field("ac_text", pa.string(), nullable=True),
            # Extensible
            pa.field("metrics", pa.string(), nullable=False),  # JSON
            pa.field("logs", pa.string(), nullable=False),  # JSON
            pa.field("extra", pa.string(), nullable=False),  # JSON
            pa.field("filtered_reason", pa.string(), nullable=True),
        ]
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class TokenRow:
    """One token DB row: a single Transition (turn) of a trajectory.

    ``run_id`` / ``run_attempt`` / ``writer_id`` are stamped by the writer on
    append; callers only need to fill the per-row fields.
    """

    # Per-row identity
    split: str
    iteration: int
    group_idx: int
    traj_idx: int
    step_idx: int
    # Tokens (canonical)
    ob_tokens: list[int]
    ac_tokens: list[int]
    ob_is_delta: bool = False
    ac_logprobs: list[float] | None = None
    stop_reason: str | None = None
    has_images: bool = False
    # Outcomes
    reward: float = 0.0
    episode_done: bool = False
    total_reward: float = 0.0
    final_reward: float = 0.0
    # Browsing (denormalized text)
    ob_text: str | None = None
    ac_text: str | None = None
    # Extensible
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    filtered_reason: str | None = None
    # Context
    sampling_client_step: int | None = None
    tags: list[str] = field(default_factory=list)
    env_row_id: str | None = None
    source: str = "rollout"  # "rollout" | "filtered" | "sample"
    ts: datetime = field(default_factory=_utc_now)
    # Writer identity (stamped by TokenDbWriter.append_rows)
    run_id: str = ""
    run_attempt: int = 0
    writer_id: str = ""


def row_to_record(row: TokenRow) -> dict[str, Any]:
    """Convert a :class:`TokenRow` to a plain dict matching :func:`arrow_schema`.

    The ``metrics`` / ``logs`` / ``extra`` dicts are serialized to JSON
    strings (non-JSON-safe values fall back to ``str()``).
    """
    return {
        "run_id": row.run_id,
        "run_attempt": row.run_attempt,
        "writer_id": row.writer_id,
        "split": row.split,
        "iteration": row.iteration,
        "sampling_client_step": row.sampling_client_step,
        "group_idx": row.group_idx,
        "traj_idx": row.traj_idx,
        "step_idx": row.step_idx,
        "tags": list(row.tags),
        "env_row_id": row.env_row_id,
        "ts": row.ts,
        "source": row.source,
        "ob_tokens": list(row.ob_tokens),
        "ob_is_delta": row.ob_is_delta,
        "ac_tokens": list(row.ac_tokens),
        "ac_logprobs": list(row.ac_logprobs) if row.ac_logprobs is not None else None,
        "stop_reason": row.stop_reason,
        "has_images": row.has_images,
        "reward": row.reward,
        "episode_done": row.episode_done,
        "total_reward": row.total_reward,
        "final_reward": row.final_reward,
        "ob_text": row.ob_text,
        "ac_text": row.ac_text,
        "metrics": json.dumps(row.metrics, default=str),
        "logs": json.dumps(row.logs, default=str),
        "extra": json.dumps(row.extra, default=str),
        "filtered_reason": row.filtered_reason,
    }


def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def compute_ob_delta(prev_sequence: list[int], ob_tokens: list[int]) -> tuple[list[int], bool]:
    """Delta-encode an observation against the previous full sequence.

    Mirrors the prefix check in ``rl/data_processing.trajectory_to_data``:
    when ``prev_sequence`` (the previous observation + action tokens) is a
    prefix of ``ob_tokens``, only the new suffix is stored and
    ``ob_is_delta=True``. On a non-prefix reset (or the first step) the full
    observation is stored with ``ob_is_delta=False``.

    Args:
        prev_sequence: Token IDs of the full sequence so far (ob + ac of all
            previous steps in the trajectory). Empty for the first step.
        ob_tokens: Token IDs of the current observation.

    Returns:
        ``(stored_ob_tokens, ob_is_delta)``.
    """
    if prev_sequence and _is_prefix(prev_sequence, ob_tokens):
        return ob_tokens[len(prev_sequence) :], True
    return list(ob_tokens), False
