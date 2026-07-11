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
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

#: Largest integer exactly representable as a float32; exact ints beyond this
#: lose precision when stored in the ``metrics`` map (warned once per key).
FLOAT32_EXACT_INT_MAX = 2**24


def arrow_schema() -> pa.Schema:
    """Return the arrow schema for token DB rows (``schema_version = 2``).

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
            # Extensible (typed)
            pa.field("metrics", pa.map_(pa.string(), pa.float32()), nullable=False),
            pa.field("attrs", pa.map_(pa.string(), pa.string()), nullable=False),
            pa.field("token_metrics", pa.map_(pa.string(), pa.list_(pa.float32())), nullable=False),
            pa.field(
                "tool_calls",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("name", pa.string(), nullable=False),
                            pa.field("args_json", pa.string(), nullable=False),
                            pa.field("error_type", pa.string(), nullable=True),
                            pa.field("should_stop", pa.bool_(), nullable=False),
                        ]
                    )
                ),
                nullable=True,
            ),
            # Extensible (free-form JSON escape hatches)
            pa.field("logs", pa.string(), nullable=False),  # JSON
            pa.field("extra", pa.string(), nullable=False),  # JSON
            pa.field("filtered_reason", pa.string(), nullable=True),
        ]
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


# Keys already warned about, so each misbehaving key logs exactly once per
# process (capture runs per rollout batch; per-value warnings would flood).
_warned_uncoercible_metric_keys: set[str] = set()
_warned_int_precision_metric_keys: set[str] = set()


def coerce_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Coerce a metrics mapping to ``dict[str, float]`` for the typed map column.

    Runs at row-build time (in the caller's thread) so a bad value can never
    reach the background flush thread, where a typed-column encode failure
    would silently drop the whole segment. Policy:

    - Values go through ``float()``; non-coercible values are dropped with a
      warning (once per key per process).
    - ``NaN`` is a legal value and passes through as a real float NaN map
      entry (envs emit NaN meaningfully, e.g. "no parse to grade").
    - Exact ints above ``FLOAT32_EXACT_INT_MAX`` (2**24) are stored but warned
      about once per key: the float32 column cannot represent them exactly.
    """
    out: dict[str, float] = {}
    for key, value in metrics.items():
        key = str(key)
        try:
            coerced = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            if key not in _warned_uncoercible_metric_keys:
                _warned_uncoercible_metric_keys.add(key)
                logger.warning(
                    "Dropping non-float-coercible metrics value for key %r "
                    "(type %s); further drops for this key will not be logged",
                    key,
                    type(value).__name__,
                )
            continue
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and abs(value) > FLOAT32_EXACT_INT_MAX
            and key not in _warned_int_precision_metric_keys
        ):
            _warned_int_precision_metric_keys.add(key)
            logger.warning(
                "metrics key %r has integer value %d beyond float32 exact range "
                "(2**24); it will lose precision in the token DB",
                key,
                value,
            )
        out[key] = coerced
    return out


def coerce_attrs(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Coerce an attrs mapping to ``dict[str, str]`` for the typed map column."""
    return {str(key): str(value) for key, value in attrs.items()}


_warned_token_metric_keys: set[str] = set()


def coerce_token_metrics(
    token_metrics: Mapping[str, Any], n_ac_tokens: int
) -> dict[str, list[float]]:
    """Coerce a token-metrics mapping to ``dict[str, list[float]]``.

    Each entry is a named per-token float array **parallel to** ``ac_tokens``
    (one value per sampled action token). Like :func:`coerce_metrics`, this
    runs at row-build time so a bad value can never fail the encode in the
    background flush thread. Policy:

    - Every element goes through ``float()``; an array with any non-coercible
      element is dropped whole (a partially coerced array would silently
      break token alignment), with a warning once per key per process.
    - ``NaN`` elements are legal and pass through as real float NaNs.
    - Arrays whose length differs from ``n_ac_tokens`` are dropped with a
      warning (once per key per process) naming the key and both lengths —
      misaligned arrays are worse than missing ones.
    """
    out: dict[str, list[float]] = {}
    for key, values in token_metrics.items():
        key = str(key)
        try:
            coerced = [float(value) for value in values]
        except (TypeError, ValueError):
            if key not in _warned_token_metric_keys:
                _warned_token_metric_keys.add(key)
                logger.warning(
                    "Dropping token_metrics array for key %r: it contains a "
                    "non-float-coercible element; further drops for this key "
                    "will not be logged",
                    key,
                )
            continue
        if len(coerced) != n_ac_tokens:
            if key not in _warned_token_metric_keys:
                _warned_token_metric_keys.add(key)
                logger.warning(
                    "Dropping token_metrics array for key %r: length %d does "
                    "not match len(ac_tokens) == %d; further drops for this "
                    "key will not be logged",
                    key,
                    len(coerced),
                    n_ac_tokens,
                )
            continue
        out[key] = coerced
    return out


@dataclass
class TokenRow:
    """One token DB row: a single Transition (turn) of a trajectory.

    ``run_id`` / ``run_attempt`` / ``writer_id`` are stamped by the writer on
    append; callers only need to fill the per-row fields.

    Extensible fields:

    - ``metrics`` (``map<string, float32>``): numeric per-row values —
      ``Transition.metrics`` keys, group-level metrics under a ``group/``
      prefix (denormalized onto every row of the trajectory, like
      ``final_reward``), and per-turn scalar tool aggregates under ``tool/``.
      Values are coerced via :func:`coerce_metrics` at construction. Numeric
      dimensions (difficulty, level, counts) belong here even when they are
      used as dimensions rather than verdicts.
    - ``attrs`` (``map<string, string>``): categorical dimensions (dataset,
      task name, player id, ...). Strings only; values are coerced via
      :func:`coerce_attrs` at construction.
    - ``token_metrics`` (``map<string, list<float32>>``): named per-token
      float arrays parallel to ``ac_tokens`` (one value per sampled action
      token). This is the channel for on-policy distillation teacher logprobs
      (``teacher/logprobs``; multi-teacher as ``teacher/<name>/logprobs``),
      per-token KL, token-level rewards/advantages, and per-token entropy.
      Arrays are coerced via :func:`coerce_token_metrics` at construction;
      arrays whose length does not match ``len(ac_tokens)`` are dropped with
      a warning. Empty by default.
    - ``tool_calls``: structured per-turn tool calls (``name`` / ``args_json``
      / ``error_type`` / ``should_stop``). No result payload: tool results are
      already part of the next turn's observation.
    - ``logs`` / ``extra``: free-form JSON escape hatches (serialized at
      encode time with a ``str()`` fallback).
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
    # Extensible (typed; values coerced in __post_init__ — metrics to float,
    # attrs to str, token_metrics to length-checked float arrays)
    metrics: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    token_metrics: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    # Extensible (free-form JSON)
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

    def __post_init__(self) -> None:
        # Coerce the typed map columns at construction (the caller's thread),
        # so a bad value can never fail the encode in the flush thread.
        self.metrics = coerce_metrics(self.metrics)
        self.attrs = coerce_attrs(self.attrs)
        self.token_metrics = coerce_token_metrics(self.token_metrics, len(self.ac_tokens))


def row_to_record(row: TokenRow) -> dict[str, Any]:
    """Convert a :class:`TokenRow` to a plain dict matching :func:`arrow_schema`.

    ``metrics`` / ``attrs`` / ``token_metrics`` pass through as typed maps
    (already coerced at row construction); the ``logs`` / ``extra`` dicts are
    serialized to JSON strings (non-JSON-safe values fall back to ``str()``).
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
        "metrics": dict(row.metrics),
        "attrs": dict(row.attrs),
        "token_metrics": dict(row.token_metrics),
        "tool_calls": row.tool_calls,
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
