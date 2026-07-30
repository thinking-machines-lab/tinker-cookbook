"""Capture-address parsing for proxy URL paths.

Chat-proxy clients (Claude Code, opencode, remote rollout processors) cannot
enter a ``capture(...)`` scope themselves, so the scope rides the base URL as
``/r/<key>/<value>/...`` path pairs in front of the API path, e.g.
``/r/run/sweep-3/iter/12/traj/0/v1/messages``. Short address keys map to the
reserved scope keys of :mod:`tinker_cookbook.capture.scope`; unknown keys are
preserved verbatim so callers can attach arbitrary extra dimensions.
"""

from __future__ import annotations

ADDRESS_KEY_MAP: dict[str, str] = {
    "run": "run_id",
    "attempt": "run_attempt",
    "split": "split",
    "iter": "iteration",
    "group": "group_idx",
    "traj": "traj_idx",
    "purpose": "purpose",
}

_INT_SCOPE_KEYS: frozenset[str] = frozenset({"run_attempt", "iteration", "group_idx", "traj_idx"})

# The capture store persists these coordinates as signed 32-bit columns, and
# wire ingestion is batch-atomic: an out-of-range value that slipped past the
# proxy would be rejected at ingest and could take unrelated captures in the
# same batch down with it. Reject at parse time instead.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def parse_address(path: str) -> dict[str, str | int]:
    """Parse ``key/value/key/value/...`` path segments into scope pairs.

    Args:
        path: The raw segment string captured after ``/r/`` (no leading or
            trailing slash required; empty segments are ignored).

    Returns:
        Scope pairs ready for ``capture(**pairs)``: short keys mapped to
        reserved scope keys, integer-typed reserved values converted to
        ``int``, unknown keys preserved as strings.

    Raises:
        ValueError: On an odd number of segments, or a non-integer or
            out-of-32-bit-range value for an integer-typed reserved key.
    """
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) % 2:
        raise ValueError(
            f"capture address must be /key/value pairs, got an odd number of segments: {segments}"
        )
    pairs: dict[str, str | int] = {}
    for raw_key, raw_value in zip(segments[::2], segments[1::2], strict=True):
        key = ADDRESS_KEY_MAP.get(raw_key, raw_key)
        if key in _INT_SCOPE_KEYS:
            try:
                value = int(raw_value)
            except ValueError:
                raise ValueError(
                    f"capture address key {raw_key!r} requires an integer value, got {raw_value!r}"
                ) from None
            if not _INT32_MIN <= value <= _INT32_MAX:
                raise ValueError(
                    f"capture address key {raw_key!r} value {value} is out of range "
                    "for a 32-bit integer"
                )
            pairs[key] = value
        else:
            pairs[key] = raw_value
    return pairs
