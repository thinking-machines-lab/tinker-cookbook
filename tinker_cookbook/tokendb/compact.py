"""Segment compaction for the token DB.

A long training run leaves many small parquet segments (one per writer flush).
Compaction coalesces them into fewer, larger segments sorted by
``(split, iteration, group_idx, traj_idx, step_idx)``, which speeds up reads
without changing a single row (``run_id`` / ``run_attempt`` / ``writer_id``
columns and all payload fields are preserved exactly; only the file layout
changes). The one exception is schema generation: v1 segments in a
mixed-generation store are normalized to the current (v2) shape — the same
load-time normalization the reader applies — so compacted stores are always
uniformly current-schema.

Crash-safe ordering (never loses data):

1. New segments are fully written under a fresh ``compact-...`` writer ID.
2. The new manifest (``manifest-compact-....jsonl``) is written.
3. Only then are the old segments deleted, followed by the old manifests.

A crash before step 3 completes leaves both old and new files, so readers see
duplicate rows but never missing ones. Recovery: delete the
``seg-compact-*`` segments and ``manifest-compact-*`` manifest from the
interrupted attempt, then re-run compaction.

Compaction must not race a live writer, so it refuses to run when any
manifest was modified within ``min_quiet_s`` seconds (a heuristic, not a
lock: writers append a manifest line at least every ``flush_interval_s``
while active, so a quiet manifest implies a stopped writer). Only compact
runs whose training process has exited.

CLI::

    python -m tinker_cookbook.tokendb.compact log_path=... \\
        [target_rows_per_segment=65536] [dry_run=True]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import chz

from tinker_cookbook.stores.storage import Storage, storage_from_uri
from tinker_cookbook.tokendb.schema import SCHEMA_VERSION
from tinker_cookbook.tokendb.writer import (
    SEGMENTS_DIR,
    TOKENS_DIR,
    _observed_keys,
    make_writer_id,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_ROWS_PER_SEGMENT = 65536
DEFAULT_MIN_QUIET_S = 60.0

SORT_KEYS = ("split", "iteration", "group_idx", "traj_idx", "step_idx")


class ActiveWriterError(RuntimeError):
    """A manifest was modified too recently; a writer may still be running."""


@dataclass(frozen=True)
class CompactionPlan:
    """What a compaction run would do (returned as-is under ``dry_run``)."""

    old_segments: list[str]  # names under tokens/segments/
    old_manifests: list[str]  # paths under tokens/
    n_rows: int
    new_segments: list[str]  # names the compacted segments will get
    new_manifest: str  # path under tokens/
    dry_run: bool


def _map_key_union(chunk: object, column: str) -> set[str]:
    """The union of map keys in a MAP column of an arrow table chunk.

    Arrow map arrays come back from ``to_pylist`` as lists of ``(key, value)``
    tuples.
    """
    keys: set[str] = set()
    for entries in chunk.column(column).to_pylist():  # type: ignore[attr-defined]
        keys.update(str(key) for key, _ in entries)
    return keys


def _list_segment_names(storage: Storage) -> list[str]:
    return sorted(name for name in storage.list_dir(SEGMENTS_DIR) if name.endswith(".parquet"))


def _list_manifest_paths(storage: Storage) -> list[str]:
    return sorted(
        f"{TOKENS_DIR}/{name}"
        for name in storage.list_dir(TOKENS_DIR)
        if name.startswith("manifest-") and name.endswith(".jsonl")
    )


def _check_no_active_writer(storage: Storage, manifests: list[str], min_quiet_s: float) -> None:
    """Refuse to compact when any manifest looks like it has a live writer.

    Heuristic: an active writer appends a manifest line at least every
    ``flush_interval_s`` (default 5s) while producing rows, so a manifest
    untouched for ``min_quiet_s`` implies its writer has stopped. This is
    not a lock; only run compaction on runs whose training process exited.
    """
    if min_quiet_s <= 0:
        return
    now = time.time()
    for manifest in manifests:
        stat = storage.stat(manifest)
        if stat is not None and now - stat.mtime < min_quiet_s:
            raise ActiveWriterError(
                f"{manifest} was modified {now - stat.mtime:.1f}s ago (< {min_quiet_s}s); "
                "a writer may still be active. Stop the run or pass a smaller min_quiet_s."
            )


def compact(
    storage_or_log_path: Storage | str | Path,
    *,
    target_rows_per_segment: int = DEFAULT_TARGET_ROWS_PER_SEGMENT,
    dry_run: bool = False,
    min_quiet_s: float = DEFAULT_MIN_QUIET_S,
) -> CompactionPlan:
    """Coalesce all token DB segments into fewer, sorted segments.

    Reads every segment (the directory listing is the source of truth),
    sorts all rows by :data:`SORT_KEYS`, and rewrites them as chunks of
    *target_rows_per_segment* rows under a fresh ``compact-...`` writer ID
    with a single new manifest. Old segments and manifests are deleted only
    after every new file is fully written (see the module docstring for the
    crash-safety argument).

    Args:
        storage_or_log_path: The run's ``Storage`` backend or ``log_path``.
        target_rows_per_segment: Rows per compacted segment file.
        dry_run: Plan only; read segments but write and delete nothing.
        min_quiet_s: Refuse when any manifest was modified more recently
            than this (active-writer heuristic); ``0`` disables the check.

    Returns:
        The executed (or planned, under *dry_run*) :class:`CompactionPlan`.

    Raises:
        ActiveWriterError: A manifest was modified within *min_quiet_s*.
        ValueError: No segments to compact, or a non-positive target size.
    """
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    if target_rows_per_segment <= 0:
        raise ValueError(f"target_rows_per_segment must be positive, got {target_rows_per_segment}")
    if isinstance(storage_or_log_path, (str, Path)):
        storage: Storage = storage_from_uri(str(storage_or_log_path))
    else:
        storage = storage_or_log_path

    old_segments = _list_segment_names(storage)
    if not old_segments:
        raise ValueError(f"No token DB segments found under {storage.url(SEGMENTS_DIR)}")
    old_manifests = _list_manifest_paths(storage)
    _check_no_active_writer(storage, old_manifests, min_quiet_s)

    from tinker_cookbook.tokendb.reader import _normalize_segment_table

    tables = []
    for name in old_segments:
        data = storage.read(f"{SEGMENTS_DIR}/{name}")
        # Normalize v1 segments to the v2 shape (same load-time normalization
        # the reader applies), so mixed-generation stores concat cleanly and
        # compaction always writes current-schema segments.
        tables.append(_normalize_segment_table(pq.read_table(io.BytesIO(data))))
    table = pa.concat_tables(tables).sort_by([(key, "ascending") for key in SORT_KEYS])
    n_rows = table.num_rows

    writer_id = f"compact-{make_writer_id()}"
    new_manifest = f"{TOKENS_DIR}/manifest-{writer_id}.jsonl"
    chunks = [
        table.slice(offset, target_rows_per_segment)
        for offset in range(0, n_rows, target_rows_per_segment)
    ]
    new_segments = [f"seg-{writer_id}-{seq:06d}.parquet" for seq in range(len(chunks))]

    plan = CompactionPlan(
        old_segments=old_segments,
        old_manifests=old_manifests,
        n_rows=n_rows,
        new_segments=new_segments,
        new_manifest=new_manifest,
        dry_run=dry_run,
    )
    if dry_run:
        return plan

    # 1. Write every new segment...
    manifest_lines: list[str] = []
    for name, chunk in zip(new_segments, chunks, strict=True):
        sink = io.BytesIO()
        pq.write_table(chunk, sink, compression="zstd")
        storage.write(f"{SEGMENTS_DIR}/{name}", sink.getvalue())
        iterations = chunk.column("iteration").to_pylist()
        timestamps = chunk.column("ts").to_pylist()
        # Recompute the observed-keys manifest fields from the chunk itself:
        # compaction deletes the old manifests, which are the reader's only
        # source for the per-run schema card.
        metrics_keys, metrics_truncated = _observed_keys(_map_key_union(chunk, "metrics"))
        attrs_keys, attrs_truncated = _observed_keys(_map_key_union(chunk, "attrs"))
        token_metrics_keys, token_metrics_truncated = _observed_keys(
            _map_key_union(chunk, "token_metrics")
        )
        tags, tags_truncated = _observed_keys(
            {tag for row_tags in chunk.column("tags").to_pylist() for tag in row_tags}
        )
        manifest_lines.append(
            json.dumps(
                {
                    "path": f"segments/{name}",
                    "n_rows": chunk.num_rows,
                    "min_iteration": min(iterations),
                    "max_iteration": max(iterations),
                    "min_ts": min(timestamps).isoformat(),
                    "max_ts": max(timestamps).isoformat(),
                    "run_attempt": max(chunk.column("run_attempt").to_pylist()),
                    "writer_id": writer_id,
                    "schema_version": SCHEMA_VERSION,
                    "compacted": True,
                    "metrics_keys": metrics_keys,
                    "attrs_keys": attrs_keys,
                    "token_metrics_keys": token_metrics_keys,
                    "tags": tags,
                    "keys_truncated": metrics_truncated
                    or attrs_truncated
                    or token_metrics_truncated
                    or tags_truncated,
                }
            )
        )
    # 2. ...then the new manifest...
    storage.write(new_manifest, ("\n".join(manifest_lines) + "\n").encode())
    # 3. ...and only then delete the old segments, then the old manifests.
    for name in old_segments:
        storage.remove(f"{SEGMENTS_DIR}/{name}")
    for manifest in old_manifests:
        storage.remove(manifest)
    storage.flush()
    logger.info(
        "Compacted %d segments (%d rows) into %d under writer %s",
        len(old_segments),
        n_rows,
        len(new_segments),
        writer_id,
    )
    return plan


@chz.chz
class Config:
    """CLI config: ``python -m tinker_cookbook.tokendb.compact log_path=...``."""

    log_path: str
    target_rows_per_segment: int = DEFAULT_TARGET_ROWS_PER_SEGMENT
    dry_run: bool = True  # pass dry_run=False to actually rewrite/delete files
    min_quiet_s: float = DEFAULT_MIN_QUIET_S


def run(config: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    plan = compact(
        config.log_path,
        target_rows_per_segment=config.target_rows_per_segment,
        dry_run=config.dry_run,
        min_quiet_s=config.min_quiet_s,
    )
    verb = "Would compact" if plan.dry_run else "Compacted"
    print(
        f"{verb} {len(plan.old_segments)} segments ({plan.n_rows} rows) "
        f"into {len(plan.new_segments)} segments of <= {config.target_rows_per_segment} rows"
    )
    if plan.dry_run:
        print("Dry run: nothing was written or deleted. Pass dry_run=False to compact.")


if __name__ == "__main__":
    chz.nested_entrypoint(run)
