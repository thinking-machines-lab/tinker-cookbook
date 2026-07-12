"""Cross-run DuckDB read path over every run in the token DB registry.

:class:`RegistryBackend` is the read half of the ``TokenStoreBackend``
protocol spanning *all* runs registered in the run registry
(:mod:`tinker_cookbook.tokendb.registry`): one DuckDB connection whose
``rollouts`` / ``rollouts_latest`` / ``trajectories`` / ``labels`` / ``runs``
views (plus the promoted ``correct`` / ``parse_errors`` /
``context_overflows``) contain every registered run's rows, with ``run_id``
as an ordinary column. Writes still go through the per-run
:class:`~tinker_cookbook.tokendb.writer.ParquetSegmentBackend`;
``open_writer`` raises.

Unlike the single-run :class:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader`
(which materializes every segment into an in-memory table), the cross-run
reader is **lazy**: it maintains a ``segment_scan`` view over an explicit
``read_parquet([...])`` file list rebuilt on :meth:`refresh` (TTL-gated), so
DuckDB streams row groups per query and nothing is pinned in RAM between
queries.

Segments that cannot be scanned in place are staged once into a local
**segcache**:

- Cloud stores (``gs://``, ``s3://``): each segment is fetched once through
  ``Storage`` and cached; segments are immutable, so cache validity is file
  existence (no invalidation).
- v1-shaped segments (``metrics`` as a JSON string) are normalized with the
  same read path the single-run reader uses
  (:func:`~tinker_cookbook.tokendb.reader._normalize_segment_table`) and the
  v2 copy is cached under ``upgraded/``; the original file is never
  rewritten.

Default cache location: ``~/.cache/tinker-cookbook/tokendb/segcache``
(override with the ``segcache_dir`` argument or the
``TINKER_TOKENDB_SEGCACHE`` environment variable). The cache is unbounded in
this version; it is safe to delete at any time (segments re-stage on the next
refresh).

Thread-safety: one shared DuckDB connection owns the view catalog; each
executing thread queries through its own cursor, and view replacement during
refresh is serialized with a lock. In-flight queries finish against the old
scan list (segments are immutable, so an old list is merely stale, never
wrong).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinker_cookbook.stores.storage import LocalStorage, Storage, storage_from_uri
from tinker_cookbook.tokendb.reader import (
    _TRAJECTORIES_SQL,
    DEFAULT_SUBSCRIBE_POLL_INTERVAL_S,
    LABEL_FILTER_FIELDS,
    LABELS_CAST_SELECT,
    LABELS_DEDUP_VIEW_SQL,
    LABELS_PATH,
    _normalize_segment_table,
    _require_duckdb,
    _result_to_dicts,
    _run_row_from_payload,
    append_label,
    build_where,
    create_derived_views,
    get_rollout_query,
    labels_arrow_table,
    load_schema_card,
    run_attempt_payloads,
    run_search,
    runs_arrow_schema,
)
from tinker_cookbook.tokendb.registry import list_runs, resolve_registry_dir
from tinker_cookbook.tokendb.writer import SEGMENTS_DIR

if TYPE_CHECKING:
    import duckdb

    from tinker_cookbook.tokendb.interface import TokenWriter

logger = logging.getLogger(__name__)

SEGCACHE_ENV_VAR = "TINKER_TOKENDB_SEGCACHE"
DEFAULT_SEGCACHE_DIR = "~/.cache/tinker-cookbook/tokendb/segcache"
DEFAULT_REFRESH_TTL_S = 5.0

# Per-run dashboard aggregates, one GROUP-BY-run_id pass each (replacing the
# viewer's per-run reader loop).
_TOTALS_BY_RUN_SQL = """
    SELECT run_id,
           count(*) AS n_rows,
           count(*) FILTER (WHERE source = 'filtered') AS n_filtered_rows,
           max(iteration) FILTER (WHERE iteration >= 0) AS latest_iteration
    FROM rollouts
    {where}
    GROUP BY run_id
"""
# Per-iteration mean of per-trajectory total_reward, excluding filtered rows;
# QUALIFY keeps the newest `series_len` iterations per run.
_SERIES_BY_RUN_SQL = """
    SELECT run_id, iteration, avg(total_reward) AS mean_total_reward
    FROM (
        SELECT run_id, iteration, any_value(total_reward) AS total_reward
        FROM rollouts
        WHERE iteration >= 0 AND source <> 'filtered' {extra}
        GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
    )
    GROUP BY run_id, iteration
    QUALIFY row_number() OVER (PARTITION BY run_id ORDER BY iteration DESC) <= {series_len}
    ORDER BY run_id, iteration
"""


def resolve_segcache_dir(segcache_dir: str | Path | None = None) -> Path:
    """Resolve the local segment-cache directory.

    Precedence: explicit *segcache_dir*, then the ``TINKER_TOKENDB_SEGCACHE``
    environment variable, then :data:`DEFAULT_SEGCACHE_DIR`.
    """
    if segcache_dir is None:
        segcache_dir = os.environ.get(SEGCACHE_ENV_VAR) or DEFAULT_SEGCACHE_DIR
    return Path(segcache_dir).expanduser()


def _log_path_key(log_path: str) -> str:
    """Stable per-store segcache subdirectory name."""
    return hashlib.sha1(log_path.encode()).hexdigest()[:16]


def _atomic_write_bytes(data: bytes, target: Path) -> None:
    """Write *data* to *target* atomically (tmp file + rename)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_parquet(table: Any, target: Path) -> None:
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    _atomic_write_bytes(sink.getvalue(), target)


def _is_v1_schema(schema: Any) -> bool:
    """True when a segment's own arrow schema is v1-shaped.

    Same sniff the single-run reader uses (``metrics`` as a JSON string
    column); a file without a ``metrics`` column is not a token DB segment.
    """
    import pyarrow as pa

    if "metrics" not in schema.names:
        raise ValueError("not a token DB segment (no 'metrics' column)")
    return pa.types.is_string(schema.field("metrics").type)


@dataclass
class _RunState:
    """Everything the backend tracks for one registered run."""

    run_id: str
    log_path: str
    storage: Storage | None
    record: dict[str, Any] = field(default_factory=dict)
    #: segment file name -> path the scan list references (in-place or cached)
    segments: dict[str, str] = field(default_factory=dict)
    #: unreadable segments, remembered so each is warned about once
    skipped: set[str] = field(default_factory=set)
    error: str | None = None
    card: dict[str, Any] | None = None
    card_key: Any = None


class RegistryBackend:
    """Lazy cross-run reader over every run in the token DB registry.

    Implements the read half of ``TokenStoreBackend``; ``open_writer``
    raises (writes stay per-run). ``sql()`` is the backend-specific escape
    hatch, exactly as on the single-run reader, with identical view names
    now spanning runs.

    Args:
        registry_dir: Registry directory override; ``None`` resolves via
            ``TINKER_TOKENDB_REGISTRY`` then the default.
        segcache_dir: Local cache directory for cloud-staged and v1-upgraded
            segments; ``None`` resolves via ``TINKER_TOKENDB_SEGCACHE`` then
            ``~/.cache/tinker-cookbook/tokendb/segcache``. Unbounded in this
            version; safe to delete between runs of the viewer.
        refresh_ttl_s: Minimum seconds between full registry rescans;
            :meth:`refresh` calls inside the window are no-ops so agent tool
            loops and dashboard polls stay cheap.
        storage_factory: Override for constructing a run's ``Storage`` from
            its ``log_path`` (tests inject counting stubs).
    """

    def __init__(
        self,
        registry_dir: str | None = None,
        *,
        segcache_dir: str | Path | None = None,
        refresh_ttl_s: float = DEFAULT_REFRESH_TTL_S,
        storage_factory: Callable[[str], Storage] | None = None,
    ) -> None:
        directory = resolve_registry_dir(registry_dir)
        if directory is None:
            raise ValueError(
                "The run registry is disabled (empty registry_dir / "
                "TINKER_TOKENDB_REGISTRY); the cross-run reader has nothing to scan."
            )
        self._registry_dir = directory
        self._segcache_dir = resolve_segcache_dir(segcache_dir)
        self._refresh_ttl_s = refresh_ttl_s
        self._storage_factory: Callable[[str], Storage] = storage_factory or storage_from_uri
        self._states: dict[str, _RunState] = {}
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._tlocal = threading.local()
        # Serializes refresh (registry rescan + catalog writes); per-thread
        # cursors let queries run concurrently with each other.
        self._lock = threading.Lock()
        self._last_refresh: float | None = None
        self._empty_segment_path: Path | None = None

    # --- Connection / views ---

    def _connection(self) -> duckdb.DuckDBPyConnection:
        conn = self._conn
        if conn is None:
            duckdb = _require_duckdb()
            conn = duckdb.connect(":memory:")
            self._empty_segment_path = self._ensure_empty_segment()
            self._replace_scan_view(conn, [])
            self._create_views(conn)
            self._rebuild_runs(conn, [])
            self._rebuild_labels(conn, [])
            self._conn = conn
        return conn

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        """This thread's cursor on the shared connection.

        DuckDB cursors share the database instance (and therefore the view
        catalog) while giving each thread its own execution state.
        """
        cursor = getattr(self._tlocal, "cursor", None)
        if cursor is None:
            conn = self._connection()
            with self._lock:
                cursor = conn.cursor()
            self._tlocal.cursor = cursor
        return cursor

    def _fetch(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return _result_to_dicts(self._cursor().execute(sql, list(params)))

    def _ensure_empty_segment(self) -> Path:
        """A zero-row v2 parquet file always included in the scan list.

        Keeps ``read_parquet([...])`` valid (and the views' column types
        anchored) when no run has segments yet.
        """
        import pyarrow as pa

        from tinker_cookbook.tokendb.schema import arrow_schema

        path = self._segcache_dir / "_empty-v2.parquet"
        if not path.exists():
            _atomic_write_parquet(pa.Table.from_pylist([], schema=arrow_schema()), path)
        return path

    def _replace_scan_view(self, conn: duckdb.DuckDBPyConnection, paths: Sequence[str]) -> None:
        assert self._empty_segment_path is not None
        all_paths = [str(self._empty_segment_path), *paths]
        quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in all_paths)
        conn.execute(
            "CREATE OR REPLACE VIEW segment_scan AS "
            f"SELECT * FROM read_parquet([{quoted}], union_by_name = true)"
        )

    def _create_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        # All rows across runs, tagged: `superseded` is computed per run
        # (same (split, iteration) in different runs never supersede each
        # other), matching the single-run reader within one run.
        conn.execute(
            """
            CREATE VIEW rollouts AS
            SELECT *,
                   run_attempt < max(run_attempt)
                       OVER (PARTITION BY run_id, split, iteration) AS superseded
            FROM segment_scan
            """
        )
        conn.execute(
            """
            CREATE VIEW trajectories AS
            SELECT run_id, run_attempt, split, iteration, group_idx, traj_idx,
                   count(*) AS n_steps,
                   sum(len(ac_tokens)) AS n_ac_tokens,
                   any_value(total_reward) AS total_reward,
                   any_value(final_reward) AS final_reward,
                   arg_max(stop_reason, step_idx) AS stop_reason,
                   any_value(filtered_reason) AS filtered_reason,
                   any_value(env_row_id) AS env_row_id,
                   min(ts) AS ts
            FROM segment_scan
            GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
            """
        )
        create_derived_views(conn)

    # --- Refresh: registry rescan + scan-list rebuild (TTL-gated) ---

    def refresh(self, *, force: bool = False) -> list[tuple[str, str]]:
        """Rescan the registry and pick up new runs/segments.

        TTL-gated: calls within ``refresh_ttl_s`` of the last full rescan
        are no-ops (pass ``force=True`` to override). A run whose store
        errors is skipped with a warning and reported with an ``error``
        field in :meth:`dashboard_stats`; one bad run or segment never
        poisons the connection. Returns the newly seen
        ``(run_id, segment_name)`` pairs.
        """
        with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._last_refresh is not None
                and now - self._last_refresh < self._refresh_ttl_s
            ):
                return []
            conn = self._connection()
            new_pairs: list[tuple[str, str]] = []
            changed = False
            seen: set[str] = set()
            for record in list_runs(self._registry_dir):
                run_id = str(record.get("run_id") or "")
                log_path = record.get("log_path")
                if not run_id or not log_path:
                    continue
                seen.add(run_id)
                state = self._states.get(run_id)
                if state is None:
                    state = self._make_state(run_id, str(log_path))
                    self._states[run_id] = state
                    changed = True
                state.record = record
                if state.storage is None:
                    continue
                state.error = None
                try:
                    names = [
                        n for n in state.storage.list_dir(SEGMENTS_DIR) if n.endswith(".parquet")
                    ]
                except FileNotFoundError:
                    names = []
                except Exception as e:
                    state.error = str(e)
                    logger.warning("Skipping run %s during registry refresh: %s", run_id, e)
                    continue
                for name in sorted(names):
                    if name in state.segments or name in state.skipped:
                        continue
                    try:
                        state.segments[name] = self._scan_path(state, name)
                    except Exception as e:
                        state.skipped.add(name)
                        logger.warning(
                            "Skipping unreadable segment %s of run %s: %s", name, run_id, e
                        )
                        continue
                    new_pairs.append((run_id, name))
                    changed = True
            for run_id in set(self._states) - seen:
                del self._states[run_id]
                changed = True
            if changed:
                paths = sorted(
                    path for state in self._states.values() for path in state.segments.values()
                )
                self._replace_scan_view(conn, paths)
            states = list(self._states.values())
            self._rebuild_runs(conn, states)
            self._rebuild_labels(conn, states)
            self._last_refresh = time.monotonic()
            return new_pairs

    def _make_state(self, run_id: str, log_path: str) -> _RunState:
        try:
            storage = self._storage_factory(log_path)
        except Exception as e:
            logger.warning("Could not open the store of run %s (%s): %s", run_id, log_path, e)
            return _RunState(run_id=run_id, log_path=log_path, storage=None, error=str(e))
        return _RunState(run_id=run_id, log_path=log_path, storage=storage)

    def _scan_path(self, state: _RunState, name: str) -> str:
        """Resolve one segment file to the path the scan list references.

        Local v2 files are scanned in place (zero copy). Cloud and v1 files
        are staged into the segcache exactly once (existence = validity;
        segments are immutable), with v1 files normalized to v2 at cache
        fill. Raises on unreadable files (the caller skips and warns).
        """
        import pyarrow.parquet as pq

        assert state.storage is not None
        cache_dir = self._segcache_dir / _log_path_key(state.log_path)
        upgraded = cache_dir / "upgraded" / name
        if upgraded.exists():
            return str(upgraded)
        staged = cache_dir / name
        if staged.exists():
            return str(staged)
        if isinstance(state.storage, LocalStorage):
            local = state.storage.root / SEGMENTS_DIR / name
            # Footer-only sniff; also rejects corrupt files at refresh time
            # instead of poisoning the shared scan.
            if not _is_v1_schema(pq.read_schema(local)):
                return str(local)
            _atomic_write_parquet(_normalize_segment_table(pq.read_table(local)), upgraded)
            return str(upgraded)
        # Cloud (or otherwise non-local) store: one fetch through Storage.
        data = state.storage.read(f"{SEGMENTS_DIR}/{name}")
        table = pq.read_table(io.BytesIO(data))
        if _is_v1_schema(table.schema):
            _atomic_write_parquet(_normalize_segment_table(table), upgraded)
            return str(upgraded)
        _atomic_write_bytes(data, staged)
        return str(staged)

    def _rebuild_runs(self, conn: duckdb.DuckDBPyConnection, states: Sequence[_RunState]) -> None:
        """(Re)build the ``runs`` relation: every store's run-attempt rows."""
        import pyarrow as pa

        rows: list[dict[str, Any]] = []
        for state in states:
            if state.storage is None or state.error is not None:
                continue
            try:
                payloads = run_attempt_payloads(state.storage)
            except Exception as e:
                logger.warning("Could not read run attempts of run %s: %s", state.run_id, e)
                continue
            rows.extend(_run_row_from_payload(payload) for payload in payloads)
        rows.sort(key=lambda r: (str(r.get("run_id")), str(r.get("run_attempt"))))
        table = pa.Table.from_pylist(rows, schema=runs_arrow_schema())
        conn.register("_registry_runs_src", table)
        # A real table (not a view over the registration): cursors share the
        # catalog but not the parent connection's registered objects.
        conn.execute("CREATE OR REPLACE TABLE runs AS SELECT * FROM _registry_runs_src")
        conn.unregister("_registry_runs_src")

    def _rebuild_labels(self, conn: duckdb.DuckDBPyConnection, states: Sequence[_RunState]) -> None:
        """(Re)build ``labels``: the union of every run's ``labels.jsonl``.

        Records missing a ``run_id`` are backfilled with the owning run's,
        so cross-run label queries always have run attribution.
        """
        records: list[dict[str, Any]] = []
        for state in states:
            if state.storage is None or state.error is not None:
                continue
            try:
                if not state.storage.exists(LABELS_PATH):
                    continue
                raw = state.storage.read(LABELS_PATH).decode()
            except Exception as e:
                logger.warning("Could not read labels of run %s: %s", state.run_id, e)
                continue
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    if record.get("run_id") is None:
                        record["run_id"] = state.run_id
                    records.append(record)
        conn.register("_registry_labels_src", labels_arrow_table(records))
        conn.execute(
            "CREATE OR REPLACE TABLE _labels_union AS "
            + LABELS_CAST_SELECT.format(src="_registry_labels_src")
        )
        conn.unregister("_registry_labels_src")
        conn.execute(LABELS_DEDUP_VIEW_SQL.format(source="_labels_union"))

    # --- TokenStoreBackend surface ---

    def open_writer(self, run_context: Mapping[str, Any]) -> TokenWriter:
        raise NotImplementedError(
            "RegistryBackend is a read surface across runs; open a writer on the "
            "run's own store (ParquetSegmentBackend) instead."
        )

    def query(
        self,
        *,
        latest_only: bool = False,
        limit: int | None = None,
        offset: int = 0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Structured row query over the cross-run ``rollouts`` view.

        Same filters as :meth:`ParquetSegmentReader.query`; ``run_id`` is an
        ordinary optional filter, and omitting it spans every run.
        """
        self.refresh()
        where, params = build_where(**filters)
        view = "rollouts_latest" if latest_only else "rollouts"
        sql = f"SELECT * FROM {view} {where} ORDER BY iteration, ts, group_idx, traj_idx, step_idx"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        if offset:
            sql += f" OFFSET {int(offset)}"
        return self._fetch(sql, params)

    def trajectories(
        self,
        *,
        latest_only: bool = False,
        limit: int = 500,
        offset: int = 0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Trajectory-grain aggregation of :meth:`query`, across runs."""
        self.refresh()
        where, params = build_where(**filters)
        if latest_only:
            where = f"{where} AND NOT superseded" if where else "WHERE NOT superseded"
        sql = _TRAJECTORIES_SQL.format(where=where, limit=int(limit), offset=int(offset))
        return self._fetch(sql, params)

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Regex / token-subsequence search across every run's rows.

        See :meth:`ParquetSegmentReader.search`; filter by ``run_id`` to
        restrict to one run.
        """
        self.refresh()
        return run_search(self._fetch, **kwargs)

    def search_hit_counts(self, **search_kwargs: Any) -> dict[int, int]:
        """Grouped-by-iteration hit counts for a :meth:`search` call."""
        counts: dict[int, int] = {}
        for row in self.search(**search_kwargs):
            counts[row["iteration"]] = counts.get(row["iteration"], 0) + 1
        return counts

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one trajectory's step rows; needs ``run_id`` beyond one run."""
        self.refresh()
        run_id = self._require_run_id(run_id, "get_rollout")
        sql, params = get_rollout_query(split, iteration, group_idx, traj_idx, run_attempt, run_id)
        return self._fetch(sql, params)

    def group_traj_idxs(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        """One group's distinct ``traj_idx`` values; needs ``run_id`` beyond one run."""
        self.refresh()
        run_id = self._require_run_id(run_id, "group_traj_idxs")
        sql = (
            "SELECT DISTINCT traj_idx FROM rollouts"
            " WHERE split = ? AND iteration = ? AND group_idx = ?"
        )
        params: list[Any] = [split, iteration, group_idx]
        if run_attempt is not None:
            sql += " AND run_attempt = ?"
            params.append(run_attempt)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY traj_idx"
        return [row["traj_idx"] for row in self._fetch(sql, params)]

    def _require_run_id(self, run_id: str | None, method: str) -> str | None:
        """Rollout identity is only unique per run: with more than one run
        registered, per-rollout methods must name the run."""
        if run_id is None and len(self._states) > 1:
            raise ValueError(
                f"{method} needs a run_id: {len(self._states)} runs are registered "
                "and rollout identity is only unique per run"
            )
        return run_id

    def runs(self) -> list[dict[str, Any]]:
        """The cross-run ``runs`` view: one row per (run_id, run_attempt)."""
        self.refresh()
        return self._fetch("SELECT * FROM runs ORDER BY run_id, run_attempt")

    def dashboard_stats(
        self, *, recent_k: int = 5, series_len: int = 50, run_id: str | None = None
    ) -> dict[str, Any]:
        """Dashboard aggregates in one GROUP-BY-run_id pass.

        With ``run_id``, the shape matches the single-run reader's
        ``dashboard_stats`` exactly. Without it, the same top-level
        aggregates span every run and a ``per_run`` mapping adds the per-run
        breakdown (runs whose store errored during refresh carry an
        ``error`` field with null counts; runs with no rows yet report
        zeros).
        """
        self.refresh()
        per_run = self._per_run_stats(recent_k=recent_k, series_len=series_len, run_id=run_id)
        if run_id is not None:
            return per_run.get(run_id) or _zero_run_stats()
        totals = self._fetch(_TOTALS_BY_RUN_SQL.format(where=""))
        n_rows = sum(t["n_rows"] for t in totals)
        n_filtered = sum(t["n_filtered_rows"] for t in totals)
        latest = [t["latest_iteration"] for t in totals if t["latest_iteration"] is not None]
        # Cross-run series: mean over every run's trajectories per iteration.
        series = self._fetch(
            f"""
            SELECT iteration, avg(total_reward) AS mean_total_reward
            FROM (
                SELECT iteration, any_value(total_reward) AS total_reward
                FROM rollouts
                WHERE iteration >= 0 AND source <> 'filtered'
                GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
            )
            GROUP BY iteration
            ORDER BY iteration DESC
            LIMIT {int(series_len)}
            """
        )
        series.reverse()
        return {
            "n_rows": n_rows,
            "n_filtered_rows": n_filtered,
            "latest_iteration": max(latest) if latest else None,
            "mean_recent_reward": _mean_recent(series, recent_k),
            "reward_series": series,
            "per_run": per_run,
        }

    def _per_run_stats(
        self, *, recent_k: int, series_len: int, run_id: str | None
    ) -> dict[str, dict[str, Any]]:
        per_run: dict[str, dict[str, Any]] = {}
        for state in self._states.values():
            if run_id is not None and state.run_id != run_id:
                continue
            stats = _zero_run_stats()
            if state.storage is None or state.error is not None:
                stats.update(
                    {
                        "n_rows": None,
                        "n_filtered_rows": None,
                        "latest_iteration": None,
                        "error": state.error or "store unavailable",
                    }
                )
            per_run[state.run_id] = stats
        where, params = ("WHERE run_id = ?", [run_id]) if run_id is not None else ("", [])
        for totals in self._fetch(_TOTALS_BY_RUN_SQL.format(where=where), params):
            stats = per_run.setdefault(str(totals["run_id"]), _zero_run_stats())
            stats["n_rows"] = totals["n_rows"]
            stats["n_filtered_rows"] = totals["n_filtered_rows"]
            stats["latest_iteration"] = totals["latest_iteration"]
        extra = "AND run_id = ?" if run_id is not None else ""
        for point in self._fetch(
            _SERIES_BY_RUN_SQL.format(extra=extra, series_len=int(series_len)), params
        ):
            stats = per_run.get(str(point["run_id"]))
            if stats is not None:
                stats["reward_series"].append(
                    {
                        "iteration": point["iteration"],
                        "mean_total_reward": point["mean_total_reward"],
                    }
                )
        for stats in per_run.values():
            stats["mean_recent_reward"] = _mean_recent(stats["reward_series"], recent_k)
        return per_run

    def schema_card(self) -> dict[str, Any]:
        """Aggregate every run's observed-keys card with per-run attribution.

        Union shape matches the single-run card (``metrics_keys`` /
        ``attrs_keys`` / ``token_metrics_keys`` / ``tags`` /
        ``keys_truncated``), plus ``runs``: a per-``run_id`` mapping of the
        contributing cards. Per-run cards are manifest-only and cached until
        the run's manifest activity changes.
        """
        self.refresh()
        union: dict[str, set[str]] = {
            "metrics_keys": set(),
            "attrs_keys": set(),
            "token_metrics_keys": set(),
            "tags": set(),
        }
        truncated = False
        runs_cards: dict[str, dict[str, Any]] = {}
        for state in self._states.values():
            card = self._run_card(state)
            if card is None:
                continue
            runs_cards[state.run_id] = card
            for field_name in union:
                union[field_name].update(card.get(field_name) or [])
            truncated = truncated or bool(card.get("keys_truncated"))
        return {
            **{field_name: sorted(values) for field_name, values in union.items()},
            "keys_truncated": truncated,
            "runs": runs_cards,
        }

    def _run_card(self, state: _RunState) -> dict[str, Any] | None:
        if state.storage is None:
            return None
        status = state.record.get("status") or {}
        key = (status.get("last_activity_ts"), status.get("n_segments"))
        if state.card is None or state.card_key != key:
            try:
                state.card = load_schema_card(state.storage)
            except Exception as e:
                logger.warning("Could not build the schema card of run %s: %s", state.run_id, e)
                state.card = None
            state.card_key = key
        return state.card

    # --- Labels ---

    def labels(self, **filters: Any) -> list[dict[str, Any]]:
        """Current labels across every run (last-write-wins, tombstones dropped).

        Same filters as :meth:`ParquetSegmentReader.labels`; the label
        sources are re-read on every call (labels are written out of band).
        """
        bad = set(filters) - LABEL_FILTER_FIELDS
        if bad:
            raise ValueError(f"Unsupported label filters: {sorted(bad)}")
        self.refresh()
        with self._lock:  # labels change out of band: re-resolve per read
            self._rebuild_labels(self._connection(), list(self._states.values()))
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in filters.items():
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM labels"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY ts"
        rows = self._fetch(sql, params)
        for row in rows:
            row["label_value"] = json.loads(row["label_value"])
        return rows

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        """Append an annotation to the owning run's ``labels.jsonl``.

        ``key`` must include ``run_id`` (labels are stored per run; the
        cross-run reader only routes the write).
        """
        run_id = key.get("run_id")
        if not run_id:
            raise ValueError("add_label on the cross-run reader needs run_id in the key")
        self.refresh()
        state = self._states.get(str(run_id))
        if state is None or state.storage is None:
            raise ValueError(f"unknown run_id {run_id!r} (not in the registry)")
        append_label(state.storage, key, label_key, label_value, author=author, note=note)

    # --- Live tail (per-run only in this version) ---

    def subscribe(
        self,
        *,
        poll_interval_s: float = DEFAULT_SUBSCRIBE_POLL_INTERVAL_S,
        **filters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Tail one run's newly written rows; requires a ``run_id`` filter.

        A multiplexed all-runs subscription is deliberately not offered yet;
        the cross-run views pick new files up via :meth:`refresh` regardless.
        """
        run_id = filters.get("run_id")
        if not run_id:
            raise ValueError("RegistryBackend.subscribe requires a run_id filter")
        self.refresh()
        state = self._states.get(str(run_id))
        if state is None:
            self.refresh(force=True)
            state = self._states.get(str(run_id))
        if state is None or state.storage is None:
            raise ValueError(f"unknown run_id {run_id!r} (not in the registry)")
        return self._tail(state, filters, poll_interval_s)

    async def _tail(
        self, state: _RunState, filters: dict[str, Any], poll_interval_s: float
    ) -> AsyncIterator[dict[str, Any]]:
        # Baseline: everything already in the store is "old". New rows are
        # matched on a scratch connection so the tail never contends with
        # the shared catalog.
        known = set(await asyncio.to_thread(self._segment_names, state))
        where, params = build_where(**filters)
        duckdb = _require_duckdb()
        conn = duckdb.connect(":memory:")
        while True:
            await asyncio.sleep(poll_interval_s)
            names = await asyncio.to_thread(self._segment_names, state)
            for name in sorted(set(names) - known):
                known.add(name)
                table = await asyncio.to_thread(self._load_segment_table, state, name)
                if table is None:
                    continue
                conn.register("_new_segment", table)
                rows = _result_to_dicts(
                    conn.execute(
                        f"SELECT * FROM _new_segment {where}"
                        " ORDER BY iteration, ts, group_idx, traj_idx, step_idx",
                        params,
                    )
                )
                conn.unregister("_new_segment")
                for row in rows:
                    yield row

    def _segment_names(self, state: _RunState) -> list[str]:
        assert state.storage is not None
        try:
            return [n for n in state.storage.list_dir(SEGMENTS_DIR) if n.endswith(".parquet")]
        except FileNotFoundError:
            return []

    def _load_segment_table(self, state: _RunState, name: str) -> Any:
        import pyarrow.parquet as pq

        assert state.storage is not None
        try:
            data = state.storage.read(f"{SEGMENTS_DIR}/{name}")
            return _normalize_segment_table(pq.read_table(io.BytesIO(data)))
        except Exception as e:
            logger.warning("Skipping unreadable segment %s of run %s: %s", name, state.run_id, e)
            return None

    # --- Backend-specific escape hatch (not part of TokenStoreBackend) ---

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Run a read-only DuckDB SQL query over the cross-run views.

        Same contract and view names as the single-run reader's ``sql()``
        (``rollouts``, ``rollouts_latest``, ``trajectories``, ``labels``,
        ``runs``, ``correct``, ``parse_errors``, ``context_overflows``), now
        spanning every registered run with ``run_id`` as a normal column.
        Only a single ``SELECT`` (or ``WITH ... SELECT``) statement is
        allowed.
        """
        duckdb = _require_duckdb()
        self.refresh()
        cursor = self._cursor()
        statements = cursor.extract_statements(query)
        if len(statements) != 1:
            raise ValueError("sql() accepts exactly one statement")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError(f"sql() only accepts SELECT statements, got {statements[0].type}")
        return self._fetch(query, params or [])


def _zero_run_stats() -> dict[str, Any]:
    return {
        "n_rows": 0,
        "n_filtered_rows": 0,
        "latest_iteration": None,
        "mean_recent_reward": None,
        "reward_series": [],
    }


def _mean_recent(series: list[dict[str, Any]], recent_k: int) -> float | None:
    recent = [
        p["mean_total_reward"]
        for p in series[-int(recent_k) :]
        if p["mean_total_reward"] is not None
    ]
    return (sum(recent) / len(recent)) if recent else None
