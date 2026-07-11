"""DuckDB read path for the parquet segment token store.

:class:`ParquetSegmentReader` is the read half of
:class:`~tinker_cookbook.tokendb.writer.ParquetSegmentBackend`: an in-memory
DuckDB connection over the immutable segment files, with views for browsing
(``rollouts`` / ``rollouts_latest`` / ``trajectories`` / ``labels``) and the
structured ``TokenStoreBackend`` read methods (``query`` / ``get_rollout`` /
``search`` / ``subscribe`` / ``add_label`` / ``labels``).

Segment registration is incremental: the reader tracks which segment files it
has loaded and, on refresh, lists the segments directory (the directory
listing is the source of truth; manifests are only a liveness hint) and loads
only new files. Segments are immutable, so there is no cache invalidation.
Segment bytes are read through the ``Storage`` protocol and registered as
arrow tables, so the reader works against local and cloud stores alike.

:class:`TokenDB` is the thin user-facing facade: ``TokenDB(log_path)``
constructs the default backend and exposes the structured methods plus
``sql()`` as the backend-specific, SELECT-only escape hatch (``sql()`` is
deliberately not part of the ``TokenStoreBackend`` protocol).

DuckDB is imported lazily; importing this module does not require it.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinker_cookbook.stores.storage import LocalStorage, Storage, storage_from_uri
from tinker_cookbook.tokendb.writer import SEGMENTS_DIR, TOKENS_DIR, ParquetSegmentBackend

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

LABELS_PATH = f"{TOKENS_DIR}/labels.jsonl"

DEFAULT_SUBSCRIBE_POLL_INTERVAL_S = 2.0

_LABEL_KEY_FIELDS = ("run_id", "split", "iteration", "group_idx", "traj_idx", "step_idx")

_LABEL_COLUMNS_SQL = (
    "{run_id: 'VARCHAR', split: 'VARCHAR', iteration: 'INTEGER', group_idx: 'INTEGER', "
    "traj_idx: 'INTEGER', step_idx: 'INTEGER', label_key: 'VARCHAR', label_value: 'JSON', "
    "author: 'VARCHAR', ts: 'TIMESTAMP', note: 'VARCHAR'}"
)


def _result_to_dicts(result: Any) -> list[dict[str, Any]]:
    """Fetch a DuckDB result as a list of dicts (via arrow, so list/ts columns
    come back as Python lists / datetimes, and map columns as dicts).
    Tolerates the ``fetch_arrow_table`` -> ``to_arrow_table`` rename across
    duckdb versions."""
    to_arrow = getattr(result, "to_arrow_table", None)
    table = to_arrow() if to_arrow is not None else result.fetch_arrow_table()
    return table.to_pylist(maps_as_pydicts="strict")


def _parse_v1_metrics(raw: str | None) -> dict[str, float]:
    """Parse a v1 JSON ``metrics`` string into v2 map entries.

    Lenient by design: v1 serialized with ``json.dumps``, which emits bare
    ``NaN`` / ``Infinity`` tokens (invalid JSON) for those float values;
    Python's ``json.loads`` reads them back as the real floats. Only
    float-coercible values become map entries; everything else (strings,
    nested objects, unparseable payloads) is silently dropped — this is
    read-time normalization of historical data, not a validation gate.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _normalize_segment_table(table: Any) -> Any:
    """Normalize a segment's arrow table to the v2 schema shape.

    Per-segment version detection sniffs the parquet file's own arrow schema
    (self-describing; covers orphan segments with no manifest line and
    mixed-version stores from a ``run_attempt`` resume after an upgrade). v1
    segments (``metrics`` is a JSON string column) are rewritten in Python:
    ``metrics`` parsed into map entries (:func:`_parse_v1_metrics`), ``attrs``
    and ``token_metrics`` empty, ``tool_calls`` NULL. The DuckDB connection
    only ever sees v2 shape, so the positional ``INSERT INTO ... SELECT *``
    stays valid.
    """
    import pyarrow as pa

    from tinker_cookbook.tokendb.schema import arrow_schema

    if not pa.types.is_string(table.schema.field("metrics").type):
        return table  # already v2
    rows = table.to_pylist()
    for row in rows:
        row["metrics"] = _parse_v1_metrics(row["metrics"])
        row["attrs"] = {}
        row["token_metrics"] = {}
        row["tool_calls"] = None
    return pa.Table.from_pylist(rows, schema=arrow_schema())


def _require_duckdb() -> Any:
    """Import duckdb, raising an actionable error if the extra is missing."""
    try:
        import duckdb
    except ImportError as e:
        raise ImportError(
            "duckdb is required for the token DB read path. "
            "Install the token DB extra with: pip install 'tinker-cookbook[tokendb]'"
        ) from e
    return duckdb


def reconstruct_full_ob(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Expand delta-encoded observations back to full token sequences.

    Args:
        rows: The step rows of one trajectory, ordered by ``step_idx`` (as
            returned by :meth:`ParquetSegmentReader.get_rollout`). Each row
            needs ``ob_tokens``, ``ob_is_delta``, and ``ac_tokens``.

    Returns:
        One full observation token list per row. A delta row's observation is
        the previous full sequence (previous ob + ac) plus the stored suffix;
        a non-delta row's observation is its stored tokens as-is.
    """
    full_obs: list[list[int]] = []
    prev_sequence: list[int] = []
    for row in rows:
        ob_tokens = list(row["ob_tokens"])
        full_ob = prev_sequence + ob_tokens if row["ob_is_delta"] else ob_tokens
        full_obs.append(full_ob)
        prev_sequence = full_ob + list(row["ac_tokens"])
    return full_obs


def _contains_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    """Return True if *needle* appears as a contiguous subsequence of *haystack*."""
    if not needle:
        return True
    n = len(needle)
    first = needle[0]
    for i, tok in enumerate(haystack):
        if tok == first and len(haystack) - i >= n and list(haystack[i : i + n]) == list(needle):
            return True
    return False


class ParquetSegmentReader:
    """DuckDB reader over a parquet segment token store.

    Not thread-safe (one DuckDB connection); create one reader per thread if
    needed. All I/O goes through the ``Storage`` protocol.
    """

    def __init__(self, storage_or_log_path: Storage | str | Path) -> None:
        if isinstance(storage_or_log_path, (str, Path)):
            self._storage: Storage = storage_from_uri(str(storage_or_log_path))
        else:
            self._storage = storage_or_log_path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._registered: set[str] = set()

    # --- Connection / registration ---

    def _connection(self) -> duckdb.DuckDBPyConnection:
        conn = self._conn
        if conn is None:
            duckdb = _require_duckdb()
            import pyarrow as pa

            from tinker_cookbook.tokendb.schema import arrow_schema

            conn = duckdb.connect(":memory:")
            empty = pa.Table.from_pylist([], schema=arrow_schema())
            conn.register("_empty_segment", empty)
            conn.execute("CREATE TABLE segment_rows AS SELECT * FROM _empty_segment")
            conn.unregister("_empty_segment")
            self._create_views(conn)
            self._conn = conn
        return conn

    def _create_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        # All rows, tagged: `superseded` is true when a later run_attempt
        # produced rows for the same (split, iteration). Both attempts are
        # shown by default (decision: superseded rows are tagged, not hidden).
        conn.execute(
            """
            CREATE VIEW rollouts AS
            SELECT *,
                   run_attempt < max(run_attempt) OVER (PARTITION BY split, iteration)
                       AS superseded
            FROM segment_rows
            """
        )
        # Dedup convenience: only the latest attempt per (split, iteration).
        conn.execute("CREATE VIEW rollouts_latest AS SELECT * FROM rollouts WHERE NOT superseded")
        # Trajectory grain.
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
            FROM segment_rows
            GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
            """
        )

    def _list_segment_files(self) -> list[str]:
        return sorted(
            name
            for name in self._storage.list_dir(SEGMENTS_DIR)
            if name.endswith(".parquet") and name not in self._registered
        )

    def _load_segment_table(self, name: str) -> Any:
        import pyarrow.parquet as pq

        data = self._storage.read(f"{SEGMENTS_DIR}/{name}")
        return _normalize_segment_table(pq.read_table(io.BytesIO(data)))

    def refresh(self) -> list[str]:
        """Register segment files that appeared since the last refresh.

        The directory listing is the source of truth (manifests are only a
        hint); segments are immutable, so already-registered files never need
        invalidation. Returns the newly registered file names.
        """
        conn = self._connection()
        new_files = self._list_segment_files()
        for name in new_files:
            table = self._load_segment_table(name)
            conn.register("_new_segment", table)
            conn.execute("INSERT INTO segment_rows SELECT * FROM _new_segment")
            conn.unregister("_new_segment")
            self._registered.add(name)
        return new_files

    # --- Structured query ---

    def _build_where(
        self,
        *,
        run_id: str | None = None,
        run_attempt: int | None = None,
        split: str | None = None,
        iteration: int | None = None,
        min_iteration: int | None = None,
        max_iteration: int | None = None,
        group_idx: int | None = None,
        traj_idx: int | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        stop_reason: str | None = None,
        source: str | None = None,
        filtered_reason: str | None = None,
        tag: str | None = None,
        env_row_id: str | None = None,
        text_regex: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in [
            ("run_id", run_id),
            ("run_attempt", run_attempt),
            ("split", split),
            ("iteration", iteration),
            ("group_idx", group_idx),
            ("traj_idx", traj_idx),
            ("stop_reason", stop_reason),
            ("source", source),
            ("filtered_reason", filtered_reason),
            ("env_row_id", env_row_id),
        ]:
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if min_iteration is not None:
            clauses.append("iteration >= ?")
            params.append(min_iteration)
        if max_iteration is not None:
            clauses.append("iteration <= ?")
            params.append(max_iteration)
        if min_reward is not None:
            clauses.append("total_reward >= ?")
            params.append(min_reward)
        if max_reward is not None:
            clauses.append("total_reward <= ?")
            params.append(max_reward)
        if tag is not None:
            clauses.append("list_contains(tags, ?)")
            params.append(tag)
        if text_regex is not None:
            clauses.append(
                "(regexp_matches(coalesce(ob_text, ''), ?)"
                " OR regexp_matches(coalesce(ac_text, ''), ?))"
            )
            params.extend([text_regex, text_regex])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def query(
        self,
        *,
        latest_only: bool = False,
        limit: int | None = None,
        offset: int = 0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Structured row query over the ``rollouts`` view.

        Filters (all optional, ANDed): ``run_id``, ``run_attempt``, ``split``,
        ``iteration``, ``min_iteration`` / ``max_iteration``, ``group_idx``,
        ``traj_idx``, ``min_reward`` / ``max_reward`` (over ``total_reward``),
        ``stop_reason``, ``source``, ``filtered_reason``, ``tag`` (tags list
        contains), ``env_row_id``, ``text_regex`` (over ``ob_text`` /
        ``ac_text``). ``latest_only=True`` queries ``rollouts_latest``
        (dropping attempts superseded by a resume). Rows are ordered by
        ``(iteration, ts)`` and include the computed ``superseded`` flag.
        """
        self.refresh()
        where, params = self._build_where(**filters)
        view = "rollouts_latest" if latest_only else "rollouts"
        sql = f"SELECT * FROM {view} {where} ORDER BY iteration, ts, group_idx, traj_idx, step_idx"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        if offset:
            sql += f" OFFSET {int(offset)}"
        return self._fetch(sql, params)

    def _fetch(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._connection()
        return _result_to_dicts(conn.execute(sql, list(params)))

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch the full trajectory: all step rows ordered by ``step_idx``.

        ``run_attempt=None`` selects the latest attempt that produced rows for
        this trajectory. Delta-encoded observations are returned as stored;
        use :func:`reconstruct_full_ob` to expand them.
        """
        self.refresh()
        params: list[Any] = [split, iteration, group_idx, traj_idx]
        sql = """
            SELECT * FROM rollouts
            WHERE split = ? AND iteration = ? AND group_idx = ? AND traj_idx = ?
        """
        if run_attempt is None:
            sql += """
              AND run_attempt = (
                  SELECT max(run_attempt) FROM rollouts
                  WHERE split = ? AND iteration = ? AND group_idx = ? AND traj_idx = ?
              )
            """
            params += [split, iteration, group_idx, traj_idx]
        else:
            sql += " AND run_attempt = ?"
            params.append(run_attempt)
        sql += " ORDER BY step_idx"
        return self._fetch(sql, params)

    # --- Search ---

    def search(
        self,
        *,
        regex: str | None = None,
        fields: Sequence[str] = ("ac_text", "ob_text"),
        token_subsequence: Sequence[int] | None = None,
        limit: int | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Search rows by text regex and/or token-ID subsequence.

        ``regex`` matches over the given text ``fields`` (any of ``ac_text``,
        ``ob_text``, ``metrics``, ``logs``). ``token_subsequence`` matches a
        contiguous run of token IDs inside ``ac_tokens`` (useful for special
        tokens with unstable decodings). Additional structured ``filters`` are
        the same as :meth:`query`. Results are ordered by ``(iteration, ts)``.
        """
        allowed_fields = {"ac_text", "ob_text", "metrics", "logs"}
        bad = set(fields) - allowed_fields
        if bad:
            raise ValueError(f"Unsupported search fields: {sorted(bad)}")
        self.refresh()
        where, params = self._build_where(**filters)
        clauses = [where[len("WHERE ") :]] if where else []
        if regex is not None:
            field_matches = [f"regexp_matches(coalesce({f}, ''), ?)" for f in fields]
            clauses.append(f"({' OR '.join(field_matches)})")
            params.extend([regex] * len(fields))
        if token_subsequence:
            # Cheap SQL prefilter on the first token; the exact contiguous
            # match runs in Python below (portable across backends).
            clauses.append("list_contains(ac_tokens, ?)")
            params.append(int(token_subsequence[0]))
        sql = "SELECT * FROM rollouts"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY iteration, ts, group_idx, traj_idx, step_idx"
        rows = self._fetch(sql, params)
        if token_subsequence:
            needle = [int(t) for t in token_subsequence]
            rows = [row for row in rows if _contains_subsequence(row["ac_tokens"], needle)]
        if limit is not None:
            rows = rows[: int(limit)]
        return rows

    def search_hit_counts(self, **search_kwargs: Any) -> dict[int, int]:
        """Grouped-by-iteration hit counts for a :meth:`search` call."""
        counts: dict[int, int] = {}
        for row in self.search(**search_kwargs):
            counts[row["iteration"]] = counts.get(row["iteration"], 0) + 1
        return counts

    # --- Labels ---

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        """Append an annotation to ``labels.jsonl`` (last-write-wins by ``ts``).

        Args:
            key: Rollout identity — any of ``run_id``, ``split``,
                ``iteration``, ``group_idx``, ``traj_idx``, ``step_idx``
                (``step_idx`` omitted/None means the whole trajectory).
            label_key: Label name (e.g. ``"quality"``).
            label_value: JSON-serializable value; ``None`` writes a tombstone
                that deletes the label.
            author: Who wrote the label (user name, ``"agent"``, ...).
            note: Optional free-form note.
        """
        bad_keys = set(key) - set(_LABEL_KEY_FIELDS)
        if bad_keys:
            raise ValueError(f"Unsupported label key fields: {sorted(bad_keys)}")
        record: dict[str, Any] = {field: key.get(field) for field in _LABEL_KEY_FIELDS}
        record.update(
            {
                "label_key": label_key,
                "label_value": label_value,
                "author": author,
                "ts": datetime.now(UTC).isoformat(),
                "note": note,
            }
        )
        self._storage.append(LABELS_PATH, (json.dumps(record, default=str) + "\n").encode())

    def _refresh_labels_source(self, conn: duckdb.DuckDBPyConnection) -> None:
        """(Re)create the ``labels`` view over the current ``labels.jsonl``.

        Labels are append-only and written by other processes, so the source
        is re-resolved on every read. For a local store the view reads the
        file directly via ``read_json``; otherwise the file is fetched through
        ``Storage`` and registered as an in-memory table.
        """
        conn.execute("DROP VIEW IF EXISTS labels")
        if not self._storage.exists(LABELS_PATH):
            conn.execute(
                "CREATE VIEW labels AS SELECT"
                " NULL::VARCHAR AS run_id, NULL::VARCHAR AS split,"
                " NULL::INTEGER AS iteration, NULL::INTEGER AS group_idx,"
                " NULL::INTEGER AS traj_idx, NULL::INTEGER AS step_idx,"
                " NULL::VARCHAR AS label_key, NULL::JSON AS label_value,"
                " NULL::VARCHAR AS author, NULL::TIMESTAMP AS ts, NULL::VARCHAR AS note"
                " WHERE false"
            )
            return
        if isinstance(self._storage, LocalStorage):
            path = str(self._storage.root / LABELS_PATH).replace("'", "''")
            source = (
                f"read_json('{path}', format = 'newline_delimited', columns = {_LABEL_COLUMNS_SQL})"
            )
        else:
            records = [
                json.loads(line)
                for line in self._storage.read(LABELS_PATH).decode().splitlines()
                if line.strip()
            ]
            import pyarrow as pa

            table = pa.Table.from_pylist(
                [
                    {
                        **{field: rec.get(field) for field in _LABEL_KEY_FIELDS},
                        "label_key": rec.get("label_key"),
                        "label_value": json.dumps(rec.get("label_value")),
                        "author": rec.get("author"),
                        "ts": rec.get("ts"),
                        "note": rec.get("note"),
                    }
                    for rec in records
                ]
            )
            conn.register("_labels_raw", table)
            source = (
                "(SELECT run_id, split, iteration::INTEGER AS iteration,"
                " group_idx::INTEGER AS group_idx, traj_idx::INTEGER AS traj_idx,"
                " step_idx::INTEGER AS step_idx, label_key, label_value::JSON AS label_value,"
                " author, ts::TIMESTAMP AS ts, note FROM _labels_raw)"
            )
        # Last write (by ts) wins per (identity, label_key); tombstones
        # (null label_value) win first, then drop out of the view.
        conn.execute(
            f"""
            CREATE VIEW labels AS
            SELECT * EXCLUDE (_rank) FROM (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY run_id, split, iteration, group_idx, traj_idx,
                                        step_idx, label_key
                           ORDER BY ts DESC
                       ) AS _rank
                FROM {source}
            )
            WHERE _rank = 1
              AND label_value IS NOT NULL
              AND CAST(label_value AS VARCHAR) <> 'null'
            """
        )

    def labels(self, **filters: Any) -> list[dict[str, Any]]:
        """Fetch current labels (after last-write-wins and tombstone filtering).

        Filters: ``run_id``, ``split``, ``iteration``, ``group_idx``,
        ``traj_idx``, ``step_idx``, ``label_key``, ``author``. Returned
        ``label_value`` is decoded back to a Python value.
        """
        allowed = set(_LABEL_KEY_FIELDS) | {"label_key", "author"}
        bad = set(filters) - allowed
        if bad:
            raise ValueError(f"Unsupported label filters: {sorted(bad)}")
        conn = self._connection()
        self._refresh_labels_source(conn)
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

    # --- Live tail ---

    async def subscribe(
        self,
        *,
        poll_interval_s: float = DEFAULT_SUBSCRIBE_POLL_INTERVAL_S,
        **filters: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield rows from segment files written after the subscription starts.

        Polls the segments directory every ``poll_interval_s`` seconds for new
        files (segments are immutable, so a new file is the unit of new data)
        and yields its rows that match ``filters`` (same as :meth:`query`),
        ordered by ``(iteration, ts)`` within each batch. Runs until the
        caller stops iterating.
        """
        # Baseline: everything already on disk is "old".
        await asyncio.to_thread(self.refresh)
        where, params = self._build_where(**filters)
        conn = self._connection()
        while True:
            await asyncio.sleep(poll_interval_s)
            new_files = await asyncio.to_thread(self._list_segment_files)
            for name in sorted(new_files):
                table = await asyncio.to_thread(self._load_segment_table, name)
                conn.register("_new_segment", table)
                conn.execute("INSERT INTO segment_rows SELECT * FROM _new_segment")
                rows = _result_to_dicts(
                    conn.execute(
                        f"SELECT * FROM _new_segment {where}"
                        " ORDER BY iteration, ts, group_idx, traj_idx, step_idx",
                        params,
                    )
                )
                conn.unregister("_new_segment")
                self._registered.add(name)
                for row in rows:
                    yield row

    # --- Backend-specific escape hatch (not part of TokenStoreBackend) ---

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Run a read-only DuckDB SQL query over the store's views.

        Available relations: ``segment_rows`` (raw), ``rollouts``,
        ``rollouts_latest``, ``trajectories``, ``labels``. Only a single
        ``SELECT`` (or ``WITH ... SELECT``) statement is allowed; anything
        else is rejected. Prefer ``?`` placeholders with *params* over string
        interpolation.
        """
        duckdb = _require_duckdb()
        conn = self._connection()
        self.refresh()
        self._refresh_labels_source(conn)
        statements = conn.extract_statements(query)
        if len(statements) != 1:
            raise ValueError("sql() accepts exactly one statement")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError(f"sql() only accepts SELECT statements, got {statements[0].type}")
        return self._fetch(query, params or [])


class TokenDB:
    """Thin facade over a token store backend for agent/programmatic access.

    ``TokenDB(log_path)`` opens the default :class:`ParquetSegmentBackend` for
    that run's log directory (local path or cloud URI). All methods delegate
    to the backend's structured read half; :meth:`sql` is the backend-specific
    escape hatch (SELECT-only) and is not part of the portable protocol.
    """

    def __init__(self, log_path_or_backend: str | Path | Storage | ParquetSegmentBackend) -> None:
        if isinstance(log_path_or_backend, ParquetSegmentBackend):
            self._backend = log_path_or_backend
        else:
            self._backend = ParquetSegmentBackend(log_path_or_backend)

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.query`."""
        return self._backend.query(**kwargs)

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
    ) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.get_rollout`."""
        return self._backend.get_rollout(split, iteration, group_idx, traj_idx, run_attempt)

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.search`."""
        return self._backend.search(**kwargs)

    def search_hit_counts(self, **kwargs: Any) -> dict[int, int]:
        """See :meth:`ParquetSegmentReader.search_hit_counts`."""
        return self._backend.search_hit_counts(**kwargs)

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        """See :meth:`ParquetSegmentReader.add_label`."""
        self._backend.add_label(key, label_key, label_value, author=author, note=note)

    def labels(self, **filters: Any) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.labels`."""
        return self._backend.labels(**filters)

    def iter_new(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over newly written rows; see
        :meth:`ParquetSegmentReader.subscribe`."""
        return self._backend.subscribe(**kwargs)

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Backend-specific SQL escape hatch (SELECT-only); see
        :meth:`ParquetSegmentReader.sql`."""
        return self._backend.sql(query, params)
