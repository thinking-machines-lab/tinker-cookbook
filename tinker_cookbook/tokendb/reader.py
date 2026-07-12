"""DuckDB read path for the parquet segment token store.

:class:`ParquetSegmentReader` is the read half of
:class:`~tinker_cookbook.tokendb.writer.ParquetSegmentBackend`: an in-memory
DuckDB connection over the immutable segment files, with views for browsing
(``rollouts`` / ``rollouts_latest`` / ``trajectories`` / ``labels`` /
``runs``, plus the promoted ``correct`` / ``parse_errors`` /
``context_overflows``) and the structured ``TokenStoreBackend`` read methods
(``query`` / ``get_rollout`` / ``search`` / ``subscribe`` / ``add_label`` /
``labels``).

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
from tinker_cookbook.tokendb.interface import TokenStoreBackend
from tinker_cookbook.tokendb.writer import (
    RUN_ATTEMPTS_PATH,
    RUN_JSON_PATH,
    SEGMENTS_DIR,
    TOKENS_DIR,
    ParquetSegmentBackend,
)

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

LABELS_PATH = f"{TOKENS_DIR}/labels.jsonl"

DEFAULT_SUBSCRIBE_POLL_INTERVAL_S = 2.0

_LABEL_KEY_FIELDS = ("run_id", "split", "iteration", "group_idx", "traj_idx", "step_idx")
LABEL_FILTER_FIELDS = frozenset(_LABEL_KEY_FIELDS) | {"label_key", "author"}

_LABEL_COLUMNS_SQL = (
    "{run_id: 'VARCHAR', split: 'VARCHAR', iteration: 'INTEGER', group_idx: 'INTEGER', "
    "traj_idx: 'INTEGER', step_idx: 'INTEGER', label_key: 'VARCHAR', label_value: 'JSON', "
    "author: 'VARCHAR', ts: 'TIMESTAMP', note: 'VARCHAR'}"
)

# Casts a raw (all-strings-and-ints) label relation to the typed label columns.
LABELS_CAST_SELECT = (
    "(SELECT run_id, split, iteration::INTEGER AS iteration,"
    " group_idx::INTEGER AS group_idx, traj_idx::INTEGER AS traj_idx,"
    " step_idx::INTEGER AS step_idx, label_key, label_value::JSON AS label_value,"
    " author, ts::TIMESTAMP AS ts, note FROM {src})"
)

# Last write (by ts) wins per (identity, label_key); tombstones (null
# label_value) win first, then drop out of the view.
LABELS_DEDUP_VIEW_SQL = """
    CREATE OR REPLACE VIEW labels AS
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

# Trajectory-grain aggregation over the row-grain `rollouts` view (the
# `trajectories()` backend method). Matches the `trajectories` view but keeps
# the browsing columns the viewer's feed needs (tags, ac preview, superseded).
_TRAJECTORIES_SQL = """
    WITH filtered AS (SELECT * FROM rollouts {where})
    SELECT run_id, run_attempt, split, iteration, group_idx, traj_idx,
           count(*) AS n_steps,
           sum(len(ac_tokens))::BIGINT AS n_ac_tokens,
           any_value(total_reward) AS total_reward,
           any_value(final_reward) AS final_reward,
           arg_max(stop_reason, step_idx) AS stop_reason,
           any_value(filtered_reason) AS filtered_reason,
           any_value(env_row_id) AS env_row_id,
           any_value(tags) AS tags,
           any_value(source) AS source,
           any_value(sampling_client_step) AS sampling_client_step,
           arg_max(ac_text, step_idx) AS ac_preview,
           bool_or(superseded) AS superseded,
           min(ts) AS ts
    FROM filtered
    GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
    ORDER BY iteration DESC, ts DESC, group_idx, traj_idx
    LIMIT {limit} OFFSET {offset}
"""

# Dashboard aggregates (the `dashboard_stats()` backend method): one cheap
# pass over the store.
_DASHBOARD_TOTALS_SQL = """
    SELECT count(*) AS n_rows,
           count(*) FILTER (WHERE source = 'filtered') AS n_filtered_rows,
           max(iteration) FILTER (WHERE iteration >= 0) AS latest_iteration
    FROM rollouts
"""
# Per-iteration mean of per-trajectory total_reward, excluding filtered rows
# (their placeholder reward of 0.0 would skew the mean).
_DASHBOARD_SERIES_SQL = """
    SELECT iteration, avg(total_reward) AS mean_total_reward
    FROM (
        SELECT iteration, any_value(total_reward) AS total_reward
        FROM rollouts
        WHERE iteration >= 0 AND source <> 'filtered'
        GROUP BY run_attempt, split, iteration, group_idx, traj_idx
    )
    GROUP BY iteration
    ORDER BY iteration DESC
    LIMIT {series_len}
"""


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


# --- runs table: per-attempt run records with redacted config ---

#: Case-insensitive key-substring patterns whose values are redacted from
#: ``config_json`` in the ``runs`` view. Deliberately aggressive (substring,
#: not exact match): a token store can end up on shared storage, so leaking
#: an API key is much worse than over-redacting. Known false positives (e.g.
#: ``max_tokens`` matches ``token``) are acceptable because the hot config
#: dimensions are promoted to typed columns before redaction.
CONFIG_REDACT_PATTERNS = ("key", "token", "secret", "password", "credential")

_REDACTED = "[redacted]"

#: Typed columns of the ``runs`` view extracted from the run context/config
#: (NULL when absent): name -> coercion.
RUNS_TYPED_COLUMNS: dict[str, type] = {
    "temperature": float,
    "max_tokens": int,
    "renderer_name": str,
    "lora_rank": int,
    "seed": int,
    "group_size": int,
    "loss_fn": str,
    "learning_rate": float,
}


def redact_config(value: Any) -> Any:
    """Recursively redact sensitive values from a config structure.

    Any dict entry whose key contains one of :data:`CONFIG_REDACT_PATTERNS`
    (case-insensitive substring) has its value replaced with ``"[redacted]"``;
    nested dicts and lists are walked recursively.
    """
    if isinstance(value, dict):
        return {
            key: (
                _REDACTED
                if any(pattern in str(key).lower() for pattern in CONFIG_REDACT_PATTERNS)
                else redact_config(entry)
            )
            for key, entry in value.items()
        }
    if isinstance(value, list):
        return [redact_config(entry) for entry in value]
    return value


def _find_config_value(config: Any, key: str) -> Any:
    """Breadth-first search for *key* in a nested dict structure.

    Shallowest match wins, so a top-level ``temperature`` beats one buried in
    a sub-config. Returns ``None`` when absent.
    """
    queue: list[Any] = [config]
    while queue:
        next_queue: list[Any] = []
        for node in queue:
            if not isinstance(node, dict):
                continue
            if key in node:
                return node[key]
            next_queue.extend(node.values())
        queue = next_queue
    return None


def _run_row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One ``runs`` row from one run.json / run-attempts.jsonl payload."""
    context = payload.get("context") or {}
    row: dict[str, Any] = {
        "run_id": payload.get("run_id"),
        "run_attempt": payload.get("run_attempt"),
        "model_name": context.get("model_name"),
        "recipe_name": context.get("recipe_name"),
        "started_at": payload.get("updated_at"),
    }
    for column, coerce in RUNS_TYPED_COLUMNS.items():
        value = _find_config_value(context, column)
        try:
            row[column] = coerce(value) if value is not None else None
        except (TypeError, ValueError):
            row[column] = None
    row["config_json"] = json.dumps(redact_config(context), default=str)
    return row


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


# --- Shared query building (used by the single-run reader and the cross-run
# --- registry backend; see registry_backend.py) ---


def build_where(
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
    attr_eq: Mapping[str, str] | None = None,
    metric_min: Mapping[str, float] | None = None,
    metric_max: Mapping[str, float] | None = None,
) -> tuple[str, list[Any]]:
    """Build a parameterized ``WHERE`` clause from the structured row filters."""
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
            "(regexp_matches(coalesce(ob_text, ''), ?) OR regexp_matches(coalesce(ac_text, ''), ?))"
        )
        params.extend([text_regex, text_regex])
    # Structured map filters. Keys bind as parameters (map access on a
    # missing key yields NULL, so these clauses also drop rows lacking
    # the key). TRY_CAST guards the numeric comparison; note DuckDB
    # orders NaN above every number, so a NaN metric passes metric_min.
    for key, value in (attr_eq or {}).items():
        clauses.append("attrs[?] = ?")
        params.extend([str(key), str(value)])
    for key, value in (metric_min or {}).items():
        clauses.append("TRY_CAST(metrics[?] AS DOUBLE) >= ?")
        params.extend([str(key), float(value)])
    for key, value in (metric_max or {}).items():
        clauses.append("TRY_CAST(metrics[?] AS DOUBLE) <= ?")
        params.extend([str(key), float(value)])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def run_search(
    fetch: Any,
    *,
    regex: str | None = None,
    fields: Sequence[str] = ("ac_text", "ob_text"),
    token_subsequence: Sequence[int] | None = None,
    limit: int | None = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Shared :meth:`~ParquetSegmentReader.search` implementation.

    *fetch* is a ``(sql, params) -> list[dict]`` callable bound to a
    connection whose ``rollouts`` view is current (the caller refreshes).
    """
    string_fields = {"ac_text", "ob_text", "logs"}
    map_fields = {"metrics", "attrs", "token_metrics"}
    bad = set(fields) - string_fields - map_fields
    if bad:
        raise ValueError(f"Unsupported search fields: {sorted(bad)}")
    where, params = build_where(**filters)
    clauses = [where[len("WHERE ") :]] if where else []
    if regex is not None:
        field_matches = []
        for f in fields:
            if f in map_fields:
                # regexp_matches on a MAP is a BinderException; match keys.
                field_matches.append(
                    "EXISTS (SELECT 1 FROM unnest(map_keys("
                    f"{f})) AS _k(key) WHERE regexp_matches(_k.key, ?))"
                )
            else:
                field_matches.append(f"regexp_matches(coalesce({f}, ''), ?)")
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
    rows = fetch(sql, params)
    if token_subsequence:
        needle = [int(t) for t in token_subsequence]
        rows = [row for row in rows if _contains_subsequence(row["ac_tokens"], needle)]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def get_rollout_query(
    split: str,
    iteration: int,
    group_idx: int,
    traj_idx: int,
    run_attempt: int | None,
    run_id: str | None,
) -> tuple[str, list[Any]]:
    """The (sql, params) fetching one trajectory's step rows from ``rollouts``.

    ``run_attempt=None`` selects the latest attempt that produced rows for
    this trajectory (within ``run_id``, when given).
    """
    ident = "split = ? AND iteration = ? AND group_idx = ? AND traj_idx = ?"
    ident_params: list[Any] = [split, iteration, group_idx, traj_idx]
    if run_id is not None:
        ident += " AND run_id = ?"
        ident_params.append(run_id)
    sql = f"SELECT * FROM rollouts WHERE {ident}"
    params = list(ident_params)
    if run_attempt is None:
        sql += f" AND run_attempt = (SELECT max(run_attempt) FROM rollouts WHERE {ident})"
        params += ident_params
    else:
        sql += " AND run_attempt = ?"
        params.append(run_attempt)
    sql += " ORDER BY step_idx"
    return sql, params


def run_attempt_payloads(storage: Storage) -> list[dict[str, Any]]:
    """All run-attempt payloads for one store, one per (run_id, run_attempt).

    Reads ``run-attempts.jsonl`` (one line appended per attempt), falling
    back to ``run.json`` (latest attempt only) for stores written before
    the append-per-attempt record existed. Deduped by
    ``(run_id, run_attempt)``, last line wins.
    """
    payloads: dict[tuple[Any, Any], dict[str, Any]] = {}
    if storage.exists(RUN_ATTEMPTS_PATH):
        for line in storage.read(RUN_ATTEMPTS_PATH).decode().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads[(payload.get("run_id"), payload.get("run_attempt"))] = payload
    if not payloads and storage.exists(RUN_JSON_PATH):
        try:
            payload = json.loads(storage.read(RUN_JSON_PATH).decode())
            if isinstance(payload, dict):
                payloads[(payload.get("run_id"), payload.get("run_attempt"))] = payload
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return [payloads[key] for key in sorted(payloads, key=lambda k: (str(k[0]), str(k[1])))]


def runs_arrow_schema() -> Any:
    """Arrow schema of the ``runs`` view (typed config columns + config_json)."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("run_attempt", pa.int32()),
            pa.field("model_name", pa.string()),
            pa.field("recipe_name", pa.string()),
            pa.field("started_at", pa.string()),
            pa.field("temperature", pa.float64()),
            pa.field("max_tokens", pa.int64()),
            pa.field("renderer_name", pa.string()),
            pa.field("lora_rank", pa.int64()),
            pa.field("seed", pa.int64()),
            pa.field("group_size", pa.int64()),
            pa.field("loss_fn", pa.string()),
            pa.field("learning_rate", pa.float64()),
            pa.field("config_json", pa.string()),
        ]
    )


def load_schema_card(storage: Storage) -> dict[str, Any]:
    """Aggregate one store's observed-keys manifest lines into a card.

    Unions ``metrics_keys`` / ``attrs_keys`` / ``token_metrics_keys`` /
    ``tags`` across every manifest line of every writer (cheap: manifest
    lines only, no segment bytes). ``keys_truncated`` is true if any
    contributing line hit the writer's per-line key cap, i.e. the lists
    may be incomplete. Lines from writers that predate observed-keys
    manifests contribute nothing.
    """
    union: dict[str, set[str]] = {
        "metrics_keys": set(),
        "attrs_keys": set(),
        "token_metrics_keys": set(),
        "tags": set(),
    }
    truncated = False
    try:
        names = [
            name
            for name in storage.list_dir(TOKENS_DIR)
            if name.startswith("manifest-") and name.endswith(".jsonl")
        ]
    except FileNotFoundError:
        names = []
    for name in names:
        try:
            lines = storage.read(f"{TOKENS_DIR}/{name}").decode().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in union:
                values = entry.get(field)
                if isinstance(values, list):
                    union[field].update(str(v) for v in values)
            truncated = truncated or bool(entry.get("keys_truncated"))
    return {
        **{field: sorted(values) for field, values in union.items()},
        "keys_truncated": truncated,
    }


def append_label(
    storage: Storage,
    key: Mapping[str, Any],
    label_key: str,
    label_value: Any,
    *,
    author: str,
    note: str | None = None,
) -> None:
    """Append one annotation record to a store's ``labels.jsonl``.

    See :meth:`ParquetSegmentReader.add_label` for the key/tombstone
    semantics; this is the shared write path.
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
    storage.append(LABELS_PATH, (json.dumps(record, default=str) + "\n").encode())
    # FsspecStorage stages append()s locally and only uploads on flush();
    # nothing else ever flushes this storage, so without this a label
    # written against a gs:///s3:// store would silently never reach the
    # store. LocalStorage.flush() is a no-op, and FsspecStorage re-uploads
    # only files with staged appends (here: labels.jsonl, which stays small).
    storage.flush()


def labels_arrow_table(records: Sequence[Mapping[str, Any]]) -> Any:
    """Label records as an arrow table (``label_value`` JSON-encoded).

    An explicit schema keeps the shape stable for empty record lists.
    """
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("split", pa.string()),
            pa.field("iteration", pa.int64()),
            pa.field("group_idx", pa.int64()),
            pa.field("traj_idx", pa.int64()),
            pa.field("step_idx", pa.int64()),
            pa.field("label_key", pa.string()),
            pa.field("label_value", pa.string()),
            pa.field("author", pa.string()),
            pa.field("ts", pa.string()),
            pa.field("note", pa.string()),
        ]
    )
    return pa.Table.from_pylist(
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
        ],
        schema=schema,
    )


def create_derived_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the views derived from ``rollouts`` (shared across backends).

    The caller creates ``rollouts`` (whose ``superseded`` window differs
    between the single-run reader and the cross-run registry backend);
    everything defined on top of it is identical.
    """
    # Dedup convenience: only the latest attempt per superseded partition.
    conn.execute("CREATE VIEW rollouts_latest AS SELECT * FROM rollouts WHERE NOT superseded")
    # Promoted views: thin CREATE VIEW sugar over `rollouts` for the
    # hottest metrics keys, so common questions need no map syntax.
    # Retroactive and reversible (zero write cost); the key choices match
    # what the cookbook envs actually emit:
    # - `correct`: ProblemEnv/code_rl/search_tool/eval benchmarks emit
    #   metrics['correct']; group-level correctness lands as
    #   metrics['group/correct'] (which wins when both are present).
    conn.execute(
        """
        CREATE VIEW correct AS
        SELECT *, coalesce(metrics['group/correct'], metrics['correct']) AS correct
        FROM rollouts
        WHERE metrics['correct'] IS NOT NULL OR metrics['group/correct'] IS NOT NULL
        """
    )
    # - `parse_errors`: message-env turns whose response failed renderer
    #   parsing (metrics['parse_error'] = 1.0 in rl/message_env.py).
    conn.execute(
        """
        CREATE VIEW parse_errors AS
        SELECT * FROM rollouts WHERE metrics['parse_error'] >= 1.0
        """
    )
    # - `context_overflows`: turns that hit max_tokens — stop_reason
    #   'length', or the message-env's metrics['max_tokens_reached'].
    conn.execute(
        """
        CREATE VIEW context_overflows AS
        SELECT * FROM rollouts
        WHERE stop_reason = 'length' OR metrics['max_tokens_reached'] >= 1.0
        """
    )


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
        create_derived_views(conn)

    def _refresh_runs_source(self, conn: duckdb.DuckDBPyConnection) -> None:
        """(Re)create the ``runs`` view: one row per (run_id, run_attempt).

        Typed columns (:data:`RUNS_TYPED_COLUMNS`, NULL when absent) come from
        a breadth-first search of the attempt's recorded context; the full
        context is exposed as ``config_json`` after :func:`redact_config`.
        Re-resolved on every read (a resume can append an attempt while a
        reader is open).
        """
        import pyarrow as pa

        rows = [_run_row_from_payload(payload) for payload in run_attempt_payloads(self._storage)]
        table = pa.Table.from_pylist(rows, schema=runs_arrow_schema())
        conn.execute("DROP VIEW IF EXISTS runs")
        conn.register("_runs_raw", table)
        conn.execute("CREATE VIEW runs AS SELECT * FROM _runs_raw")

    def runs(self) -> list[dict[str, Any]]:
        """The ``runs`` view content: one row per (run_id, run_attempt)."""
        conn = self._connection()
        self._refresh_runs_source(conn)
        return self._fetch("SELECT * FROM runs ORDER BY run_id, run_attempt")

    def schema_card(self) -> dict[str, Any]:
        """Aggregate the observed-keys manifest lines into a per-run card.

        See :func:`load_schema_card` (the shared implementation): unions
        ``metrics_keys`` / ``attrs_keys`` / ``token_metrics_keys`` / ``tags``
        across every manifest line of every writer, with ``keys_truncated``
        set when any contributing line hit the writer's per-line key cap.
        """
        return load_schema_card(self._storage)

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
        ``ac_text``), plus structured map filters — ``attr_eq`` (dict:
        ``attrs[key] = value``), ``metric_min`` / ``metric_max`` (dicts:
        ``metrics[key]`` at least / at most the value; rows lacking the key
        never match). ``latest_only=True`` queries ``rollouts_latest``
        (dropping attempts superseded by a resume). Rows are ordered by
        ``(iteration, ts)`` and include the computed ``superseded`` flag.
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
        """Trajectory-grain aggregation of :meth:`query`.

        One row per (run_id, run_attempt, split, iteration, group_idx,
        traj_idx) with step/token counts, rewards, tags, the last step's
        ``ac_text`` as ``ac_preview``, and the ``superseded`` flag. *filters*
        are the same as :meth:`query` and apply at row grain before
        aggregation, so the semantics match a row-grain query exactly.
        Ordered newest iteration first.
        """
        self.refresh()
        where, params = build_where(**filters)
        if latest_only:
            where = f"{where} AND NOT superseded" if where else "WHERE NOT superseded"
        sql = _TRAJECTORIES_SQL.format(where=where, limit=int(limit), offset=int(offset))
        return self._fetch(sql, params)

    def dashboard_stats(self, *, recent_k: int = 5, series_len: int = 50) -> dict[str, Any]:
        """Store-level aggregates for the multi-run dashboard.

        Returns ``n_rows``, ``n_filtered_rows``, ``latest_iteration`` (max
        iteration >= 0, or None), ``reward_series`` (per-iteration mean of
        per-trajectory ``total_reward`` excluding filtered rows, oldest
        first, at most *series_len* points), and ``mean_recent_reward`` (the
        mean over the last *recent_k* series points, or None).
        """
        self.refresh()
        totals = self._fetch(_DASHBOARD_TOTALS_SQL)[0]
        series = self._fetch(_DASHBOARD_SERIES_SQL.format(series_len=int(series_len)))
        series.reverse()  # oldest -> newest for sparklines
        recent = [
            p["mean_total_reward"]
            for p in series[-int(recent_k) :]
            if p["mean_total_reward"] is not None
        ]
        return {
            "n_rows": totals["n_rows"],
            "n_filtered_rows": totals["n_filtered_rows"],
            "latest_iteration": totals["latest_iteration"],
            "mean_recent_reward": (sum(recent) / len(recent)) if recent else None,
            "reward_series": series,
        }

    def group_traj_idxs(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        """The distinct ``traj_idx`` values of one group, ascending.

        ``run_attempt=None`` spans every attempt (the viewer's sibling
        navigation shows all trajectories that ever existed for the group).
        ``run_id`` is an optional extra filter (a single-run store holds one
        run, so it is rarely needed here).
        """
        self.refresh()
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
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch the full trajectory: all step rows ordered by ``step_idx``.

        ``run_attempt=None`` selects the latest attempt that produced rows for
        this trajectory. ``run_id`` is an optional extra filter (a single-run
        store holds one run). Delta-encoded observations are returned as
        stored; use :func:`reconstruct_full_ob` to expand them.
        """
        self.refresh()
        sql, params = get_rollout_query(split, iteration, group_idx, traj_idx, run_attempt, run_id)
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

        ``regex`` matches over the given ``fields``: string columns
        (``ac_text``, ``ob_text``, ``logs``) are regexp'd directly; MAP
        columns (``metrics``, ``attrs``, ``token_metrics``) are never
        regexp'd as values — the regex matches their **keys** (via
        ``map_keys``), which is the useful discovery question for typed maps.
        ``token_subsequence`` matches a contiguous run of token IDs inside
        ``ac_tokens`` (useful for special tokens with unstable decodings).
        Additional structured ``filters`` are the same as :meth:`query`.
        Results are ordered by ``(iteration, ts)``.
        """
        self.refresh()
        return run_search(
            self._fetch,
            regex=regex,
            fields=fields,
            token_subsequence=token_subsequence,
            limit=limit,
            **filters,
        )

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
        append_label(self._storage, key, label_key, label_value, author=author, note=note)

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
            conn.register("_labels_raw", labels_arrow_table(records))
            source = LABELS_CAST_SELECT.format(src="_labels_raw")
        conn.execute(LABELS_DEDUP_VIEW_SQL.format(source=source))

    def labels(self, **filters: Any) -> list[dict[str, Any]]:
        """Fetch current labels (after last-write-wins and tombstone filtering).

        Filters: ``run_id``, ``split``, ``iteration``, ``group_idx``,
        ``traj_idx``, ``step_idx``, ``label_key``, ``author``. Returned
        ``label_value`` is decoded back to a Python value.
        """
        bad = set(filters) - LABEL_FILTER_FIELDS
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
        where, params = build_where(**filters)
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
        ``rollouts_latest``, ``trajectories``, ``labels``, ``runs`` (one row
        per run attempt with typed config columns and redacted
        ``config_json``), and the promoted views ``correct`` /
        ``parse_errors`` / ``context_overflows``. Only a single ``SELECT``
        (or ``WITH ... SELECT``) statement is allowed; anything else is
        rejected. Prefer ``?`` placeholders with *params* over string
        interpolation.
        """
        duckdb = _require_duckdb()
        conn = self._connection()
        self.refresh()
        self._refresh_labels_source(conn)
        self._refresh_runs_source(conn)
        statements = conn.extract_statements(query)
        if len(statements) != 1:
            raise ValueError("sql() accepts exactly one statement")
        if statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError(f"sql() only accepts SELECT statements, got {statements[0].type}")
        return self._fetch(query, params or [])


class TokenDB:
    """Thin facade over a token store backend for agent/programmatic access.

    ``TokenDB(log_path)`` opens the default :class:`ParquetSegmentBackend` for
    that run's log directory (local path or cloud URI); any other
    :class:`~tinker_cookbook.tokendb.interface.TokenStoreBackend`
    implementation can be passed instead. All methods delegate to the
    backend's structured read half; :meth:`sql` is the backend-specific
    escape hatch (SELECT-only) and is not part of the portable protocol.
    """

    def __init__(self, log_path_or_backend: str | Path | Storage | TokenStoreBackend) -> None:
        if isinstance(log_path_or_backend, (str, Path)):
            self._backend: TokenStoreBackend = ParquetSegmentBackend(log_path_or_backend)
        elif isinstance(log_path_or_backend, TokenStoreBackend):
            self._backend = log_path_or_backend
        else:  # a bare Storage: wrap it in the default backend
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
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.get_rollout`."""
        return self._backend.get_rollout(split, iteration, group_idx, traj_idx, run_attempt, run_id)

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.search`."""
        return self._backend.search(**kwargs)

    def search_hit_counts(self, **kwargs: Any) -> dict[int, int]:
        """See :meth:`ParquetSegmentReader.search_hit_counts`."""
        return self._backend.search_hit_counts(**kwargs)

    def trajectories(self, **kwargs: Any) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.trajectories`."""
        return self._backend.trajectories(**kwargs)

    def dashboard_stats(self, *, recent_k: int = 5, series_len: int = 50) -> dict[str, Any]:
        """See :meth:`ParquetSegmentReader.dashboard_stats`."""
        return self._backend.dashboard_stats(recent_k=recent_k, series_len=series_len)

    def group_traj_idxs(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        run_attempt: int | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        """See :meth:`ParquetSegmentReader.group_traj_idxs`."""
        return self._backend.group_traj_idxs(split, iteration, group_idx, run_attempt, run_id)

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

    def runs(self) -> list[dict[str, Any]]:
        """See :meth:`ParquetSegmentReader.runs`."""
        return self._backend.runs()

    def schema_card(self) -> dict[str, Any]:
        """See :meth:`ParquetSegmentReader.schema_card`."""
        return self._backend.schema_card()

    def iter_new(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over newly written rows; see
        :meth:`ParquetSegmentReader.subscribe`."""
        return self._backend.subscribe(**kwargs)

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Backend-specific SQL escape hatch (SELECT-only); see
        :meth:`ParquetSegmentReader.sql`.

        ``sql()`` is deliberately not part of ``TokenStoreBackend``, so a
        backend without it raises :class:`NotImplementedError`.
        """
        sql_fn = getattr(self._backend, "sql", None)
        if sql_fn is None:
            raise NotImplementedError("this token store backend has no raw SQL escape hatch")
        return sql_fn(query, params)
