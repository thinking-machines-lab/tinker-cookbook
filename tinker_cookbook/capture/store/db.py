"""SQLite (WAL) storage for the capture store daemon.

Two tables share one monotone cursor sequence, so ``/stream`` can interleave
wire rows and annotations in exact insert order and resume from any cursor:

- ``wire_rows``: one row per sampled sequence, deduped on
  ``(sampling_session_id, seq_id, sample_idx)`` when all three are non-null
  (SQLite unique indexes treat NULLs as distinct, which gives exactly the
  "dedupe only when fully keyed" semantics).
- ``annotations``: free-form events, deduped on ``event_id``.

There is no run registration: a run exists once rows or annotations carrying
its ``run_id`` arrive.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinker_cookbook.capture.store.schema import (
    ANNOTATIONS_COLUMNS,
    WIRE_ROWS_COLUMNS,
    Column,
    ListOf,
    Scalar,
    json_encoded_column_names,
    render_sqlite_ddl,
)

# Rows and annotations arrive as arbitrary JSON from the HTTP boundary and
# leave as JSON decoded from SQLite; Any is the honest value type there.
JsonDict = dict[str, Any]

# The CREATE TABLE statements are rendered from the declarative column specs
# in schema.py (the single source of truth for the store's columns); only the
# storage-mechanism pieces (shared cursor sequence, dedup/covering indexes)
# are stated here.
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS seq (value INTEGER NOT NULL);
{render_sqlite_ddl(WIRE_ROWS_COLUMNS, table="wire_rows", primary_key="cursor")};
CREATE UNIQUE INDEX IF NOT EXISTS wire_rows_dedupe
    ON wire_rows (sampling_session_id, seq_id, sample_idx);
CREATE INDEX IF NOT EXISTS wire_rows_run ON wire_rows (run_id, cursor);
{render_sqlite_ddl(ANNOTATIONS_COLUMNS, table="annotations", primary_key="cursor", unique=("event_id",))};
CREATE INDEX IF NOT EXISTS annotations_run ON annotations (run_id, cursor);
"""

_WIRE_JSON_FIELDS = json_encoded_column_names(WIRE_ROWS_COLUMNS)
# SQLite binds integers as signed 64-bit; Python ints are unbounded, so an
# out-of-range value passes a type check but raises OverflowError at bind.
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_INT_RANGES = {
    "int32": (-(2**31), 2**31 - 1),
    "int64": (_INT64_MIN, _INT64_MAX),
}


def _scalar_kinds(columns: tuple[Column, ...], *, exclude: tuple[str, ...]) -> dict[str, str]:
    """column name -> declared scalar kind, for the validated scalar columns."""
    return {
        col.name: col.type.kind
        for col in columns
        if isinstance(col.type, Scalar) and col.type.kind != "json" and col.name not in exclude
    }


_WIRE_FIELD_KINDS = _scalar_kinds(WIRE_ROWS_COLUMNS, exclude=("cursor",))
_ANNOTATION_FIELD_KINDS = _scalar_kinds(ANNOTATIONS_COLUMNS, exclude=("cursor", "event_id"))

#: List-typed wire columns -> declared element kind (e.g. prompt_tokens ->
#: int32). Derived from the spec so composite validation cannot drift.
#: ``json``-kind columns (metadata, payload) stay free-form by design.
_WIRE_LIST_KINDS: dict[str, str] = {
    col.name: col.type.item.kind
    for col in WIRE_ROWS_COLUMNS
    if isinstance(col.type, ListOf) and isinstance(col.type.item, Scalar)
}


def _validate_column(label: str, field_name: str, value: object, kind: str) -> None:
    """Enforce the schema's declared scalar type (and range) for one value.

    A merely-scalar check would let SQLite's type affinity store e.g.
    ``"iteration": "twelve"`` as TEXT in an INTEGER column, producing rows
    the integer-typed query filters and response shapes can never serve.
    ``None`` is allowed everywhere here (absent fields; ``run_id`` has its
    own coercion to ``"unattributed"``).
    """
    if value is None:
        return
    if kind in _INT_RANGES:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{label} field {field_name!r} must be an integer, got {type(value).__name__}"
            )
        low, high = _INT_RANGES[kind]
        if not low <= value <= high:
            bits = "signed 64-bit" if kind == "int64" else "signed 32-bit"
            raise ValueError(f"{label} field {field_name!r} exceeds the {bits} integer range")
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(
                f"{label} field {field_name!r} must be a string, got {type(value).__name__}"
            )
    elif kind in ("float32", "float64"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{label} field {field_name!r} must be a number, got {type(value).__name__}"
            )
        if isinstance(value, int) and not _INT64_MIN <= value <= _INT64_MAX:
            # SQLite binds Python ints as integers regardless of column
            # affinity, so an unbounded int still overflows at bind time.
            raise ValueError(
                f"{label} field {field_name!r} exceeds the signed 64-bit integer range"
            )
    else:  # pragma: no cover - no other kinds are declared in the specs
        raise ValueError(f"{label} field {field_name!r} has unsupported kind {kind!r}")


def _validate_list(label: str, field_name: str, value: object, item_kind: str) -> None:
    """Enforce a ListOf column: a list whose elements match the declared kind.

    Accepting any JSON here would store shapes (e.g. a dict in
    ``prompt_tokens``, strings in ``sampled_tokens``) that the declared
    ``List<Int32>``/``List<Float32>`` schemas, and therefore the rendered
    Arrow/ClickHouse representations, can never hold.
    """
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"{label} field {field_name!r} must be a list, got {type(value).__name__}")
    for idx, item in enumerate(value):
        element = f"{field_name}[{idx}]"
        if item is None:
            raise ValueError(f"{label} field {element!r} must not be null")
        _validate_column(label, element, item, item_kind)


_WIRE_FIELDS = tuple(
    col.name
    for col in WIRE_ROWS_COLUMNS
    if col.name != "cursor" and col.name not in _WIRE_JSON_FIELDS
)
_FILTER_FIELDS = ("split", "iteration", "group_idx", "traj_idx", "purpose")


@dataclass
class IngestResult:
    """Counts from one ingest batch."""

    inserted: int = 0
    deduped: int = 0


@dataclass
class RunSummary:
    """Aggregate view of one run (no registration; derived from rows)."""

    run_id: str
    latest_attempt: int | None = None
    num_wire_rows: int = 0
    num_annotations: int = 0

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "run_id": self.run_id,
            "latest_attempt": self.latest_attempt,
            "num_wire_rows": self.num_wire_rows,
            "num_annotations": self.num_annotations,
        }


@dataclass
class StreamEvent:
    """One event on the shared cursor sequence."""

    cursor: int
    event_type: str  # "wire_row" | "annotation"
    data: JsonDict = field(default_factory=dict)


class CaptureDB:
    """Thread-safe SQLite-backed capture store."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        if self._conn.execute("SELECT COUNT(*) FROM seq").fetchone()[0] == 0:
            self._conn.execute("INSERT INTO seq (value) VALUES (0)")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Commit on success, roll back on any failure. Caller holds the lock."""
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    def _next_cursor(self) -> int:
        self._conn.execute("UPDATE seq SET value = value + 1")
        return int(self._conn.execute("SELECT value FROM seq").fetchone()[0])

    # ── ingest ────────────────────────────────────────────────────────

    def ingest_wire(self, rows: list[JsonDict]) -> IngestResult:
        """Insert wire rows, deduping fully-keyed duplicates. Idempotent.

        The batch is atomic: a malformed row (e.g. a non-JSON-serializable
        value) rolls back every earlier row AND the cursor sequence, so a
        failed ingest leaves no partial batch visible and the client can
        retry the whole request idempotently.
        """
        result = IngestResult()
        with self._lock, self._transaction():
            for row in rows:
                # Scalar columns must match the schema's declared types; a
                # collection, wrong-typed scalar, or out-of-range integer
                # here would otherwise be stored via type affinity or raise
                # from SQLite's binding as a 500.
                for field_name, kind in _WIRE_FIELD_KINDS.items():
                    _validate_column("wire row", field_name, row.get(field_name), kind)
                for field_name, item_kind in _WIRE_LIST_KINDS.items():
                    _validate_list("wire row", field_name, row.get(field_name), item_kind)
                cursor = self._next_cursor()
                values: list[object] = [cursor]
                values.extend(
                    row.get(f) if f != "run_id" else (row.get("run_id") or "unattributed")
                    for f in _WIRE_FIELDS
                )
                values.extend(
                    json.dumps(row[f]) if row.get(f) is not None else None
                    for f in _WIRE_JSON_FIELDS
                )
                placeholders = ",".join("?" * len(values))
                inserted = self._conn.execute(
                    f"INSERT OR IGNORE INTO wire_rows "
                    f"(cursor,{','.join(_WIRE_FIELDS)},{','.join(_WIRE_JSON_FIELDS)}) "
                    f"VALUES ({placeholders})",
                    values,
                ).rowcount
                if inserted:
                    result.inserted += 1
                else:
                    result.deduped += 1
        return result

    def ingest_annotations(self, annotations: list[JsonDict]) -> IngestResult:
        """Insert annotations, deduping on ``event_id``. Idempotent.

        Atomic per batch, like :meth:`ingest_wire`.
        """
        result = IngestResult()
        with self._lock, self._transaction():
            for ann in annotations:
                # Validate up front: INSERT OR IGNORE would otherwise
                # suppress the NOT NULL violation for a null event_id and
                # silently drop the row as "deduped".
                event_id = ann.get("event_id")
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError(
                        f"annotation requires a non-empty string event_id, got {event_id!r}"
                    )
                for field_name, kind in _ANNOTATION_FIELD_KINDS.items():
                    _validate_column("annotation", field_name, ann.get(field_name), kind)
                cursor = self._next_cursor()
                payload = ann.get("payload")
                inserted = self._conn.execute(
                    "INSERT OR IGNORE INTO annotations "
                    "(cursor, event_id, run_id, kind, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        cursor,
                        event_id,
                        ann.get("run_id") or "unattributed",
                        ann.get("kind"),
                        json.dumps(payload) if payload is not None else None,
                        ann.get("created_at"),
                    ),
                ).rowcount
                if inserted:
                    result.inserted += 1
                else:
                    result.deduped += 1
        return result

    # ── query ─────────────────────────────────────────────────────────

    def list_runs(self) -> list[RunSummary]:
        """Aggregate runs over both tables. Latest attempt = MAX(run_attempt)."""
        with self._lock:
            summaries: dict[str, RunSummary] = {}
            for row in self._conn.execute(
                "SELECT run_id, MAX(run_attempt) AS latest, COUNT(*) AS n "
                "FROM wire_rows GROUP BY run_id"
            ):
                summaries[row["run_id"]] = RunSummary(
                    run_id=row["run_id"], latest_attempt=row["latest"], num_wire_rows=row["n"]
                )
            for row in self._conn.execute(
                "SELECT run_id, COUNT(*) AS n FROM annotations GROUP BY run_id"
            ):
                summary = summaries.setdefault(row["run_id"], RunSummary(run_id=row["run_id"]))
                summary.num_annotations = row["n"]
        return sorted(summaries.values(), key=lambda s: s.run_id)

    def query_rows(
        self,
        run_id: str,
        *,
        filters: dict[str, str | int] | None = None,
        cursor: int = 0,
        limit: int = 1000,
    ) -> list[JsonDict]:
        """Wire rows for a run with equality filters, cursor-paged."""
        clauses = ["run_id = ?", "cursor > ?"]
        params: list[object] = [run_id, cursor]
        for key, value in (filters or {}).items():
            if key not in _FILTER_FIELDS:
                raise ValueError(f"Unsupported filter: {key}")
            clauses.append(f"{key} = ?")
            params.append(value)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM wire_rows WHERE {' AND '.join(clauses)} ORDER BY cursor LIMIT ?",
                params,
            ).fetchall()
        return [self._wire_row_to_dict(row) for row in rows]

    def query_annotations(
        self,
        run_id: str,
        *,
        kind: str | None = None,
        cursor: int = 0,
        limit: int = 1000,
    ) -> list[JsonDict]:
        """Annotations for a run, optionally kind-filtered, cursor-paged.

        The read-path counterpart of ``query_rows``: train-op records land
        here (they carry no sampled sequences), and without this they were
        reachable only via ``/stream``.
        """
        clauses = ["run_id = ?", "cursor > ?"]
        params: list[object] = [run_id, cursor]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM annotations WHERE {' AND '.join(clauses)} ORDER BY cursor LIMIT ?",
                params,
            ).fetchall()
        return [self._annotation_to_dict(row) for row in rows]

    def stream_events(
        self, run_id: str | None, cursor: int = 0, limit: int = 1000
    ) -> list[StreamEvent]:
        """Interleaved wire rows and annotations in insert (cursor) order."""
        run_clause = "AND run_id = ?" if run_id is not None else ""
        params_wire: list[object] = [cursor]
        if run_id is not None:
            params_wire.append(run_id)
        params_wire.append(limit)
        with self._lock:
            wire = self._conn.execute(
                f"SELECT * FROM wire_rows WHERE cursor > ? {run_clause} ORDER BY cursor LIMIT ?",
                params_wire,
            ).fetchall()
            anns = self._conn.execute(
                f"SELECT * FROM annotations WHERE cursor > ? {run_clause} ORDER BY cursor LIMIT ?",
                params_wire,
            ).fetchall()
        events = [
            StreamEvent(row["cursor"], "wire_row", self._wire_row_to_dict(row)) for row in wire
        ] + [
            StreamEvent(row["cursor"], "annotation", self._annotation_to_dict(row)) for row in anns
        ]
        events.sort(key=lambda e: e.cursor)
        return events[:limit]

    @staticmethod
    def _wire_row_to_dict(row: sqlite3.Row) -> JsonDict:
        out = {key: row[key] for key in row.keys()}  # noqa: SIM118 - sqlite3.Row iterates values
        for f in _WIRE_JSON_FIELDS:
            if out.get(f) is not None:
                out[f] = json.loads(out[f])
        return out

    @staticmethod
    def _annotation_to_dict(row: sqlite3.Row) -> JsonDict:
        out = {key: row[key] for key in row.keys()}  # noqa: SIM118 - sqlite3.Row iterates values
        if out.get("payload") is not None:
            out["payload"] = json.loads(out["payload"])
        return out
