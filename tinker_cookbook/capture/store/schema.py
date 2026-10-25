"""Declarative column specs for the capture store: the single source of truth.

The specs (:data:`WIRE_ROWS_COLUMNS`, :data:`ANNOTATIONS_COLUMNS`) declare
every persisted column once; the physical representations are RENDERED from
them (:func:`render_sqlite_ddl` for the local daemon's SQLite tables,
:func:`render_clickhouse_ddl` for a hosted ClickHouse backend,
:func:`render_arrow_schema` for analytics export), so a column is added in
exactly one place and drift between backends is a test failure
(``schema_test.py``).

pyarrow is imported lazily so importing this module never requires it (it is
not a base dependency); :func:`render_arrow_schema` raises a clear
``ImportError`` when it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True)
class Scalar:
    """A scalar logical type.

    ``kind`` is one of: ``string``, ``int32``, ``int64``, ``float32``,
    ``float64``, ``bool``, ``timestamp_us_utc``, or ``json`` (free-form
    JSON-encoded text: stored as TEXT/String, but flagged so storage layers
    know to encode/decode it).
    """

    kind: str


@dataclass(frozen=True)
class ListOf:
    item: LogicalType


@dataclass(frozen=True)
class MapOf:
    key: LogicalType
    value: LogicalType


@dataclass(frozen=True)
class StructOf:
    fields: tuple[Column, ...]


LogicalType = Scalar | ListOf | MapOf | StructOf


@dataclass(frozen=True)
class Column:
    """One column of a persisted relation: name, logical type, nullability.

    ``nullable`` is the LOGICAL contract (may this value be absent?). SQLite
    renders it as the presence/absence of ``NOT NULL``; Arrow expresses it
    directly; ClickHouse cannot wrap composite types in ``Nullable(...)``, so
    nullable lists/maps/structs render non-nullable there and absence is
    represented by emptiness.
    """

    name: str
    type: LogicalType
    nullable: bool = False


_S = Scalar  # local shorthand for the specs below

#: Wire rows: one row per sampled sequence, deduped on
#: ``(sampling_session_id, seq_id, sample_idx)`` when all three are non-null.
#: ``cursor`` is the shared monotone sequence both tables draw from, so
#: ``/stream`` can interleave them in exact insert order.
WIRE_ROWS_COLUMNS: tuple[Column, ...] = (
    Column("cursor", _S("int64")),
    Column("run_id", _S("string")),
    Column("run_attempt", _S("int32"), nullable=True),
    Column("split", _S("string"), nullable=True),
    Column("iteration", _S("int32"), nullable=True),
    Column("group_idx", _S("int32"), nullable=True),
    Column("traj_idx", _S("int32"), nullable=True),
    Column("purpose", _S("string"), nullable=True),
    Column("sampling_session_id", _S("string"), nullable=True),
    Column("seq_id", _S("int64"), nullable=True),
    Column("sample_idx", _S("int32"), nullable=True),
    Column("policy_version", _S("string"), nullable=True),
    Column("created_at", _S("float64"), nullable=True),
    Column("prompt_tokens", ListOf(_S("int32")), nullable=True),
    Column("sampled_tokens", ListOf(_S("int32")), nullable=True),
    Column("logprobs", ListOf(_S("float32")), nullable=True),
    Column("metadata", _S("json"), nullable=True),
)

#: Annotations: free-form events on the shared cursor, deduped on ``event_id``.
ANNOTATIONS_COLUMNS: tuple[Column, ...] = (
    Column("cursor", _S("int64")),
    Column("event_id", _S("string")),
    Column("run_id", _S("string")),
    Column("kind", _S("string"), nullable=True),
    Column("payload", _S("json"), nullable=True),
    Column("created_at", _S("float64"), nullable=True),
)

#: Natural primary sort; a hosted MergeTree ORDER BY derives from it (it
#: mirrors the local ``(run_id, cursor)`` covering indexes).
WIRE_ROWS_ORDER_BY: tuple[str, ...] = ("run_id", "cursor")
ANNOTATIONS_ORDER_BY: tuple[str, ...] = ("run_id", "cursor")


def json_encoded_column_names(columns: tuple[Column, ...]) -> tuple[str, ...]:
    """Columns SQLite stores as JSON-encoded TEXT (composites and ``json``).

    Derived from the spec so ingest/read encode-decode lists cannot drift
    when a column is added.
    """
    return tuple(
        col.name for col in columns if not isinstance(col.type, Scalar) or col.type.kind == "json"
    )


# --- Rendering: SQLite DDL (the local daemon's physical schema) ---

_SQLITE_SCALARS = {
    "string": "TEXT",
    "int32": "INTEGER",
    "int64": "INTEGER",
    "float32": "REAL",
    "float64": "REAL",
    "bool": "INTEGER",
    "timestamp_us_utc": "TEXT",
    "json": "TEXT",
}


def sqlite_type(logical: LogicalType) -> str:
    """Render one logical type as a SQLite column type.

    Composite types (lists, maps, structs) are stored as JSON-encoded TEXT;
    SQLite has no native representation for them.
    """
    if isinstance(logical, Scalar):
        return _SQLITE_SCALARS[logical.kind]
    return "TEXT"


def render_sqlite_ddl(
    columns: tuple[Column, ...],
    *,
    table: str,
    primary_key: str | None = None,
    unique: tuple[str, ...] = (),
) -> str:
    """Render a column spec as a ``CREATE TABLE IF NOT EXISTS`` statement.

    Args:
        columns: The column spec.
        table: Table name.
        primary_key: Column rendered as ``PRIMARY KEY`` (implies NOT NULL in
            intent; SQLite integer primary keys are the rowid).
        unique: Columns rendered with an inline ``UNIQUE`` constraint.
            Multi-column unique/dedup indexes stay separate ``CREATE UNIQUE
            INDEX`` statements at the call site.
    """
    lines = []
    for col in columns:
        line = f"{col.name} {sqlite_type(col.type)}"
        if col.name == primary_key:
            line += " PRIMARY KEY"
        elif not col.nullable:
            line += " NOT NULL"
        if col.name in unique:
            line += " UNIQUE"
        lines.append(line)
    body = ",\n    ".join(lines)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {body}\n)"


# --- Rendering: Arrow (analytics export; pyarrow optional) ---

_ARROW_SCALARS = {
    "string": lambda pa: pa.string(),
    "int32": lambda pa: pa.int32(),
    "int64": lambda pa: pa.int64(),
    "float32": lambda pa: pa.float32(),
    "float64": lambda pa: pa.float64(),
    "bool": lambda pa: pa.bool_(),
    "timestamp_us_utc": lambda pa: pa.timestamp("us", tz="UTC"),
    "json": lambda pa: pa.string(),
}


def _arrow_type(logical: LogicalType, pa: Any) -> Any:
    if isinstance(logical, Scalar):
        return _ARROW_SCALARS[logical.kind](pa)
    if isinstance(logical, ListOf):
        return pa.list_(_arrow_type(logical.item, pa))
    if isinstance(logical, MapOf):
        return pa.map_(_arrow_type(logical.key, pa), _arrow_type(logical.value, pa))
    if isinstance(logical, StructOf):
        return pa.struct(
            [
                pa.field(col.name, _arrow_type(col.type, pa), nullable=col.nullable)
                for col in logical.fields
            ]
        )
    raise TypeError(f"unknown logical type {logical!r}")


def render_arrow_schema(columns: tuple[Column, ...]) -> pa.Schema:
    """Render a column spec as an Arrow schema.

    Raises:
        ImportError: If pyarrow is not installed (it is an optional
            dependency; only this renderer needs it).
    """
    try:
        import pyarrow as pa
    except ImportError as e:
        raise ImportError(
            "render_arrow_schema requires pyarrow, which is not installed; "
            "install pyarrow to render Arrow schemas (the capture store "
            "itself does not need it)"
        ) from e
    return pa.schema(
        [pa.field(col.name, _arrow_type(col.type, pa), nullable=col.nullable) for col in columns]
    )


# --- Rendering: ClickHouse DDL (hosted-backend parity; pure strings) ---

_CLICKHOUSE_SCALARS = {
    "string": "String",
    "int32": "Int32",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "bool": "Bool",
    "timestamp_us_utc": "DateTime64(6, 'UTC')",
    "json": "String",
}


def clickhouse_type(logical: LogicalType, *, nullable: bool = False) -> str:
    """Render one logical type as a ClickHouse type string.

    Only scalars can be ``Nullable(...)`` in ClickHouse; nullable composites
    render non-nullable and use emptiness for absence (see :class:`Column`).
    """
    if isinstance(logical, Scalar):
        base = _CLICKHOUSE_SCALARS[logical.kind]
        return f"Nullable({base})" if nullable else base
    if isinstance(logical, ListOf):
        return f"Array({clickhouse_type(logical.item)})"
    if isinstance(logical, MapOf):
        return f"Map({clickhouse_type(logical.key)}, {clickhouse_type(logical.value)})"
    if isinstance(logical, StructOf):
        fields = ", ".join(
            f"{col.name} {clickhouse_type(col.type, nullable=col.nullable)}"
            for col in logical.fields
        )
        return f"Tuple({fields})"
    raise TypeError(f"unknown logical type {logical!r}")


def render_clickhouse_ddl(
    columns: tuple[Column, ...],
    *,
    table: str,
    order_by: tuple[str, ...],
    engine: str = "MergeTree",
) -> str:
    """Render a column spec as a ClickHouse ``CREATE TABLE`` statement.

    This is what a hosted backend would run; locally it is only a string
    (no ClickHouse dependency), pinned by a golden test so it cannot drift
    from the SQLite schema.
    """
    body = ",\n    ".join(
        f"{col.name} {clickhouse_type(col.type, nullable=col.nullable)}" for col in columns
    )
    order = f"({', '.join(order_by)})" if order_by else "tuple()"
    return f"CREATE TABLE {table} (\n    {body}\n) ENGINE = {engine}\nORDER BY {order}"
