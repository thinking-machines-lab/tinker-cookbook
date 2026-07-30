"""Conformance tests for the declarative column specs.

The specs in ``schema.py`` are the single source of truth; the SQLite DDL,
the ClickHouse DDL, and the Arrow schema are all rendered from them. These
tests pin the renderings: the SQLite rendering is compared (via PRAGMA
table_info) against a frozen snapshot of the hand-written schema it
replaced, the ClickHouse rendering against a golden string, and the Arrow
rendering against expected field types (skipped when pyarrow is absent,
where the renderer must instead fail with a clear ImportError).
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from tinker_cookbook.capture.store.db import CaptureDB
from tinker_cookbook.capture.store.schema import (
    ANNOTATIONS_COLUMNS,
    ANNOTATIONS_ORDER_BY,
    WIRE_ROWS_COLUMNS,
    WIRE_ROWS_ORDER_BY,
    ListOf,
    Scalar,
    json_encoded_column_names,
    render_arrow_schema,
    render_clickhouse_ddl,
    render_sqlite_ddl,
)

# Frozen snapshot of the hand-written CREATE TABLE statements the rendered
# DDL replaced: a fresh DB must be equivalent on disk.
_FROZEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS wire_rows (
    cursor INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    run_attempt INTEGER,
    split TEXT,
    iteration INTEGER,
    group_idx INTEGER,
    traj_idx INTEGER,
    purpose TEXT,
    sampling_session_id TEXT,
    seq_id INTEGER,
    sample_idx INTEGER,
    policy_version TEXT,
    created_at REAL,
    prompt_tokens TEXT,
    sampled_tokens TEXT,
    logprobs TEXT,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS annotations (
    cursor INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    kind TEXT,
    payload TEXT,
    created_at REAL
);
"""


def _table_info(conn: sqlite3.Connection, table: str) -> list[tuple[str, str, int, int]]:
    """(name, type, notnull, pk) per column, the on-disk column contract."""
    return [
        (row[1], row[2], row[3], row[5])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    ]


def test_sqlite_rendering_matches_frozen_hand_written_schema(tmp_path: Path) -> None:
    """A fresh CaptureDB must produce tables identical (PRAGMA table_info)
    to the hand-written schema the spec rendering replaced."""
    frozen = sqlite3.connect(":memory:")
    frozen.executescript(_FROZEN_SCHEMA)
    db = CaptureDB(tmp_path / "db.sqlite")
    rendered = sqlite3.connect(str(tmp_path / "db.sqlite"))
    for table in ("wire_rows", "annotations"):
        assert _table_info(rendered, table) == _table_info(frozen, table), table
    # The dedup contract: event_id unique, and the wire dedupe index on the
    # full (sampling_session_id, seq_id, sample_idx) key.
    dedupe_cols = [row[2] for row in rendered.execute("PRAGMA index_info(wire_rows_dedupe)")]
    assert dedupe_cols == ["sampling_session_id", "seq_id", "sample_idx"]
    rendered.close()
    frozen.close()
    db.close()


def test_sqlite_ddl_creates_usable_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(render_sqlite_ddl(WIRE_ROWS_COLUMNS, table="wire_rows", primary_key="cursor"))
    conn.execute(
        render_sqlite_ddl(
            ANNOTATIONS_COLUMNS, table="annotations", primary_key="cursor", unique=("event_id",)
        )
    )
    conn.execute("INSERT INTO wire_rows (cursor, run_id) VALUES (1, 'r1')")
    conn.execute("INSERT INTO annotations (cursor, event_id, run_id) VALUES (2, 'e1', 'r1')")
    with pytest.raises(sqlite3.IntegrityError):  # event_id UNIQUE holds
        conn.execute("INSERT INTO annotations (cursor, event_id, run_id) VALUES (3, 'e1', 'r1')")
    with pytest.raises(sqlite3.IntegrityError):  # run_id NOT NULL holds
        conn.execute("INSERT INTO wire_rows (cursor, run_id) VALUES (4, NULL)")
    conn.close()


def test_json_encoded_columns_derived_from_spec() -> None:
    assert json_encoded_column_names(WIRE_ROWS_COLUMNS) == (
        "prompt_tokens",
        "sampled_tokens",
        "logprobs",
        "metadata",
    )
    assert json_encoded_column_names(ANNOTATIONS_COLUMNS) == ("payload",)


def test_clickhouse_ddl_golden() -> None:
    assert render_clickhouse_ddl(
        WIRE_ROWS_COLUMNS, table="wire_rows", order_by=WIRE_ROWS_ORDER_BY
    ) == (
        "CREATE TABLE wire_rows (\n"
        "    cursor Int64,\n"
        "    run_id String,\n"
        "    run_attempt Nullable(Int32),\n"
        "    split Nullable(String),\n"
        "    iteration Nullable(Int32),\n"
        "    group_idx Nullable(Int32),\n"
        "    traj_idx Nullable(Int32),\n"
        "    purpose Nullable(String),\n"
        "    sampling_session_id Nullable(String),\n"
        "    seq_id Nullable(Int64),\n"
        "    sample_idx Nullable(Int32),\n"
        "    policy_version Nullable(String),\n"
        "    created_at Nullable(Float64),\n"
        "    prompt_tokens Array(Int32),\n"
        "    sampled_tokens Array(Int32),\n"
        "    logprobs Array(Float32),\n"
        "    metadata Nullable(String)\n"
        ") ENGINE = MergeTree\n"
        "ORDER BY (run_id, cursor)"
    )
    annotations_ddl = render_clickhouse_ddl(
        ANNOTATIONS_COLUMNS, table="annotations", order_by=ANNOTATIONS_ORDER_BY
    )
    assert "event_id String" in annotations_ddl
    assert "payload Nullable(String)" in annotations_ddl


def test_arrow_rendering() -> None:
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match="pyarrow"):
            render_arrow_schema(WIRE_ROWS_COLUMNS)
        pytest.skip("pyarrow not installed; verified the clear ImportError instead")
    import pyarrow as pa

    schema = render_arrow_schema(WIRE_ROWS_COLUMNS)
    assert schema.names == [col.name for col in WIRE_ROWS_COLUMNS]
    assert schema.field("cursor").type == pa.int64()
    assert not schema.field("run_id").nullable
    assert schema.field("prompt_tokens").type == pa.list_(pa.int32())
    assert schema.field("logprobs").type == pa.list_(pa.float32())
    assert schema.field("metadata").type == pa.string()


def test_spec_shape_sanity() -> None:
    """Both relations start at the shared cursor and share the scope columns."""
    assert WIRE_ROWS_COLUMNS[0] == ANNOTATIONS_COLUMNS[0]
    assert isinstance(WIRE_ROWS_COLUMNS[0].type, Scalar)
    prompt = next(col for col in WIRE_ROWS_COLUMNS if col.name == "prompt_tokens")
    assert isinstance(prompt.type, ListOf)
