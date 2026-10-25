"""Tests for the capture store daemon: DB, HTTP API, and SSE streaming."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from tinker_cookbook.capture.store.app import make_app
from tinker_cookbook.capture.store.db import CaptureDB


def _wire_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "run-1",
        "run_attempt": 0,
        "split": "train",
        "iteration": 1,
        "group_idx": 0,
        "traj_idx": 0,
        "purpose": "rollout",
        "sampling_session_id": "sess-1",
        "seq_id": 1,
        "sample_idx": 0,
        "policy_version": "v1",
        "created_at": 123.0,
        "prompt_tokens": [1, 2],
        "sampled_tokens": [3, 4],
        "logprobs": [-0.5, -0.6],
        "metadata": {"stop_reason": "length"},
    }
    row.update(overrides)
    return row


# ── CaptureDB ─────────────────────────────────────────────────────────


def test_ingest_wire_idempotent(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    assert db.ingest_wire([_wire_row()]).inserted == 1
    result = db.ingest_wire([_wire_row(), _wire_row(sample_idx=1)])
    assert result.inserted == 1
    assert result.deduped == 1
    rows = db.query_rows("run-1")
    assert len(rows) == 2
    assert rows[0]["sampled_tokens"] == [3, 4]
    db.close()


def test_null_keys_never_dedupe(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    row = _wire_row(sampling_session_id=None)
    assert db.ingest_wire([row, row]).inserted == 2
    db.close()


def test_ingest_annotations_idempotent(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    ann = {"event_id": "e1", "run_id": "run-1", "kind": "note", "payload": {"x": 1}}
    assert db.ingest_annotations([ann]).inserted == 1
    result = db.ingest_annotations([ann, {"event_id": "e2", "run_id": "run-1"}])
    assert result.deduped == 1
    assert result.inserted == 1
    db.close()


def test_runs_exist_without_registration(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    assert db.list_runs() == []
    db.ingest_wire([_wire_row(), _wire_row(run_attempt=2, sample_idx=1)])
    db.ingest_annotations([{"event_id": "e1", "run_id": "run-2", "kind": "note"}])
    runs = {r.run_id: r for r in db.list_runs()}
    assert runs["run-1"].latest_attempt == 2
    assert runs["run-1"].num_wire_rows == 2
    assert runs["run-2"].num_annotations == 1
    db.close()


def test_query_filters_and_cursor(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    db.ingest_wire(
        [
            _wire_row(iteration=1, seq_id=1),
            _wire_row(iteration=2, seq_id=2),
            _wire_row(iteration=2, split="test", seq_id=3),
            _wire_row(iteration=2, purpose="eval", seq_id=4),
        ]
    )
    assert len(db.query_rows("run-1", filters={"iteration": 2})) == 3
    assert len(db.query_rows("run-1", filters={"iteration": 2, "split": "train"})) == 2
    assert len(db.query_rows("run-1", filters={"purpose": "eval"})) == 1
    page1 = db.query_rows("run-1", limit=2)
    page2 = db.query_rows("run-1", cursor=page1[-1]["cursor"], limit=10)
    assert [r["seq_id"] for r in page1 + page2] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="Unsupported filter"):
        db.query_rows("run-1", filters={"run_id": "x"})
    db.close()


def test_stream_events_interleave_in_insert_order(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    db.ingest_wire([_wire_row(seq_id=1)])
    db.ingest_annotations([{"event_id": "e1", "run_id": "run-1", "kind": "note"}])
    db.ingest_wire([_wire_row(seq_id=2)])
    events = db.stream_events("run-1")
    assert [e.event_type for e in events] == ["wire_row", "annotation", "wire_row"]
    assert [e.cursor for e in events] == sorted(e.cursor for e in events)
    # Exact resume
    resumed = db.stream_events("run-1", cursor=events[0].cursor)
    assert [e.event_type for e in resumed] == ["annotation", "wire_row"]
    db.close()


# ── HTTP API ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    db = CaptureDB(tmp_path / "db.sqlite")
    app = make_app(db)
    test_client = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()
    db.close()


@pytest.mark.asyncio
async def test_http_ingest_query_roundtrip(client: TestClient) -> None:
    resp = await client.post("/ingest/wire", json={"rows": [_wire_row(), _wire_row()]})
    body = await resp.json()
    assert body == {"inserted": 1, "deduped": 1}

    resp = await client.post(
        "/ingest/annotations",
        json={"annotations": [{"event_id": "e1", "run_id": "run-1", "kind": "note"}]},
    )
    assert (await resp.json())["inserted"] == 1

    resp = await client.get("/runs")
    runs = (await resp.json())["runs"]
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["num_wire_rows"] == 1
    assert runs[0]["num_annotations"] == 1

    resp = await client.get("/runs/run-1/rows", params={"iteration": "1", "split": "train"})
    body = await resp.json()
    assert len(body["rows"]) == 1
    assert body["next_cursor"] == body["rows"][0]["cursor"]

    resp = await client.get("/healthz")
    assert (await resp.json())["status"] == "ok"


async def _read_sse_events(resp, count: int) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """Parse `count` SSE events (id/event/data blocks) from a streaming response."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    while len(events) < count:
        line = (await asyncio.wait_for(resp.content.readline(), timeout=10.0)).decode()
        line = line.rstrip("\n")
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(": ")
        if key == "data":
            current["data"] = json.loads(value)
        else:
            current[key] = value
    return events


@pytest.mark.asyncio
async def test_sse_stream_order_and_resume(client: TestClient) -> None:
    await client.post("/ingest/wire", json={"rows": [_wire_row(seq_id=1)]})
    await client.post(
        "/ingest/annotations",
        json={"annotations": [{"event_id": "e1", "run_id": "run-1", "kind": "note"}]},
    )

    resp = await client.get("/stream", params={"run_id": "run-1"})
    events = await _read_sse_events(resp, 2)
    assert [e["event"] for e in events] == ["wire_row", "annotation"]

    # New rows arrive while streaming.
    await client.post("/ingest/wire", json={"rows": [_wire_row(seq_id=2)]})
    more = await _read_sse_events(resp, 1)
    assert more[0]["event"] == "wire_row"
    assert more[0]["data"]["seq_id"] == 2
    resp.close()

    # Exact resume from the second event's id.
    resume_cursor = events[1]["id"]
    resp2 = await client.get("/stream", params={"run_id": "run-1", "cursor": resume_cursor})
    resumed = await _read_sse_events(resp2, 1)
    assert resumed[0]["data"]["seq_id"] == 2
    assert int(resumed[0]["id"]) > int(resume_cursor)
    resp2.close()


@pytest.mark.asyncio
async def test_idle_monitor_fires_and_respects_activity(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    fired = asyncio.Event()
    app = make_app(db, idle_shutdown_sec=0.4, shutdown_cb=fired.set)
    test_client = TestClient(TestServer(app))
    await test_client.start_server()
    try:
        # Activity (non-healthz) keeps it alive.
        for _ in range(4):
            await test_client.get("/runs")
            await asyncio.sleep(0.15)
        assert not fired.is_set()
        # healthz does not count as activity; idleness triggers shutdown_cb.
        await asyncio.wait_for(fired.wait(), timeout=5.0)
    finally:
        await test_client.close()
        db.close()


@pytest.mark.asyncio
async def test_large_wire_batch_over_one_mib_accepted(client: TestClient) -> None:
    """The app must accept batches beyond aiohttp's 1 MiB default cap."""
    big_tokens = list(range(60_000))
    rows = [
        _wire_row(seq_id=i, sampled_tokens=big_tokens, logprobs=None, prompt_tokens=None)
        for i in range(5)
    ]
    body = json.dumps({"rows": rows})
    assert len(body.encode()) > 1_500_000  # comfortably over the old cap
    resp = await client.post(
        "/ingest/wire", data=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status == 200
    assert (await resp.json())["inserted"] == 5


def test_ingest_wire_batch_atomic_on_failure(tmp_path: Path) -> None:
    """A malformed later row rolls back the WHOLE batch (rows and the cursor
    sequence), so a failed ingest is invisible and the client can retry."""
    db = CaptureDB(tmp_path / "db.sqlite")
    bad = _wire_row(seq_id=2, metadata={1, 2})  # a set is not JSON-serializable
    with pytest.raises(TypeError):
        db.ingest_wire([_wire_row(seq_id=1), bad])
    assert db.query_rows("run-1") == []
    result = db.ingest_wire([_wire_row(seq_id=1), _wire_row(seq_id=2)])
    assert result.inserted == 2
    # The cursor sequence rolled back too: the retry starts at cursor 1.
    assert [e.cursor for e in db.stream_events("run-1")] == [1, 2]
    db.close()


def test_ingest_annotations_batch_atomic_on_failure(tmp_path: Path) -> None:
    db = CaptureDB(tmp_path / "db.sqlite")
    good = {"event_id": "e1", "run_id": "run-1", "kind": "note"}
    with pytest.raises(ValueError, match="event_id"):
        db.ingest_annotations([good, {"run_id": "run-1"}])  # missing event_id
    assert db.list_runs() == []
    assert db.ingest_annotations([good]).inserted == 1
    db.close()


def test_ingest_annotations_rejects_null_keys(tmp_path: Path) -> None:
    """Explicit null event_id must raise (INSERT OR IGNORE would otherwise
    swallow the NOT NULL violation as a silent "dedupe"); a null run_id is
    coerced to "unattributed" like everywhere else."""
    db = CaptureDB(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match="event_id"):
        db.ingest_annotations([{"event_id": None, "run_id": "run-1"}])
    result = db.ingest_annotations([{"event_id": "e1", "run_id": None}])
    assert result.inserted == 1
    assert db.list_runs()[0].run_id == "unattributed"
    db.close()


@pytest.mark.asyncio
async def test_http_ingest_malformed_batch_400(client: TestClient) -> None:
    resp = await client.post(
        "/ingest/annotations", json={"annotations": [{"event_id": None, "run_id": "r"}]}
    )
    assert resp.status == 400
    assert "event_id" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_http_ingest_wrong_shape_400(client: TestClient) -> None:
    """Valid JSON with the wrong shape must be a clean 400, never a 500."""
    # Batch given as a mapping instead of a list.
    resp = await client.post("/ingest/annotations", json={"annotations": {"event_id": "e1"}})
    assert resp.status == 400
    assert "list of JSON objects" in (await resp.json())["error"]

    # Null entry inside the list.
    resp = await client.post("/ingest/annotations", json={"annotations": [None]})
    assert resp.status == 400

    # String entry inside the list (wire path).
    resp = await client.post("/ingest/wire", json={"rows": ["not-an-object"]})
    assert resp.status == 400

    # Whole body not an object.
    resp = await client.post("/ingest/wire", json=[1, 2, 3])
    assert resp.status == 400


def test_ingest_rejects_collections_in_scalar_columns(tmp_path: Path) -> None:
    """Collections in scalar-typed columns (per the schema spec) must raise
    a clean ValueError, not a sqlite3 binding error."""
    db = CaptureDB(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match="run_id"):
        db.ingest_wire([_wire_row(run_id=["not", "scalar"])])
    with pytest.raises(ValueError, match="seq_id"):
        db.ingest_wire([_wire_row(seq_id={"nested": 1})])
    with pytest.raises(ValueError, match="created_at"):
        db.ingest_annotations([{"event_id": "e1", "run_id": "r", "created_at": {}}])
    assert db.list_runs() == []
    db.close()


@pytest.mark.asyncio
async def test_http_ingest_collection_in_scalar_column_400(client: TestClient) -> None:
    resp = await client.post("/ingest/wire", json={"rows": [_wire_row(run_id=[])]})
    assert resp.status == 400
    assert "run_id" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_rows_limit_validation(client: TestClient) -> None:
    await client.post("/ingest/wire", json={"rows": [_wire_row()]})
    for bad in ("-1", "0", "abc"):
        resp = await client.get("/runs/run-1/rows", params={"limit": bad})
        assert resp.status == 400, bad
    # Oversized limits are clamped, not rejected.
    resp = await client.get("/runs/run-1/rows", params={"limit": "999999"})
    assert resp.status == 200
    assert len((await resp.json())["rows"]) == 1
    # Bad filter/cursor values are also client errors, not 500s.
    resp = await client.get("/runs/run-1/rows", params={"iteration": "abc"})
    assert resp.status == 400
    resp = await client.get("/stream", params={"cursor": "abc"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_http_ingest_invalid_json_400(client: TestClient) -> None:
    resp = await client.post(
        "/ingest/wire", data=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status == 400
    assert "valid JSON" in (await resp.json())["error"]


def test_ingest_rejects_out_of_range_integers(tmp_path: Path) -> None:
    """Unbounded Python ints pass the type check but overflow SQLite's
    signed 64-bit binding; they must be a clean ValueError instead."""
    db = CaptureDB(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match="64-bit"):
        db.ingest_wire([_wire_row(seq_id=2**63)])
    with pytest.raises(ValueError, match="64-bit"):
        db.ingest_annotations([{"event_id": "e1", "run_id": "r", "created_at": -(2**63) - 1}])
    assert db.list_runs() == []
    db.close()


@pytest.mark.asyncio
async def test_query_params_bounded_to_int64(client: TestClient) -> None:
    await client.post("/ingest/wire", json={"rows": [_wire_row()]})
    big = str(2**63)
    resp = await client.get("/runs/run-1/rows", params={"cursor": big})
    assert resp.status == 400
    assert "64-bit" in (await resp.json())["error"]
    resp = await client.get("/runs/run-1/rows", params={"iteration": big})
    assert resp.status == 400
    resp = await client.get("/stream", params={"cursor": big})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_http_ingest_missing_batch_key_400(client: TestClient) -> None:
    """A missing/misspelled batch key must not silently succeed as empty."""
    resp = await client.post("/ingest/wire", json={"row": [_wire_row()]})
    assert resp.status == 400
    assert "'rows'" in (await resp.json())["error"]
    resp = await client.post("/ingest/annotations", json={"rows": []})
    assert resp.status == 400
    assert "'annotations'" in (await resp.json())["error"]
    # An explicitly empty batch is still fine.
    resp = await client.post("/ingest/wire", json={"rows": []})
    assert resp.status == 200


def test_ingest_enforces_declared_column_types(tmp_path: Path) -> None:
    """Scalars must match the schema's declared kinds, not just be scalars;
    type affinity would otherwise store TEXT in INTEGER columns that the
    integer-typed filters can never serve."""
    db = CaptureDB(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match=r"iteration.*integer"):
        db.ingest_wire([_wire_row(iteration="twelve")])
    with pytest.raises(ValueError, match=r"seq_id.*integer"):
        db.ingest_wire([_wire_row(seq_id=1.5)])
    with pytest.raises(ValueError, match=r"group_idx.*integer"):
        db.ingest_wire([_wire_row(group_idx=True)])  # bools are not ints here
    with pytest.raises(ValueError, match=r"run_id.*string"):
        db.ingest_wire([_wire_row(run_id=7)])
    with pytest.raises(ValueError, match="32-bit"):
        db.ingest_wire([_wire_row(iteration=2**31)])
    assert db.list_runs() == []
    db.close()


@pytest.mark.asyncio
async def test_touch_counts_as_activity_healthz_does_not(client: TestClient) -> None:
    from tinker_cookbook.capture.store.app import _ACTIVITY_KEY

    app = client.app
    assert app is not None
    app[_ACTIVITY_KEY][0] = 0.0
    resp = await client.get("/healthz")
    assert resp.status == 200
    assert app[_ACTIVITY_KEY][0] == 0.0  # healthz never counts
    resp = await client.get("/touch")
    assert resp.status == 200
    assert (await resp.json()) == {"status": "ok"}
    assert app[_ACTIVITY_KEY][0] > 0.0  # touch resets the idle timer


def test_ingest_validates_list_columns_against_schema(tmp_path: Path) -> None:
    """List columns must be lists of the declared element kind; arbitrary
    JSON would be incompatible with the rendered Arrow/ClickHouse schemas."""
    db = CaptureDB(tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match=r"prompt_tokens.*list"):
        db.ingest_wire([_wire_row(prompt_tokens={"a": 1})])
    with pytest.raises(ValueError, match=r"prompt_tokens.*list"):
        db.ingest_wire([_wire_row(prompt_tokens="not-a-list")])
    with pytest.raises(ValueError, match=r"sampled_tokens\[0\].*integer"):
        db.ingest_wire([_wire_row(sampled_tokens=["x", "y"])])
    with pytest.raises(ValueError, match=r"sampled_tokens\[0\].*integer"):
        db.ingest_wire([_wire_row(sampled_tokens=[True, 2])])  # bools are not token ids
    with pytest.raises(ValueError, match=r"sampled_tokens\[1\].*32-bit"):
        db.ingest_wire([_wire_row(sampled_tokens=[1, 1099511627776])])
    with pytest.raises(ValueError, match=r"logprobs\[0\].*number"):
        db.ingest_wire([_wire_row(logprobs=["bad"])])
    with pytest.raises(ValueError, match=r"logprobs\[1\].*null"):
        db.ingest_wire([_wire_row(logprobs=[-0.5, None])])
    # metadata is declared `json` (free-form by design): any shape is fine.
    assert db.ingest_wire([_wire_row(metadata="free-form string")]).inserted == 1
    db.close()


@pytest.mark.asyncio
async def test_shutting_down_daemon_rejects_all_requests(client: TestClient) -> None:
    """Once the idle monitor commits to shutdown, /touch (and everything
    else, healthz included) must 503 so ensure_daemon cannot claim a dying
    daemon."""
    from tinker_cookbook.capture.store.app import _SHUTTING_DOWN_KEY

    app = client.app
    assert app is not None
    app[_SHUTTING_DOWN_KEY][0] = True
    assert (await client.get("/touch")).status == 503
    assert (await client.get("/healthz")).status == 503
    assert (await client.post("/ingest/wire", json={"rows": []})).status == 503
    app[_SHUTTING_DOWN_KEY][0] = False
    assert (await client.get("/touch")).status == 200


@pytest.mark.asyncio
async def test_annotations_read_path(client: TestClient) -> None:
    """Train-op records land in annotations; they must be queryable (kind
    filter + cursor pagination), not only reachable via /stream."""
    annotations = [
        {"event_id": f"e{i}", "run_id": "run-1", "kind": kind, "payload": {"i": i}}
        for i, kind in enumerate(["train_op", "train_op", "note", "train_op"])
    ]
    resp = await client.post("/ingest/annotations", json={"annotations": annotations})
    assert (await resp.json())["inserted"] == 4

    resp = await client.get("/runs/run-1/annotations", params={"kind": "train_op"})
    body = await resp.json()
    assert [a["event_id"] for a in body["annotations"]] == ["e0", "e1", "e3"]
    assert body["next_cursor"] == body["annotations"][-1]["cursor"]
    assert body["annotations"][0]["payload"] == {"i": 0}

    # Cursor pagination.
    resp = await client.get("/runs/run-1/annotations", params={"kind": "train_op", "limit": "2"})
    page1 = (await resp.json())["annotations"]
    assert [a["event_id"] for a in page1] == ["e0", "e1"]
    resp = await client.get(
        "/runs/run-1/annotations",
        params={"kind": "train_op", "cursor": str(page1[-1]["cursor"])},
    )
    assert [a["event_id"] for a in (await resp.json())["annotations"]] == ["e3"]

    # No filter returns everything; bad limit is a 400 like /rows.
    resp = await client.get("/runs/run-1/annotations")
    assert len((await resp.json())["annotations"]) == 4
    resp = await client.get("/runs/run-1/annotations", params={"limit": "0"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_stream_cursor_header_400_is_carrier_agnostic(client: TestClient) -> None:
    """A malformed Last-Event-ID header must not be blamed on a 'query
    parameter'; the message names both carriers."""
    resp = await client.get("/stream", headers={"Last-Event-ID": "abc"})
    assert resp.status == 400
    error = (await resp.json())["error"]
    assert "Last-Event-ID" in error
    assert "'cursor' must be an integer" in error
