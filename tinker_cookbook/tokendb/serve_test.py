"""Tests for the token DB viewer server (HTTP API + websocket push)."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer

from tinker_cookbook.tokendb.schema import TokenRow, compute_ob_delta
from tinker_cookbook.tokendb.serve import build_app
from tinker_cookbook.tokendb.writer import TokenDbWriter


def make_row(**overrides) -> TokenRow:
    defaults: dict = {
        "split": "train",
        "iteration": 0,
        "group_idx": 0,
        "traj_idx": 0,
        "step_idx": 0,
        "ob_tokens": [1, 2, 3],
        "ac_tokens": [4, 5],
    }
    defaults.update(overrides)
    return TokenRow(**defaults)


def make_delta_trajectory(iteration: int = 1) -> list[TokenRow]:
    """A 2-step trajectory whose step-1 ob is a prefix extension (delta-encoded)."""
    obs_full = [[10, 11, 12], [10, 11, 12, 20, 21, 30]]
    acs = [[20, 21], [40]]
    rows = []
    prev_sequence: list[int] = []
    for step_idx, (ob, ac) in enumerate(zip(obs_full, acs)):
        stored_ob, is_delta = compute_ob_delta(prev_sequence, ob)
        rows.append(
            make_row(
                iteration=iteration,
                step_idx=step_idx,
                ob_tokens=stored_ob,
                ob_is_delta=is_delta,
                ac_tokens=ac,
                ac_logprobs=[-0.1] * len(ac),
                episode_done=step_idx == 1,
            )
        )
        prev_sequence = ob + ac
    return rows


@pytest.fixture
def populated_store(tmp_path: Path) -> Path:
    """A store with two run attempts, a delta-ob trajectory, and a filtered row."""
    log_path = tmp_path / "run"
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer:
        writer.append_rows(
            [
                make_row(
                    iteration=0,
                    total_reward=1.0,
                    final_reward=1.0,
                    stop_reason="stop_token",
                    tags=["math"],
                    ac_text="the answer is 4",
                ),
                make_row(iteration=0, traj_idx=1, total_reward=-1.0, stop_reason="length"),
                make_row(
                    iteration=0,
                    group_idx=1,
                    source="filtered",
                    filtered_reason="constant_reward",
                ),
            ]
        )
        writer.append_rows(make_delta_trajectory(iteration=1))
    # Second attempt (resume): re-runs iteration 1, superseding the rows above.
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer:
        writer.append_rows([make_row(iteration=1, total_reward=2.0)])
    return log_path


def run_with_client(log_path: Path, fn):
    """Run coroutine ``fn(client)`` against a test client for the given store.

    ``static_dir=None`` exercises the "UI not built" fallback;
    ``load_tokenizer=False`` keeps tests offline.
    """

    async def main():
        app = build_app(log_path, static_dir=None, load_tokenizer=False)
        async with TestClient(TestServer(app)) as client:
            return await fn(client)

    return asyncio.run(main())


def test_run_endpoint(populated_store: Path):
    async def fn(client):
        resp = await client.get("/api/run")
        assert resp.status == 200
        payload = await resp.json()
        assert payload["run_attempt"] == 2
        assert payload["context"]["model_name"] == "test-model"

    run_with_client(populated_store, fn)


def test_run_endpoint_missing(tmp_path: Path):
    async def fn(client):
        resp = await client.get("/api/run")
        assert resp.status == 404

    run_with_client(tmp_path, fn)


def test_rollouts_trajectory_grain_and_superseded(populated_store: Path):
    async def fn(client):
        resp = await client.get("/api/rollouts")
        assert resp.status == 200
        payload = await resp.json()
        assert payload["grain"] == "trajectories"
        rows = payload["rows"]
        # 5 trajectories: 4 from attempt 1 (one filtered) + 1 from attempt 2.
        assert len(rows) == 5
        by_key = {
            (r["iteration"], r["group_idx"], r["traj_idx"], r["run_attempt"]): r for r in rows
        }
        # The attempt-1 iteration-1 trajectory is superseded by attempt 2.
        assert by_key[(1, 0, 0, 1)]["superseded"] is True
        assert by_key[(1, 0, 0, 2)]["superseded"] is False
        assert by_key[(0, 0, 0, 1)]["superseded"] is False
        assert by_key[(0, 1, 0, 1)]["filtered_reason"] == "constant_reward"
        assert by_key[(1, 0, 0, 1)]["n_steps"] == 2
        assert by_key[(1, 0, 0, 1)]["n_ac_tokens"] == 3  # a JSON int, not a stringified HUGEINT
        assert by_key[(0, 0, 0, 1)]["ac_preview"] == "the answer is 4"

        # latest_only drops the superseded trajectory.
        resp = await client.get("/api/rollouts", params={"latest_only": "true"})
        rows = (await resp.json())["rows"]
        assert (1, 0, 0, 1) not in {
            (r["iteration"], r["group_idx"], r["traj_idx"], r["run_attempt"]) for r in rows
        }

        # Structured filters pass through.
        resp = await client.get(
            "/api/rollouts", params={"stop_reason": "length", "grain": "rollouts"}
        )
        rows = (await resp.json())["rows"]
        assert len(rows) == 1 and rows[0]["traj_idx"] == 1

        resp = await client.get("/api/rollouts", params={"iteration": "notanint"})
        assert resp.status == 400

    run_with_client(populated_store, fn)


def test_rollout_detail_delta_reconstruction(populated_store: Path):
    async def fn(client):
        resp = await client.get("/api/rollout/train/1/0/0", params={"run_attempt": "1"})
        assert resp.status == 200
        payload = await resp.json()
        steps = payload["steps"]
        assert [s["step_idx"] for s in steps] == [0, 1]
        assert steps[1]["ob_is_delta"] is True
        assert steps[1]["ob_tokens"] == [30]  # stored delta suffix
        assert steps[0]["ob_full_tokens"] == [10, 11, 12]
        assert steps[1]["ob_full_tokens"] == [10, 11, 12, 20, 21, 30]  # reconstructed
        assert payload["group_traj_idxs"] == [0]

        # Default run_attempt resolves to the latest attempt.
        resp = await client.get("/api/rollout/train/1/0/0")
        steps = (await resp.json())["steps"]
        assert all(s["run_attempt"] == 2 for s in steps)

        resp = await client.get("/api/rollout/train/99/0/0")
        assert resp.status == 404

    run_with_client(populated_store, fn)


def test_search_endpoint(populated_store: Path):
    async def fn(client):
        resp = await client.post("/api/search", json={"regex": "answer"})
        assert resp.status == 200
        payload = await resp.json()
        assert len(payload["rows"]) == 1
        assert payload["hit_counts"] == {"0": 1}

        # Token-subsequence search matches contiguous runs inside ac_tokens.
        resp = await client.post("/api/search", json={"token_subsequence": [20, 21]})
        assert {r["iteration"] for r in (await resp.json())["rows"]} == {1}

        resp = await client.post("/api/search", json={})
        assert resp.status == 400

    run_with_client(populated_store, fn)


def test_sql_endpoint_guard(populated_store: Path):
    async def fn(client):
        resp = await client.post(
            "/api/sql",
            json={
                "query": "SELECT count(*) AS n FROM rollouts WHERE split = ?",
                "params": ["train"],
            },
        )
        assert resp.status == 200
        assert (await resp.json())["rows"][0]["n"] == 6

        # The reader's SELECT-only guard surfaces as a 400.
        resp = await client.post("/api/sql", json={"query": "DROP TABLE segment_rows"})
        assert resp.status == 400
        assert "SELECT" in (await resp.json())["error"]

        resp = await client.post("/api/sql", json={"query": "SELECT 1; SELECT 2"})
        assert resp.status == 400

    run_with_client(populated_store, fn)


def test_labels_roundtrip(populated_store: Path):
    async def fn(client):
        resp = await client.post(
            "/api/labels",
            json={
                "key": {"split": "train", "iteration": 0, "group_idx": 0, "traj_idx": 0},
                "label_key": "quality",
                "label_value": "good",
                "author": "tester",
                "note": "nice rollout",
            },
        )
        assert resp.status == 200

        resp = await client.get("/api/labels", params={"label_key": "quality"})
        labels = (await resp.json())["labels"]
        assert len(labels) == 1
        assert labels[0]["label_value"] == "good"
        assert labels[0]["author"] == "tester"
        assert labels[0]["iteration"] == 0

        resp = await client.post("/api/labels", json={"label_key": "quality"})
        assert resp.status == 400

    run_with_client(populated_store, fn)


def test_decode_degrades_without_tokenizer(populated_store: Path):
    async def fn(client):
        resp = await client.post("/api/tokens/decode", json={"tokens": [1, 2]})
        assert resp.status == 503
        assert "tokenizer" in (await resp.json())["error"]

    run_with_client(populated_store, fn)


def test_static_fallback_page(populated_store: Path):
    async def fn(client):
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "ui/README" in text  # points at the frontend build instructions

    run_with_client(populated_store, fn)


def test_websocket_pushes_new_rows(populated_store: Path):
    async def fn(client):
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(
                json.dumps(
                    {"type": "subscribe", "filters": {"split": "train"}, "poll_interval_s": 0.05}
                )
            )
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "subscribed"

            # Give the subscriber a beat to take its baseline, then write.
            await asyncio.sleep(0.2)
            with TokenDbWriter(populated_store) as writer:
                writer.append_rows([make_row(iteration=5, ac_text="fresh row")])

            row = None
            for _ in range(10):
                msg = json.loads((await ws.receive(timeout=5.0)).data)
                if msg["type"] == "row":
                    row = msg["row"]
                    break
            assert row is not None
            assert row["iteration"] == 5
            assert row["ac_text"] == "fresh row"

    run_with_client(populated_store, fn)


def test_websocket_bad_subscription(populated_store: Path):
    async def fn(client):
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "subscribe", "filters": {"nope": 1}}))
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "error"

    run_with_client(populated_store, fn)
