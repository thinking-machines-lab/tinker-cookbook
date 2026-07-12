"""Tests for the token DB viewer server (HTTP API + websocket push)."""

import asyncio
import json
import os
import time
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


@pytest.fixture
def maps_store(tmp_path: Path) -> Path:
    """A store exercising the typed map columns (attrs/metrics/token_metrics)."""
    log_path = tmp_path / "maps-run"
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer:
        writer.append_rows(
            [
                make_row(
                    iteration=0,
                    attrs={"dataset": "gsm8k"},
                    metrics={"group/score": 0.9, "acc": float("nan")},
                    token_metrics={"kl": [0.5, float("nan")]},
                    ac_text="gsm high",
                ),
                make_row(
                    iteration=1,
                    attrs={"dataset": "math"},
                    metrics={"group/score": 0.2},
                    ac_text="math low",
                ),
            ]
        )
    return log_path


def test_structured_filter_query_params(maps_store: Path):
    async def fn(client):
        # attr.<key> / metric_min.<key> dotted params (keys may contain "/").
        resp = await client.get(
            "/api/rollouts", params={"grain": "rollouts", "attr.dataset": "gsm8k"}
        )
        assert resp.status == 200
        rows = (await resp.json())["rows"]
        assert [r["ac_text"] for r in rows] == ["gsm high"]

        resp = await client.get(
            "/api/rollouts", params={"grain": "rollouts", "metric_min.group/score": "0.5"}
        )
        rows = (await resp.json())["rows"]
        assert [r["ac_text"] for r in rows] == ["gsm high"]

        resp = await client.get(
            "/api/rollouts", params={"grain": "rollouts", "metric_max.group/score": "0.5"}
        )
        rows = (await resp.json())["rows"]
        assert [r["ac_text"] for r in rows] == ["math low"]

        # Structured filters apply at the trajectories grain too.
        resp = await client.get("/api/rollouts", params={"attr.dataset": "math"})
        rows = (await resp.json())["rows"]
        assert len(rows) == 1 and rows[0]["iteration"] == 1

        resp = await client.get("/api/rollouts", params={"metric_min.group/score": "not-a-float"})
        assert resp.status == 400

    run_with_client(maps_store, fn)


def test_structured_filters_in_search_body(maps_store: Path):
    async def fn(client):
        resp = await client.post(
            "/api/search",
            json={"regex": ".", "filters": {"attr_eq": {"dataset": "gsm8k"}}},
        )
        assert resp.status == 200
        rows = (await resp.json())["rows"]
        assert [r["ac_text"] for r in rows] == ["gsm high"]

        resp = await client.post(
            "/api/search",
            json={"regex": ".", "filters": {"metric_min": {"group/score": 0.5}}},
        )
        rows = (await resp.json())["rows"]
        assert [r["ac_text"] for r in rows] == ["gsm high"]

        resp = await client.post(
            "/api/search", json={"regex": ".", "filters": {"attr_eq": "not-a-dict"}}
        )
        assert resp.status == 400

    run_with_client(maps_store, fn)


def test_map_payload_shape_and_nan_sanitization(maps_store: Path):
    async def fn(client):
        resp = await client.get("/api/rollouts", params={"grain": "rollouts", "iteration": "0"})
        assert resp.status == 200
        # If NaN leaked into the JSON body, .json() (strict JSON.parse
        # equivalent) would fail here.
        body = await resp.text()
        (row,) = json.loads(body)["rows"]
        # Maps arrive as JSON objects, not strings.
        assert row["attrs"] == {"dataset": "gsm8k"}
        assert row["metrics"]["group/score"] == pytest.approx(0.9)
        assert row["metrics"]["acc"] is None  # NaN -> null
        assert row["token_metrics"]["kl"] == [pytest.approx(0.5), None]
        assert row["tool_calls"] is None
        assert "NaN" not in body

    run_with_client(maps_store, fn)


def test_runs_view_queryable_via_sql(maps_store: Path):
    async def fn(client):
        resp = await client.post(
            "/api/sql",
            json={"query": "SELECT run_attempt, model_name, config_json FROM runs"},
        )
        assert resp.status == 200
        rows = (await resp.json())["rows"]
        assert len(rows) == 1
        assert rows[0]["run_attempt"] == 1
        assert rows[0]["model_name"] == "test-model"

    run_with_client(maps_store, fn)


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


# --- Registry mode (no log_path: run listing, dashboard, per-run proxying) ---


@pytest.fixture
def registry_stores(tmp_path: Path) -> dict:
    """Two registered runs: one live (fresh manifests), one stale (aged mtimes)."""
    live_path = tmp_path / "live-run"
    with TokenDbWriter(
        live_path, context={"model_name": "live-model", "recipe_name": "live-recipe"}
    ) as writer:
        writer.append_rows(
            [
                make_row(iteration=0, total_reward=1.0, ac_text="the answer is 4"),
                make_row(iteration=1, total_reward=3.0),
                make_row(
                    iteration=1,
                    group_idx=1,
                    source="filtered",
                    filtered_reason="constant_reward",
                ),
            ]
        )
        live_id = writer.run_id
    stale_path = tmp_path / "stale-run"
    with TokenDbWriter(stale_path, context={"model_name": "stale-model"}) as writer:
        writer.append_rows([make_row(iteration=2, total_reward=-1.0)])
        stale_id = writer.run_id
    old = time.time() - 3600
    for manifest in (stale_path / "tokens").glob("manifest-*.jsonl"):
        os.utime(manifest, (old, old))
    return {"live_id": live_id, "stale_id": stale_id, "live_path": live_path}


def run_registry_client(fn, **build_kwargs):
    """Run coroutine ``fn(client)`` against a registry-mode (no log_path) app."""

    async def main():
        app = build_app(None, static_dir=None, load_tokenizer=False, **build_kwargs)
        async with TestClient(TestServer(app)) as client:
            return await fn(client)

    return asyncio.run(main())


def test_registry_runs_endpoint(registry_stores: dict):
    async def fn(client):
        resp = await client.get("/api/runs")
        assert resp.status == 200
        runs = {r["run_id"]: r for r in (await resp.json())["runs"]}
        assert set(runs) == {registry_stores["live_id"], registry_stores["stale_id"]}
        live = runs[registry_stores["live_id"]]
        stale = runs[registry_stores["stale_id"]]
        assert live["model_name"] == "live-model"
        assert live["recipe_name"] == "live-recipe"
        assert live["run_attempt"] == 1
        assert live["status"]["live"] is True
        assert live["status"]["latest_iteration"] == 1
        assert stale["status"]["live"] is False
        assert stale["status"]["latest_iteration"] == 2
        assert stale["status"]["n_segments"] >= 1

    run_registry_client(fn)


def test_registry_dashboard(registry_stores: dict):
    async def fn(client):
        resp = await client.get("/api/dashboard")
        assert resp.status == 200
        rows = {r["run_id"]: r for r in (await resp.json())["runs"]}
        live = rows[registry_stores["live_id"]]
        assert live["live"] is True
        assert live["model_name"] == "live-model"
        assert live["recipe_name"] == "live-recipe"
        assert live["n_rows"] == 3
        assert live["n_filtered_rows"] == 1
        assert live["latest_iteration"] == 1
        assert live["reward_series"] == [
            {"iteration": 0, "mean_total_reward": 1.0},
            {"iteration": 1, "mean_total_reward": 3.0},
        ]
        assert live["mean_recent_reward"] == pytest.approx(2.0)
        stale = rows[registry_stores["stale_id"]]
        assert stale["live"] is False
        assert stale["n_rows"] == 1
        assert stale["mean_recent_reward"] == pytest.approx(-1.0)
        assert stale["last_activity_ts"] is not None

    run_registry_client(fn)


def test_registry_dashboard_ttl_cache(registry_stores: dict):
    async def fn_cached(client):
        first = {
            r["run_id"]: r for r in (await (await client.get("/api/dashboard")).json())["runs"]
        }
        # New rows written within the TTL are not reflected (consistent polls).
        with TokenDbWriter(registry_stores["live_path"]) as writer:
            writer.append_rows([make_row(iteration=9, total_reward=5.0)])
        second = {
            r["run_id"]: r for r in (await (await client.get("/api/dashboard")).json())["runs"]
        }
        assert second == first

    run_registry_client(fn_cached, dashboard_ttl_s=300.0)

    async def fn_fresh(client):
        rows = {r["run_id"]: r for r in (await (await client.get("/api/dashboard")).json())["runs"]}
        live = rows[registry_stores["live_id"]]
        # TTL 0 recomputes: sees the extra row (and the resume's new attempt).
        assert live["n_rows"] == 4
        assert live["latest_iteration"] == 9

    run_registry_client(fn_fresh, dashboard_ttl_s=0.0)


def test_registry_per_run_endpoints(registry_stores: dict):
    live_id = registry_stores["live_id"]

    async def fn(client):
        resp = await client.get(f"/api/runs/{live_id}/run")
        assert resp.status == 200
        payload = await resp.json()
        assert payload["run_id"] == live_id
        assert payload["context"]["model_name"] == "live-model"

        resp = await client.get(f"/api/runs/{live_id}/rollouts")
        assert resp.status == 200
        assert len((await resp.json())["rows"]) == 3

        resp = await client.get(f"/api/runs/{live_id}/rollout/train/1/0/0")
        assert resp.status == 200
        steps = (await resp.json())["steps"]
        assert len(steps) == 1 and steps[0]["total_reward"] == 3.0

        resp = await client.post(f"/api/runs/{live_id}/search", json={"regex": "answer"})
        assert resp.status == 200
        assert len((await resp.json())["rows"]) == 1

        resp = await client.post(
            f"/api/runs/{live_id}/sql", json={"query": "SELECT count(*) AS n FROM rollouts"}
        )
        assert (await resp.json())["rows"][0]["n"] == 3

        resp = await client.post(
            f"/api/runs/{live_id}/labels",
            json={
                "key": {"split": "train", "iteration": 0, "group_idx": 0, "traj_idx": 0},
                "label_key": "quality",
                "label_value": "good",
                "author": "tester",
            },
        )
        assert resp.status == 200
        resp = await client.get(f"/api/runs/{live_id}/labels")
        assert len((await resp.json())["labels"]) == 1

        resp = await client.post(f"/api/runs/{live_id}/tokens/decode", json={"tokens": [1]})
        assert resp.status == 503  # tokenizer loads disabled in tests

        # Unknown run IDs 404 with a JSON error body.
        resp = await client.get("/api/runs/doesnotexist/rollouts")
        assert resp.status == 404
        assert "unknown run_id" in (await resp.json())["error"]

    run_registry_client(fn)


def test_registry_websocket_run_subscription(registry_stores: dict):
    live_id = registry_stores["live_id"]
    live_path = registry_stores["live_path"]

    async def fn(client):
        async with client.ws_connect("/ws") as ws:
            # Registry mode: a subscription must name a run.
            await ws.send_str(json.dumps({"type": "subscribe", "filters": {}}))
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "error"
            assert "run_id" in msg["error"]

            await ws.send_str(json.dumps({"type": "subscribe", "run_id": "doesnotexist"}))
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "error"

            await ws.send_str(
                json.dumps({"type": "subscribe", "run_id": live_id, "poll_interval_s": 0.05})
            )
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "subscribed"

            await asyncio.sleep(0.2)
            with TokenDbWriter(live_path) as writer:
                writer.append_rows([make_row(iteration=5, ac_text="fresh row")])
            row = None
            for _ in range(10):
                msg = json.loads((await ws.receive(timeout=5.0)).data)
                if msg["type"] == "row":
                    row = msg["row"]
                    break
            assert row is not None and row["iteration"] == 5

        # The per-run mount needs no run_id in the message.
        async with client.ws_connect(f"/api/runs/{live_id}/ws") as ws:
            await ws.send_str(json.dumps({"type": "subscribe", "filters": {"split": "train"}}))
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "subscribed"

    run_registry_client(fn)


def test_registry_websocket_dashboard(registry_stores: dict):
    async def fn(client):
        async with client.ws_connect("/ws/dashboard?poll_interval_s=60") as ws:
            msg = json.loads((await ws.receive(timeout=10.0)).data)
            assert msg["type"] == "dashboard"
            run_ids = {r["run_id"] for r in msg["runs"]}
            assert run_ids == {registry_stores["live_id"], registry_stores["stale_id"]}

    run_registry_client(fn)


def test_registry_root_sql_cross_run(registry_stores: dict):
    """POST /api/sql at the registry root runs cross-run SQL over all runs."""

    async def fn(client):
        resp = await client.post("/api/sql", json={"query": "SELECT count(*) AS n FROM rollouts"})
        assert resp.status == 200
        assert (await resp.json())["rows"][0]["n"] == 4  # 3 live-run + 1 stale-run rows

        resp = await client.post(
            "/api/sql",
            json={"query": "SELECT run_id, count(*) AS n FROM rollouts GROUP BY run_id"},
        )
        rows = (await resp.json())["rows"]
        assert {r["run_id"]: r["n"] for r in rows} == {
            registry_stores["live_id"]: 3,
            registry_stores["stale_id"]: 1,
        }

        # runs is cross-run too: one row per (run_id, run_attempt).
        resp = await client.post(
            "/api/sql", json={"query": "SELECT run_id, model_name FROM runs ORDER BY model_name"}
        )
        rows = (await resp.json())["rows"]
        assert [r["model_name"] for r in rows] == ["live-model", "stale-model"]

        # The SELECT-only guard surfaces as a 400, same as the per-run endpoint.
        resp = await client.post("/api/sql", json={"query": "DROP VIEW rollouts"})
        assert resp.status == 400
        assert "SELECT" in (await resp.json())["error"]

    run_registry_client(fn)


def test_registry_mode_requires_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", "")
    with pytest.raises(ValueError, match="registry is disabled"):
        build_app(None, static_dir=None, load_tokenizer=False)


# --- Chat agent endpoints (websocket chat, config, visuals) ---

from tinker_cookbook.tokendb.agent_test import (  # noqa: E402
    GatedTransport,
    ScriptedTransport,
    anthropic_script,
)
from tinker_cookbook.tokendb.llm import DEFAULT_MODELS, KNOWN_MODELS  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat tests must control key presence themselves."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)


async def _collect_until_terminal(ws) -> list[dict]:
    """Collect frames through the terminal record (done/error/cancelled)."""
    frames = []
    while True:
        frames.append(json.loads((await ws.receive(timeout=10.0)).data))
        if frames[-1].get("type") in ("done", "error", "cancelled"):
            return frames


async def _chat_frames(ws, text: str, conversation_id: str | None = None) -> list[dict]:
    """Send a user_message and collect frames through the terminal one."""
    message: dict = {"type": "user_message", "text": text}
    if conversation_id is not None:
        message["conversation_id"] = conversation_id
    await ws.send_str(json.dumps(message))
    return await _collect_until_terminal(ws)


def _shapes(frames: list[dict]) -> list[tuple]:
    """(kind, role-or-type) per frame; acks/transient errors have kind None."""
    return [(f.get("kind"), f.get("role") or f.get("type")) for f in frames]


def test_chat_ws_end_to_end(populated_store: Path):
    transport = ScriptedTransport(
        [
            anthropic_script(
                text="Checking.",
                tool_calls=[("t1", "sql", {"query": "SELECT count(*) AS n FROM rollouts"})],
            ),
            anthropic_script(text="There are 6 rows."),
        ]
    )

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/agent/config", json={"api_key": "sk-test"})
            assert (await resp.json())["has_key"] is True
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "how many rows?")
            assert frames[0]["type"] == "conversation"
            conversation_id = frames[0]["conversation_id"]
            # After the ack, the stream is the persisted transcript records.
            assert _shapes(frames) == [
                (None, "conversation"),
                ("message", "user"),
                ("event", "text_delta"),
                ("event", "tool_call"),
                ("message", "assistant"),
                ("event", "tool_result"),
                ("message", "tool"),
                ("event", "text_delta"),
                ("message", "assistant"),
                ("event", "done"),
            ]
            records = frames[1:]
            assert [r["seq"] for r in records] == list(range(len(records)))
            assert frames[2]["text"] == "Checking."
            assert frames[3]["name"] == "sql"
            assert not frames[5]["is_error"]
            assert frames[7]["text"] == "There are 6 rows."

            # The system prompt carried the run's SQL endpoint and schema docs.
            system = transport.requests[0][2]["system"]
            assert "/api/sql" in system
            assert "ob_is_delta" in system

            # Transcript endpoints.
            resp = await client.get("/api/chats")
            conversations = (await resp.json())["conversations"]
            assert [c["conversation_id"] for c in conversations] == [conversation_id]
            assert conversations[0]["title"] == "how many rows?"
            assert conversations[0]["in_flight"] is False
            resp = await client.get(f"/api/chats/{conversation_id}")
            assert (await resp.json())["records"] == records
            resp = await client.get("/api/chats/nope-id")
            assert resp.status == 404

    asyncio.run(main())


async def _poll_records(client, path: str, done: bool = True, timeout: float = 10.0) -> list[dict]:
    """Poll a chat-detail endpoint until the transcript has a terminal event."""
    deadline = time.monotonic() + timeout
    while True:
        resp = await client.get(path)
        if resp.status == 200:
            records = (await resp.json())["records"]
            if any(r.get("type") in ("done", "error", "cancelled") for r in records):
                return records
        if time.monotonic() > deadline:
            raise AssertionError(f"no terminal event in {path} within {timeout}s")
        await asyncio.sleep(0.02)


def test_chat_ws_disconnect_does_not_cancel_turn(populated_store: Path):
    """The turn is server-owned: dropping the socket mid-turn changes nothing."""
    gate = asyncio.Event()
    transport = GatedTransport(
        [
            anthropic_script(
                text="Checking.",
                tool_calls=[("t1", "sql", {"query": "SELECT count(*) AS n FROM rollouts"})],
            ),
            anthropic_script(text="There are 6 rows."),
        ],
        gates={1: gate},  # park the turn before its second model call
    )

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            ws = await client.ws_connect("/api/chat")
            await ws.send_str(json.dumps({"type": "user_message", "text": "how many rows?"}))
            first = json.loads((await ws.receive(timeout=10.0)).data)
            assert first["type"] == "conversation"
            conversation_id = first["conversation_id"]
            # Read up to the tool message, then drop the socket mid-turn.
            frames = []
            while not frames or frames[-1].get("role") != "tool":
                frames.append(json.loads((await ws.receive(timeout=10.0)).data))
            last_seq = frames[-1]["seq"]
            # While parked: the conversation reports in_flight everywhere.
            chats = (await (await client.get("/api/chats")).json())["conversations"]
            assert chats[0]["conversation_id"] == conversation_id
            assert chats[0]["in_flight"] is True
            recent = (await (await client.get("/api/chats/recent")).json())["conversations"]
            assert recent[0]["in_flight"] is True
            # A second user_message on the same conversation is rejected.
            await ws.send_str(
                json.dumps(
                    {"type": "user_message", "text": "again", "conversation_id": conversation_id}
                )
            )
            busy = json.loads((await ws.receive(timeout=10.0)).data)
            assert busy["type"] == "error"
            assert busy["code"] == "turn_in_flight"
            await ws.close()

            gate.set()
            records = await _poll_records(client, f"/api/chats/{conversation_id}")
            # The turn finished after the disconnect: complete transcript.
            assert records[-1]["type"] == "done"
            assert [r["seq"] for r in records] == list(range(len(records)))
            assert any(r.get("content") == "There are 6 rows." for r in records)

            # A fresh socket resumes from the last seen seq with no dupes.
            async with client.ws_connect("/api/chat") as ws2:
                await ws2.send_str(
                    json.dumps(
                        {
                            "type": "subscribe_conversation",
                            "conversation_id": conversation_id,
                            "after_seq": last_seq,
                        }
                    )
                )
                ack = json.loads((await ws2.receive(timeout=10.0)).data)
                assert ack["type"] == "subscribed_conversation"
                assert ack["in_flight"] is False
                tail = await _collect_until_terminal(ws2)
            assert [r["seq"] for r in tail] == list(range(last_seq + 1, len(records)))
            assert tail == records[last_seq + 1 :]

            chats = (await (await client.get("/api/chats")).json())["conversations"]
            assert chats[0]["in_flight"] is False

    asyncio.run(main())


def test_chat_ws_cancel_running_turn(populated_store: Path):
    """A cancel frame stops the manager task; the transcript records it."""
    gate = asyncio.Event()
    transport = GatedTransport([anthropic_script(text="never sent")], gates={0: gate})

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                await ws.send_str(json.dumps({"type": "user_message", "text": "hello"}))
                first = json.loads((await ws.receive(timeout=10.0)).data)
                conversation_id = first["conversation_id"]
                user = json.loads((await ws.receive(timeout=10.0)).data)
                assert user["role"] == "user"
                await ws.send_str(json.dumps({"type": "cancel"}))
                cancelled = json.loads((await ws.receive(timeout=10.0)).data)
                assert cancelled["type"] == "cancelled"
            records = (await (await client.get(f"/api/chats/{conversation_id}")).json())["records"]
            assert [r.get("role") or r.get("type") for r in records] == ["user", "cancelled"]

    asyncio.run(main())


def test_chat_ws_subscribe_replays_full_transcript(populated_store: Path):
    """subscribe_conversation with the default after_seq replays everything."""
    transport = ScriptedTransport([anthropic_script(text="Hi there.")])

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "hello")
            conversation_id = frames[0]["conversation_id"]
            async with client.ws_connect("/api/chat") as ws:
                await ws.send_str(
                    json.dumps(
                        {"type": "subscribe_conversation", "conversation_id": conversation_id}
                    )
                )
                ack = json.loads((await ws.receive(timeout=10.0)).data)
                assert ack == {
                    "type": "subscribed_conversation",
                    "conversation_id": conversation_id,
                    "in_flight": False,
                }
                replay = await _collect_until_terminal(ws)
            assert replay == frames[1:]
            # Bad subscriptions get error frames.
            async with client.ws_connect("/api/chat") as ws:
                await ws.send_str(
                    json.dumps({"type": "subscribe_conversation", "conversation_id": "bad id!"})
                )
                msg = json.loads((await ws.receive(timeout=10.0)).data)
                assert msg["type"] == "error"

    asyncio.run(main())


def test_chats_recent_single_run(populated_store: Path):
    transport = ScriptedTransport(
        [anthropic_script(text="First answer."), anthropic_script(text="Second answer.")]
    )

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                first = await _chat_frames(ws, "first question")
            async with client.ws_connect("/api/chat") as ws:
                second = await _chat_frames(ws, "second question")
            resp = await client.get("/api/chats/recent")
            entries = (await resp.json())["conversations"]
            assert [e["title"] for e in entries] == ["second question", "first question"]
            assert [e["conversation_id"] for e in entries] == [
                second[0]["conversation_id"],
                first[0]["conversation_id"],
            ]
            for entry in entries:
                assert entry["run_id"] is None
                assert entry["in_flight"] is False
                assert entry["mtime"] is not None
            resp = await client.get("/api/chats/recent", params={"limit": "1"})
            entries = (await resp.json())["conversations"]
            assert [e["title"] for e in entries] == ["second question"]

    asyncio.run(main())


def test_chats_recent_registry_aggregates_across_runs(registry_stores: dict):
    live_id = registry_stores["live_id"]
    transport = ScriptedTransport(
        [anthropic_script(text="Global answer."), anthropic_script(text="Run answer.")]
    )

    async def main():
        app = build_app(None, static_dir=None, load_tokenizer=False, llm_transport=transport)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            # One cross-run (registry-level) chat and one per-run chat.
            async with client.ws_connect("/api/chat") as ws:
                global_frames = await _chat_frames(ws, "global question")
            async with client.ws_connect(f"/api/runs/{live_id}/chat") as ws:
                run_frames = await _chat_frames(ws, "run question")
            resp = await client.get("/api/chats/recent?limit=5")
            entries = (await resp.json())["conversations"]
            by_id = {e["conversation_id"]: e for e in entries}
            assert set(by_id) == {
                global_frames[0]["conversation_id"],
                run_frames[0]["conversation_id"],
            }
            assert by_id[global_frames[0]["conversation_id"]]["run_id"] is None
            assert by_id[run_frames[0]["conversation_id"]]["run_id"] == live_id
            for entry in entries:
                assert entry["in_flight"] is False
                assert entry["title"]
                assert entry["mtime"] is not None
            # Newest activity first.
            assert entries[0]["conversation_id"] == run_frames[0]["conversation_id"]

    asyncio.run(main())


def test_chat_system_prompt_includes_schema_card(maps_store: Path):
    """The chat system prompt carries this run's observed-keys card."""
    transport = ScriptedTransport([anthropic_script(text="hi")])

    async def main():
        app = build_app(maps_store, static_dir=None, load_tokenizer=False, llm_transport=transport)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "hello")
            assert frames[-1]["type"] == "done"
        system = transport.requests[0][2]["system"]
        assert "This run's observed keys" in system
        # The exact keys this run's data contains...
        assert "`group/score`" in system and "`acc`" in system
        assert "`dataset`" in system
        assert "`kl`" in system
        # ...plus the promoted views and runs-table columns.
        assert "parse_errors" in system and "context_overflows" in system
        assert "learning_rate" in system and "config_json" in system

    asyncio.run(main())


def test_format_schema_card_truncation_note():
    from tinker_cookbook.tokendb.agent_prompt import format_schema_card

    card = {
        "metrics_keys": ["acc"],
        "attrs_keys": [],
        "token_metrics_keys": [],
        "tags": [],
        "keys_truncated": True,
    }
    text = format_schema_card(card)
    assert "`acc`" in text
    assert "(none)" in text  # empty lists render explicitly
    assert "truncated" in text


def test_chat_ws_missing_key_error_frame(populated_store: Path):
    async def fn(client):
        async with client.ws_connect("/api/chat") as ws:
            frames = await _chat_frames(ws, "hello?")
        assert len(frames) == 1
        assert frames[0]["type"] == "error"
        assert frames[0]["code"] == "no_api_key"
        assert "ANTHROPIC_API_KEY" in frames[0]["error"]

    run_with_client(populated_store, fn)


def test_agent_config_endpoint_never_leaks_key(populated_store: Path):
    async def fn(client):
        resp = await client.get("/api/agent/config")
        assert await resp.json() == {
            "provider": "anthropic",
            "model": "claude-fable-5",
            "has_key": False,
            "providers": {
                "anthropic": {"has_key": False},
                "openai": {"has_key": False},
                "tinker": {"has_key": False},
            },
            "models": KNOWN_MODELS,
            "default_model": DEFAULT_MODELS,
        }
        resp = await client.post(
            "/api/agent/config",
            json={"provider": "openai", "model": "my-model", "api_key": "sk-supersecret"},
        )
        payload = await resp.json()
        assert payload["provider"] == "openai"
        assert payload["model"] == "my-model"
        assert payload["has_key"] is True
        resp = await client.get("/api/agent/config")
        assert "sk-supersecret" not in await resp.text()
        assert (await resp.json())["has_key"] is True
        # Clearing the key and rejecting unknown providers.
        resp = await client.post("/api/agent/config", json={"api_key": ""})
        assert (await resp.json())["has_key"] is False
        resp = await client.post("/api/agent/config", json={"provider": "nope"})
        assert resp.status == 400

    run_with_client(populated_store, fn)


def test_agent_config_has_key_from_environment(
    populated_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """An env-var key counts as configured, so the UI skips its setup card."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    async def fn(client):
        resp = await client.get("/api/agent/config")
        payload = await resp.json()
        assert payload["has_key"] is True
        assert "sk-from-env" not in await resp.text()
        # The other provider has no key.
        await client.post("/api/agent/config", json={"provider": "openai"})
        resp = await client.get("/api/agent/config")
        assert (await resp.json())["has_key"] is False

    run_with_client(populated_store, fn)


def test_agent_config_default_provider_from_openai_env(
    populated_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only OPENAI_API_KEY set: the effective default provider is openai."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    async def fn(client):
        resp = await client.get("/api/agent/config")
        payload = await resp.json()
        assert payload["provider"] == "openai"
        assert payload["model"] == DEFAULT_MODELS["openai"]
        assert payload["has_key"] is True
        assert payload["providers"] == {
            "anthropic": {"has_key": False},
            "openai": {"has_key": True},
            "tinker": {"has_key": False},
        }

    run_with_client(populated_store, fn)


def test_agent_config_default_provider_preference_order(
    populated_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both anthropic and openai keys set: anthropic wins (preference order)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-env")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    async def fn(client):
        resp = await client.get("/api/agent/config")
        payload = await resp.json()
        assert payload["provider"] == "anthropic"
        assert payload["has_key"] is True
        assert payload["providers"]["anthropic"] == {"has_key": True}
        assert payload["providers"]["openai"] == {"has_key": True}

    run_with_client(populated_store, fn)


def test_agent_config_default_provider_no_keys(populated_store: Path):
    """No keys anywhere: falls back to anthropic with has_key false."""

    async def fn(client):
        resp = await client.get("/api/agent/config")
        payload = await resp.json()
        assert payload["provider"] == "anthropic"
        assert payload["has_key"] is False

    run_with_client(populated_store, fn)


def test_agent_config_explicit_provider_pins(
    populated_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """An explicit POST pins the provider even when it has no key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    async def fn(client):
        resp = await client.post("/api/agent/config", json={"provider": "anthropic"})
        payload = await resp.json()
        assert payload["provider"] == "anthropic"
        assert payload["has_key"] is False
        # Sticks across subsequent GETs despite openai having a key.
        resp = await client.get("/api/agent/config")
        payload = await resp.json()
        assert payload["provider"] == "anthropic"
        assert payload["has_key"] is False
        assert payload["providers"]["openai"] == {"has_key": True}

    run_with_client(populated_store, fn)


def test_agent_config_runtime_key_counts_for_all_providers(populated_store: Path):
    """A runtime-configured key marks every provider as ready."""

    async def fn(client):
        resp = await client.post("/api/agent/config", json={"api_key": "sk-runtime"})
        payload = await resp.json()
        assert payload["provider"] == "anthropic"  # first in preference order
        assert payload["has_key"] is True
        assert payload["providers"] == {
            "anthropic": {"has_key": True},
            "openai": {"has_key": True},
            "tinker": {"has_key": True},
        }

    run_with_client(populated_store, fn)


class CountingModelsFetcher:
    """Stub for the tinker supported-models fetch; counts calls."""

    def __init__(self, models=None, error: Exception | None = None) -> None:
        self.models = models or []
        self.error = error
        self.calls: list[str] = []

    def __call__(self, api_key: str) -> list[str]:
        self.calls.append(api_key)
        if self.error is not None:
            raise self.error
        return list(self.models)


def test_agent_config_tinker_models(populated_store: Path, monkeypatch: pytest.MonkeyPatch):
    # An anthropic key too, so the auto-detected default provider stays
    # non-tinker and the plain GET below exercises the "no fetch" path.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    fetcher = CountingModelsFetcher(
        models=["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-8B"]
    )

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, tinker_models_fetcher=fetcher
        )
        async with TestClient(TestServer(app)) as client:
            # Plain GET with a non-tinker provider never touches the fetcher.
            resp = await client.get("/api/agent/config")
            payload = await resp.json()
            assert payload["models"]["tinker"] == []
            assert "tinker_models_error" not in payload
            assert fetcher.calls == []

            # ?provider=tinker fetches the list and picks an instruct default.
            resp = await client.get("/api/agent/config?provider=tinker")
            payload = await resp.json()
            assert payload["models"]["tinker"] == fetcher.models
            assert payload["default_model"]["tinker"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
            assert payload["tinker_models_error"] is None
            assert len(fetcher.calls) == 1

            # Cached: a second GET within the TTL does not refetch. Switching
            # the configured provider to tinker also serves from the cache.
            await client.get("/api/agent/config?provider=tinker")
            resp = await client.post("/api/agent/config", json={"provider": "tinker"})
            payload = await resp.json()
            assert payload["models"]["tinker"] == fetcher.models
            assert payload["has_key"] is True
            assert len(fetcher.calls) == 1

    asyncio.run(main())


def test_agent_config_tinker_without_key(populated_store: Path):
    fetcher = CountingModelsFetcher(models=["Qwen/Qwen3-8B"])

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, tinker_models_fetcher=fetcher
        )
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agent/config?provider=tinker")
            payload = await resp.json()
            assert payload["models"]["tinker"] == []
            assert "TINKER_API_KEY" in payload["tinker_models_error"]
            assert fetcher.calls == []  # no key, nothing to fetch

    asyncio.run(main())


def test_agent_config_tinker_fetch_error(populated_store: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    fetcher = CountingModelsFetcher(error=RuntimeError("capabilities unavailable"))

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, tinker_models_fetcher=fetcher
        )
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agent/config?provider=tinker")
            payload = await resp.json()
            assert payload["models"]["tinker"] == []
            assert "capabilities unavailable" in payload["tinker_models_error"]
            # The failure is cached too (no hammering on config polls).
            await client.get("/api/agent/config?provider=tinker")
            assert len(fetcher.calls) == 1

    asyncio.run(main())


def test_visuals_publish_list_and_serve(populated_store: Path):
    html = "<!doctype html><html><body><svg></svg></body></html>"
    transport = ScriptedTransport(
        [
            anthropic_script(
                tool_calls=[
                    ("t1", "publish_visual", {"title": "Reward", "description": "d", "html": html})
                ]
            ),
            anthropic_script(text="Published."),
        ]
    )

    async def main():
        app = build_app(
            populated_store, static_dir=None, load_tokenizer=False, llm_transport=transport
        )
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "chart the reward")
            visual = next(f for f in frames if f.get("type") == "visual_published")
            assert visual["title"] == "Reward"
            assert visual["url"].startswith("/visuals/reward-")

            resp = await client.get("/api/visuals")
            visuals = (await resp.json())["visuals"]
            assert [v["name"] for v in visuals] == [visual["name"]]

            resp = await client.get(visual["url"])
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/html")
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["Cache-Control"] == "no-cache"
            assert await resp.text() == html

            resp = await client.get("/visuals/does-not-exist.html")
            assert resp.status == 404
            resp = await client.get("/visuals/..%2Frun.json")
            assert resp.status == 404

    asyncio.run(main())


def test_registry_global_chat_lists_runs(registry_stores: dict):
    live_id = registry_stores["live_id"]
    transport = ScriptedTransport(
        [
            anthropic_script(text="Looking.", tool_calls=[("t1", "list_runs", {})]),
            anthropic_script(
                tool_calls=[
                    (
                        "t2",
                        "sql",
                        {"run_id": live_id, "query": "SELECT count(*) AS n FROM rollouts"},
                    )
                ]
            ),
            anthropic_script(text="The live run has 3 rows."),
        ]
    )

    async def main():
        app = build_app(None, static_dir=None, load_tokenizer=False, llm_transport=transport)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "what runs exist?")
            assert frames[-1]["type"] == "done"
            list_result = next(
                f for f in frames if f.get("type") == "tool_result" and f["name"] == "list_runs"
            )
            assert '"runs"' in list_result["preview"]
            # The model saw the full (untruncated-by-preview) run listing.
            conversation_id = frames[0]["conversation_id"]
            records = (await (await client.get(f"/api/chats/{conversation_id}")).json())["records"]
            list_runs_record = next(
                r for r in records if r["kind"] == "message" and r.get("role") == "tool"
            )
            assert live_id in list_runs_record["content"]
            sql_result = next(
                f for f in frames if f.get("type") == "tool_result" and f["name"] == "sql"
            )
            assert '"n": 3' in sql_result["preview"]
            # Registry-mode prompt teaches the per-run SQL endpoint template.
            assert "/api/runs/{run_id}/sql" in transport.requests[0][2]["system"]

            # The conversation is stored under the registry directory.
            registry_dir = Path(os.environ["TINKER_TOKENDB_REGISTRY"])
            assert list((registry_dir / "chats").glob("*.jsonl"))

            # Per-run chat endpoints are mounted too.
            resp = await client.get(f"/api/runs/{live_id}/chats")
            assert resp.status == 200
            assert (await resp.json())["conversations"] == []

    asyncio.run(main())


def test_registry_chat_sql_tool_cross_run(registry_stores: dict):
    """The registry chat's sql tool without a run_id spans every run."""
    transport = ScriptedTransport(
        [
            anthropic_script(
                text="Comparing runs.",
                tool_calls=[
                    (
                        "t1",
                        "sql",
                        {
                            "query": (
                                "SELECT run_id, count(*) AS n FROM rollouts "
                                "GROUP BY run_id ORDER BY n DESC"
                            )
                        },
                    )
                ],
            ),
            anthropic_script(text="One run has 3 rows, the other 1."),
        ]
    )

    async def main():
        app = build_app(None, static_dir=None, load_tokenizer=False, llm_transport=transport)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/agent/config", json={"api_key": "sk-test"})
            async with client.ws_connect("/api/chat") as ws:
                frames = await _chat_frames(ws, "compare the runs")
            assert frames[-1]["type"] == "done"
            sql_result = next(
                f for f in frames if f.get("type") == "tool_result" and f["name"] == "sql"
            )
            assert not sql_result["is_error"]
            # The model saw both runs' counts from ONE query with no run_id.
            conversation_id = frames[0]["conversation_id"]
            records = (await (await client.get(f"/api/chats/{conversation_id}")).json())["records"]
            tool_record = next(
                r for r in records if r["kind"] == "message" and r.get("role") == "tool"
            )
            content = json.loads(tool_record["content"])
            assert {row["run_id"]: row["n"] for row in content["rows"]} == {
                registry_stores["live_id"]: 3,
                registry_stores["stale_id"]: 1,
            }
            # The registry prompt teaches cross-run SQL (root endpoint, run_id
            # as a column) and carries the aggregated multi-run schema card.
            system = transport.requests[0][2]["system"]
            assert 'const SQL_URL = "/api/sql"' in system
            assert "GROUP BY run_id" in system
            assert "rollouts_latest` for cross-run aggregates" in system
            assert "Observed keys across runs (2 runs)" in system
            assert "/api/runs/{run_id}/sql" in system  # per-run template stays documented

    asyncio.run(main())


def test_chat_ws_cancel(populated_store: Path):
    """A cancel frame while no turn is running is a no-op; bad frames error."""

    async def fn(client):
        async with client.ws_connect("/api/chat") as ws:
            await ws.send_str(json.dumps({"type": "cancel"}))
            await ws.send_str(json.dumps({"type": "bogus"}))
            msg = json.loads((await ws.receive(timeout=5.0)).data)
            assert msg["type"] == "error"
            assert "bogus" in msg["error"]

    run_with_client(populated_store, fn)
