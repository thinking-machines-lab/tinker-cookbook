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


def test_registry_mode_requires_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", "")
    with pytest.raises(ValueError, match="registry is disabled"):
        build_app(None, static_dir=None, load_tokenizer=False)


# --- Chat agent endpoints (websocket chat, config, visuals) ---

from tinker_cookbook.tokendb.agent_test import ScriptedTransport, anthropic_script  # noqa: E402
from tinker_cookbook.tokendb.llm import DEFAULT_MODELS, KNOWN_MODELS  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat tests must control key presence themselves."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


async def _chat_frames(ws, text: str, conversation_id: str | None = None) -> list[dict]:
    """Send a user_message and collect frames through the terminal one."""
    message: dict = {"type": "user_message", "text": text}
    if conversation_id is not None:
        message["conversation_id"] = conversation_id
    await ws.send_str(json.dumps(message))
    frames = []
    while True:
        frames.append(json.loads((await ws.receive(timeout=10.0)).data))
        if frames[-1]["type"] in ("done", "error", "cancelled"):
            return frames


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
            types = [f["type"] for f in frames]
            assert types == [
                "conversation",
                "text_delta",
                "tool_call",
                "tool_result",
                "text_delta",
                "done",
            ]
            assert frames[1]["text"] == "Checking."
            assert frames[2]["name"] == "sql"
            assert not frames[3]["is_error"]
            assert frames[4]["text"] == "There are 6 rows."

            # The system prompt carried the run's SQL endpoint and schema docs.
            system = transport.requests[0][2]["system"]
            assert "/api/sql" in system
            assert "ob_is_delta" in system

            # Transcript endpoints.
            resp = await client.get("/api/chats")
            conversations = (await resp.json())["conversations"]
            assert [c["conversation_id"] for c in conversations] == [conversation_id]
            assert conversations[0]["title"] == "how many rows?"
            resp = await client.get(f"/api/chats/{conversation_id}")
            records = (await resp.json())["records"]
            assert [r.get("role") for r in records if r["kind"] == "message"] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            resp = await client.get("/api/chats/nope-id")
            assert resp.status == 404

    asyncio.run(main())


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
            visual = next(f for f in frames if f["type"] == "visual_published")
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
                f for f in frames if f["type"] == "tool_result" and f["name"] == "list_runs"
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
                f for f in frames if f["type"] == "tool_result" and f["name"] == "sql"
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
