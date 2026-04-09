"""Tests for Tinker Chef API routes."""

import json
from pathlib import Path

import pytest


def _create_run_dir(run_dir: Path) -> None:
    """Create a minimal training run directory with fixture data."""
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": "Llama-3.1-8B",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "n_batches": 100,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics_lines = []
    for step in range(5):
        metrics_lines.append(
            json.dumps({
                "step": step,
                "train_mean_nll": 2.5 - step * 0.1,
                "env/all/reward/total": step * 0.2,
                "time/forward_backward": 1.0 + step * 0.1,
            })
        )
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n")

    ckpt = {
        "state_path": "tinker:///ckpt/000002",
        "name": "000002",
        "kind": "both",
        "timestamp": 1700000000.0,
        "loop_state": {"epoch": 0, "batch": 2},
    }
    (run_dir / "checkpoints.jsonl").write_text(json.dumps(ckpt) + "\n")

    timing_lines = [
        {"step": 0, "spans": [
            {"name": "forward_backward", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5},
            {"name": "optim_step", "duration": 0.5, "wall_start": 1.5, "wall_end": 2.0},
            {"name": "policy_sample", "duration": 0.8, "wall_start": 0.0, "wall_end": 0.8},
            {"name": "env_step", "duration": 0.6, "wall_start": 0.2, "wall_end": 0.8},
        ]},
        {"step": 1, "spans": [
            {"name": "forward_backward", "duration": 1.3, "wall_start": 2.0, "wall_end": 3.3},
        ]},
    ]
    (run_dir / "timing_spans.jsonl").write_text(
        "\n".join(json.dumps(s) for s in timing_lines) + "\n"
    )

    for iteration in [0, 2, 4]:
        iter_dir = run_dir / f"iteration_{iteration:06d}"
        iter_dir.mkdir()

        rollouts = []
        for group_idx in range(2):
            for traj_idx in range(3):
                rollouts.append({
                    "schema_version": 1, "split": "train", "iteration": iteration,
                    "group_idx": group_idx, "traj_idx": traj_idx,
                    "tags": ["math", "gsm8k"] if group_idx == 0 else ["code", "humaneval"],
                    "sampling_client_step": iteration,
                    "total_reward": (group_idx + traj_idx) * 0.3,
                    "final_reward": (group_idx + traj_idx) * 0.1,
                    "trajectory_metrics": {"custom": 1.5},
                    "steps": [
                        {"step_idx": 0, "ob_len": 128, "ac_len": 45, "reward": 0.0,
                         "episode_done": False, "metrics": {"step_time": 0.01}, "logs": {}},
                        {"step_idx": 1, "ob_len": 200, "ac_len": 60, "reward": 1.0,
                         "episode_done": True, "metrics": {"step_time": 0.02}, "logs": {}},
                    ],
                    "final_ob_len": 512,
                })
        (iter_dir / "train_rollout_summaries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rollouts) + "\n"
        )

        logtree = {
            "title": f"RL Iteration {iteration}",
            "started_at": "2024-04-04T12:34:56.789123",
            "root": {
                "tag": "div", "attrs": {"class": "lt-root"},
                "children": [{"tag": "p", "children": ["Episode completed"]}],
                "data": {"type": "conversation", "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]},
            },
        }
        (iter_dir / "train_logtree.json").write_text(json.dumps(logtree))

    iter4 = run_dir / "iteration_000004"
    eval_rollouts = [{
        "schema_version": 1, "split": "eval", "iteration": 4,
        "group_idx": 0, "traj_idx": 0, "tags": ["test_set"],
        "total_reward": 0.9, "final_reward": 0.9,
        "steps": [{"step_idx": 0, "ob_len": 100, "ac_len": 30, "reward": 0.9,
                    "episode_done": True, "metrics": {}, "logs": {}}],
        "final_ob_len": 200,
    }]
    (iter4 / "eval_test_rollout_summaries.jsonl").write_text(
        json.dumps(eval_rollouts[0]) + "\n"
    )


@pytest.fixture
def client(tmp_path: Path):
    """Create a FastAPI TestClient with a single run."""
    from fastapi.testclient import TestClient

    from tinker_cookbook.chef.app import create_app

    run_dir = tmp_path / "my_run"
    _create_run_dir(run_dir)
    app = create_app(run_dir)
    return TestClient(app)


@pytest.fixture
def multi_client(tmp_path: Path):
    """TestClient with multiple runs."""
    from fastapi.testclient import TestClient

    from tinker_cookbook.chef.app import create_app

    parent = tmp_path / "experiments"
    parent.mkdir()
    for name in ("run_a", "run_b"):
        run_dir = parent / name
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "loss": 2.0}) + "\n"
        )
        (run_dir / "config.json").write_text(json.dumps({"model_name": name}))

    app = create_app(parent)
    return TestClient(app)


# ── Runs API ──────────────────────────────────────────────────────────


class TestRunsAPI:
    def test_list_runs(self, client) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        run = data[0]
        assert "run_id" in run
        assert run["has_config"] is True
        assert run["has_metrics"] is True
        assert "config_summary" in run
        assert run["config_summary"]["model_name"] == "Llama-3.1-8B"

    def test_list_runs_multi(self, multi_client) -> None:
        resp = multi_client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_run(self, client) -> None:
        # First get the run_id
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert "config" in data
        assert data["total_steps"] == 5

    def test_get_run_404(self, client) -> None:
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_config(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "Llama-3.1-8B"

    def test_list_iterations(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/iterations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["iteration"] == 0
        assert data[0]["has_train_rollouts"] is True

    def test_get_checkpoints(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "000002"


# ── Metrics API ───────────────────────────────────────────────────────


class TestMetricsAPI:
    def test_get_metrics(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 5
        assert len(data["records"]) == 5
        assert data["records"][0]["step"] == 0

    def test_get_metrics_filtered(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/metrics", params={"keys": "env/*"})
        assert resp.status_code == 200
        data = resp.json()
        for record in data["records"]:
            non_step_keys = [k for k in record if k != "step"]
            for key in non_step_keys:
                assert key.startswith("env/")

    def test_get_metrics_multiple_patterns(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/metrics",
            params={"keys": "env/*,time/*"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for record in data["records"]:
            non_step_keys = [k for k in record if k != "step"]
            for key in non_step_keys:
                assert key.startswith("env/") or key.startswith("time/")

    def test_get_metric_keys(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/metrics/keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert "train_mean_nll" in keys
        assert "env/all/reward/total" in keys
        assert "step" not in keys

    def test_metrics_404(self, client) -> None:
        resp = client.get("/api/runs/nonexistent/metrics")
        assert resp.status_code == 404


# ── Rollouts API ──────────────────────────────────────────────────────


class TestRolloutsAPI:
    def test_get_rollouts(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/iterations/0/rollouts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 6
        assert len(data["rollouts"]) == 6
        assert "available_tags" in data
        assert "math" in data["available_tags"]

    def test_filter_by_tag(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/iterations/0/rollouts",
            params={"tag": "math"},
        )
        data = resp.json()
        assert data["total"] == 3  # Only group 0 has "math" tag

    def test_filter_by_reward(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/iterations/0/rollouts",
            params={"min_reward": 0.5},
        )
        data = resp.json()
        # Only rollouts with total_reward >= 0.5
        for r in data["rollouts"]:
            assert r["total_reward"] >= 0.5

    def test_rollout_detail(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/iterations/0/rollouts/0/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_idx"] == 0
        assert data["traj_idx"] == 0
        assert len(data["steps"]) == 2
        assert data["steps"][0]["ob_len"] == 128

    def test_rollout_detail_404(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/iterations/0/rollouts/99/99")
        assert resp.status_code == 404

    def test_eval_rollouts(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/iterations/4/rollouts",
            params={"split": "eval", "label": "test"},
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["rollouts"][0]["total_reward"] == 0.9

    def test_logtree(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/iterations/0/logtree")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "RL Iteration 0"
        assert "root" in data


# ── Timing API ────────────────────────────────────────────────────────


class TestTimingAPI:
    def test_get_timing(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/timing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 2  # 2 step records

    def test_timing_step_filter(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(
            f"/api/runs/{run_id}/timing",
            params={"step_start": 0, "step_end": 0},
        )
        data = resp.json()
        assert data["total_records"] == 1  # One step record for step 0

    def test_timing_flat(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/timing/flat")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_spans"] == 5  # 4 spans in step 0 + 1 in step 1

    def test_concurrency_analysis(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/timing/concurrency/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 0
        assert len(data["spans"]) == 4
        assert data["max_concurrency"] >= 3  # 3 overlapping spans
        assert len(data["timeline"]) > 0

    def test_timing_tree(self, client) -> None:
        runs = client.get("/api/runs").json()
        run_id = runs[0]["run_id"]

        resp = client.get(f"/api/runs/{run_id}/timing/tree/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 0
        assert data["root"] is not None
        assert data["root"]["name"] == "iteration"
        assert len(data["root"]["children"]) > 0

    def test_timing_404(self, client) -> None:
        resp = client.get("/api/runs/nonexistent/timing")
        assert resp.status_code == 404


# ── Sources API ──────────────────────────────────────────────────────


class TestSourcesAPI:
    def test_list_sources(self, client) -> None:
        resp = client.get("/api/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "local"
        assert "url" in data[0]

    def test_add_source(self, client, tmp_path: Path) -> None:
        new_dir = tmp_path / "extra_runs"
        new_dir.mkdir()
        (new_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "loss": 1.0}) + "\n"
        )

        resp = client.post("/api/sources", json={"uri": str(new_dir)})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) == 2
        assert data["runs_discovered"] >= 2  # original + new

    def test_add_source_invalid_path(self, client) -> None:
        resp = client.post("/api/sources", json={"uri": "/nonexistent/path/12345"})
        assert resp.status_code == 400

    def test_remove_source(self, client, tmp_path: Path) -> None:
        # First add a source
        new_dir = tmp_path / "to_remove"
        new_dir.mkdir()
        (new_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 0, "loss": 1.0}) + "\n"
        )
        client.post("/api/sources", json={"uri": str(new_dir)})

        # Get current sources
        sources = client.get("/api/sources").json()
        assert len(sources) == 2

        # Remove the newly added one
        new_url = [s["url"] for s in sources if str(new_dir) in s["url"]][0]
        resp = client.delete(f"/api/sources?url={new_url}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) == 1

    def test_refresh_sources(self, client) -> None:
        resp = client.post("/api/sources/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert data["runs"] >= 1
