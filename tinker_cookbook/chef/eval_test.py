"""Tests for Tinker Chef eval API routes."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def eval_store(tmp_path: Path) -> Path:
    """Create a mock eval store directory."""
    eval_dir = tmp_path / "eval_store"
    eval_dir.mkdir()

    # runs.jsonl index
    run_index = [
        {"run_id": "eval_001", "timestamp": "2024-04-01T12:00:00", "model_name": "Llama-3.1-8B"},
        {"run_id": "eval_002", "timestamp": "2024-04-02T12:00:00", "model_name": "Llama-3.1-8B"},
    ]
    (eval_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in run_index) + "\n"
    )

    runs_dir = eval_dir / "runs"

    # eval_001: gsm8k + ifeval
    run1 = runs_dir / "eval_001"
    run1.mkdir(parents=True)
    (run1 / "metadata.json").write_text(json.dumps({
        "run_id": "eval_001",
        "model_name": "Llama-3.1-8B",
        "checkpoint_path": "tinker:///ckpt/step_100",
        "checkpoint_name": "step_100",
        "benchmarks": ["gsm8k", "ifeval"],
        "timestamp": "2024-04-01T12:00:00",
        "scores": {"gsm8k": 0.85, "ifeval": 0.72},
    }))

    gsm8k_dir = run1 / "gsm8k"
    gsm8k_dir.mkdir()
    (gsm8k_dir / "result.json").write_text(json.dumps({
        "name": "gsm8k", "score": 0.85, "num_examples": 100,
        "num_correct": 85, "num_errors": 2, "metrics": {}, "time_seconds": 120.5,
    }))
    gsm8k_trajectories = [
        {
            "idx": 0, "benchmark": "gsm8k", "example_id": "abc123",
            "turns": [
                {"role": "user", "content": "What is 2+2?", "token_count": 10, "metadata": {}},
                {"role": "assistant", "content": "4", "token_count": 5, "metadata": {}},
            ],
            "reward": 1.0, "metrics": {}, "logs": {"expected": "4", "extracted": "4"},
            "error": None, "time_seconds": 1.2,
        },
        {
            "idx": 1, "benchmark": "gsm8k", "example_id": "def456",
            "turns": [
                {"role": "user", "content": "What is 3*5?", "token_count": 10, "metadata": {}},
                {"role": "assistant", "content": "16", "token_count": 5, "metadata": {}},
            ],
            "reward": 0.0, "metrics": {}, "logs": {"expected": "15", "extracted": "16"},
            "error": None, "time_seconds": 1.5,
        },
        {
            "idx": 2, "benchmark": "gsm8k", "example_id": "ghi789",
            "turns": [], "reward": 0.0, "metrics": {}, "logs": {},
            "error": "Timeout after 300s", "time_seconds": 300.0,
        },
    ]
    (gsm8k_dir / "trajectories.jsonl").write_text(
        "\n".join(json.dumps(t) for t in gsm8k_trajectories) + "\n"
    )

    ifeval_dir = run1 / "ifeval"
    ifeval_dir.mkdir()
    (ifeval_dir / "result.json").write_text(json.dumps({
        "name": "ifeval", "score": 0.72, "num_examples": 50,
        "num_correct": 36, "num_errors": 0, "metrics": {}, "time_seconds": 60.0,
    }))

    # eval_002: gsm8k only, improved
    run2 = runs_dir / "eval_002"
    run2.mkdir()
    (run2 / "metadata.json").write_text(json.dumps({
        "run_id": "eval_002",
        "model_name": "Llama-3.1-8B",
        "checkpoint_path": "tinker:///ckpt/step_200",
        "checkpoint_name": "step_200",
        "benchmarks": ["gsm8k"],
        "timestamp": "2024-04-02T12:00:00",
        "scores": {"gsm8k": 0.92},
    }))
    gsm8k_dir2 = run2 / "gsm8k"
    gsm8k_dir2.mkdir()
    (gsm8k_dir2 / "result.json").write_text(json.dumps({
        "name": "gsm8k", "score": 0.92, "num_examples": 100,
        "num_correct": 92, "num_errors": 1, "metrics": {}, "time_seconds": 110.0,
    }))

    return eval_dir


class TestEvalAPI:
    @pytest.fixture
    def client(self, eval_store: Path):
        from fastapi.testclient import TestClient
        from tinker_cookbook.chef.app import create_app

        # Create a parent dir containing both a training run and eval data
        parent = eval_store.parent
        run_dir = parent / "my_run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n")

        # Move eval_store to "eval" (discoverable name)
        target = parent / "eval"
        if not target.exists():
            eval_store.rename(target)

        app = create_app(parent)
        return TestClient(app)

    def test_list_eval_runs(self, client) -> None:
        resp = client.get("/api/eval/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_eval_run(self, client) -> None:
        resp = client.get("/api/eval/runs/eval_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["model_name"] == "Llama-3.1-8B"
        assert "gsm8k" in data["results"]

    def test_get_eval_trajectories(self, client) -> None:
        resp = client.get("/api/eval/runs/eval_001/gsm8k/trajectories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_filter_correct_only(self, client) -> None:
        resp = client.get("/api/eval/runs/eval_001/gsm8k/trajectories?correct_only=true")
        data = resp.json()
        assert data["total"] == 1

    def test_filter_errors_only(self, client) -> None:
        resp = client.get("/api/eval/runs/eval_001/gsm8k/trajectories?errors_only=true")
        data = resp.json()
        assert data["total"] == 1

    def test_get_trajectory_detail(self, client) -> None:
        resp = client.get("/api/eval/runs/eval_001/gsm8k/trajectories/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["example_id"] == "abc123"

    def test_eval_run_404(self, client) -> None:
        resp = client.get("/api/eval/runs/nonexistent")
        assert resp.status_code == 404

    def test_scores_table(self, client) -> None:
        resp = client.get("/api/eval/scores")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
