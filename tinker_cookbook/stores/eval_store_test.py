"""Tests for EvalStore — both constructor forms and delete_run."""

import json
from pathlib import Path

import pytest

from tinker_cookbook.stores.eval_store import EvalStore, RunMetadata
from tinker_cookbook.stores.storage import LocalStorage, storage_join


class TestEvalStorePathConstructor:
    """Tests using the backward-compat ``EvalStore("/path")`` form."""

    def test_create_and_list_runs(self, tmp_path: Path) -> None:
        store = EvalStore(str(tmp_path / "eval"))
        run_id = store.create_run("test-model", ["gsm8k"])
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == run_id
        assert runs[0].model_name == "test-model"

    def test_run_dir(self, tmp_path: Path) -> None:
        store = EvalStore(str(tmp_path / "eval"))
        run_id = store.create_run("m", ["b"])
        rd = store.run_dir(run_id)
        assert run_id in rd
        assert str(tmp_path) in rd


class TestEvalStoreStorageConstructor:
    """Tests using the new ``EvalStore(storage, prefix)`` form."""

    def test_create_and_list_runs(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        store = EvalStore(storage, "my_eval")
        run_id = store.create_run("test-model", ["gsm8k"])
        # Verify files written under prefix
        assert storage.exists("my_eval/runs.jsonl")
        assert storage.exists(f"my_eval/runs/{run_id}/metadata.json")
        runs = store.list_runs()
        assert len(runs) == 1

    def test_run_dir(self, tmp_path: Path) -> None:
        store = EvalStore(LocalStorage(tmp_path), "eval")
        run_id = store.create_run("m", ["b"])
        rd = store.run_dir(run_id)
        assert "eval/runs" in rd
        assert run_id in rd

    def test_read_run(self, tmp_path: Path) -> None:
        store = EvalStore(LocalStorage(tmp_path), "eval")
        run_id = store.create_run("model-a", ["bench1", "bench2"], checkpoint_name="step_100")
        meta = store.read_run(run_id)
        assert meta.model_name == "model-a"
        assert meta.checkpoint_name == "step_100"
        assert meta.benchmarks == ["bench1", "bench2"]

    def test_read_run_missing(self, tmp_path: Path) -> None:
        store = EvalStore(LocalStorage(tmp_path), "eval")
        with pytest.raises(FileNotFoundError):
            store.read_run("nonexistent")


class TestDeleteRun:
    def test_delete_run_removes_data(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        store = EvalStore(storage, "eval")
        run_id = store.create_run("model", ["gsm8k"])
        # Write some benchmark data
        storage.write(
            f"eval/runs/{run_id}/gsm8k/result.json",
            json.dumps(
                {"name": "gsm8k", "score": 0.5, "num_examples": 10, "num_correct": 5}
            ).encode(),
        )
        storage.write(f"eval/runs/{run_id}/gsm8k/trajectories.jsonl", b"")
        # Verify data exists
        assert len(store.list_runs()) == 1
        assert store.list_benchmarks(run_id) == ["gsm8k"]
        # Delete
        store.delete_run(run_id)
        # Metadata gone → list_runs excludes it
        assert len(store.list_runs()) == 0
        assert not storage.exists(f"eval/runs/{run_id}/metadata.json")
        assert not storage.exists(f"eval/runs/{run_id}/gsm8k/result.json")
        # Directories should be cleaned up too
        assert not (tmp_path / "eval" / "runs" / run_id / "gsm8k").exists()
        assert not (tmp_path / "eval" / "runs" / run_id).exists()

    @pytest.mark.parametrize("prefix", ["", "eval"])
    @pytest.mark.parametrize("with_metadata", [False, True])
    @pytest.mark.parametrize("transcript", [b"", b'{"idx": 0, "reward": 1.0}\n'])
    def test_delete_run_removes_unfinished_benchmark(
        self, tmp_path: Path, prefix: str, with_metadata: bool, transcript: bytes
    ) -> None:
        storage = LocalStorage(tmp_path)
        store = EvalStore(storage, prefix)
        if with_metadata:
            store.create_run("model", ["complete", "unfinished"], run_id="target")
        run_path = storage_join(prefix, "runs", "target")
        storage.write(storage_join(run_path, "complete", "result.json"), b"{}")
        storage.write(storage_join(run_path, "complete", "trajectories.jsonl"), transcript)
        storage.write(storage_join(run_path, "unfinished", "trajectories.jsonl"), transcript)
        storage.write(storage_join(run_path, "summary.json"), b"{}")
        store.create_run("model", [], run_id="target-extra")
        other_store = EvalStore(storage, "other")
        other_store.create_run("model", [], run_id="target")
        index = storage.read(storage_join(prefix, "runs.jsonl"))

        assert store.list_benchmarks("target") == ["complete"]
        store.delete_run("target")
        store.delete_run("target")

        assert not storage.exists(run_path)
        assert store.list_benchmarks("target") == []
        assert [run.run_id for run in store.list_runs()] == ["target-extra"]
        assert storage.read(storage_join(prefix, "runs.jsonl")) == index
        assert store.read_run("target-extra").model_name == "model"
        assert other_store.read_run("target").model_name == "model"

    @pytest.mark.parametrize("prefix", ["", "eval"])
    @pytest.mark.parametrize("with_metadata", [False, True])
    def test_delete_empty_run(self, tmp_path: Path, prefix: str, with_metadata: bool) -> None:
        storage = LocalStorage(tmp_path)
        store = EvalStore(storage, prefix)
        run_path = storage_join(prefix, "runs", "empty")
        if with_metadata:
            store.create_run("model", [], run_id="empty")
        else:
            (tmp_path / run_path).mkdir(parents=True)

        store.delete_run("empty")
        store.delete_run("empty")

        assert not storage.exists(run_path)

    def test_delete_run_leaves_unknown_artifacts(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        store = EvalStore(storage)
        store.create_run("model", [], run_id="target")
        storage.write("runs/target/note.txt", b"unrecognized artifact")
        storage.write("runs/target/unfinished/trajectories.jsonl", b"{}\n")

        store.delete_run("target")

        assert not storage.exists("runs/target/unfinished/trajectories.jsonl")
        assert storage.read("runs/target/note.txt") == b"unrecognized artifact"
        assert store.list_runs() == []

    def test_delete_run_idempotent(self, tmp_path: Path) -> None:
        store = EvalStore(LocalStorage(tmp_path), "eval")
        # Deleting a nonexistent run should not raise
        store.delete_run("nonexistent")


class TestRunMetadata:
    def test_from_dict_missing_config(self) -> None:
        """from_dict handles missing config field (backward compat)."""
        d = {
            "run_id": "r1",
            "model_name": "m",
            "checkpoint_path": None,
            "checkpoint_name": None,
            "benchmarks": [],
            "timestamp": "t",
        }
        meta = RunMetadata.from_dict(d)
        assert meta.config == {}

    def test_from_dict_extra_fields(self) -> None:
        """from_dict ignores unknown fields."""
        d = {
            "run_id": "r1",
            "model_name": "m",
            "checkpoint_path": None,
            "checkpoint_name": None,
            "benchmarks": [],
            "timestamp": "t",
            "unknown_field": "ignored",
        }
        meta = RunMetadata.from_dict(d)
        assert meta.run_id == "r1"
