"""Tests for TrainingRunStore, IncrementalReader, and RunRegistry."""

import json
import pickle
import threading
from pathlib import Path

import pytest

from tinker_cookbook.stores._incremental import IncrementalReader
from tinker_cookbook.stores.registry import RunRegistry
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Create a minimal training run directory."""
    d = tmp_path / "my_run"
    d.mkdir()

    (d / "config.json").write_text(
        json.dumps(
            {
                "model_name": "Llama-3.1-8B",
                "learning_rate": 1e-4,
                "loss_fn": "importance_sampling",
                "lora_rank": 32,
            },
            indent=2,
        )
    )

    lines = []
    for step in range(5):
        lines.append(
            json.dumps(
                {
                    "step": step,
                    "env/all/reward/total": step * 0.2,
                    "train_mean_nll": 2.5 - step * 0.1,
                }
            )
        )
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")

    (d / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "name": "000004",
                "batch": 4,
                "final": True,
                "state_path": "tinker:///ckpt/final",
            }
        )
        + "\n"
    )

    timing = [
        json.dumps(
            {
                "step": 0,
                "spans": [
                    {"name": "sampling", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5},
                    {"name": "train_step", "duration": 0.5, "wall_start": 1.5, "wall_end": 2.0},
                ],
            }
        )
    ]
    (d / "timing_spans.jsonl").write_text("\n".join(timing) + "\n")

    for iteration in [0, 2]:
        iter_dir = d / f"iteration_{iteration:06d}"
        iter_dir.mkdir()
        rollouts = []
        for g in range(2):
            for t in range(2):
                rollouts.append(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "split": "train",
                            "iteration": iteration,
                            "group_idx": g,
                            "traj_idx": t,
                            "tags": ["math"],
                            "total_reward": g * 0.5 + t * 0.1,
                            "final_reward": 0.0,
                            "trajectory_metrics": {},
                            "final_ob_len": 100,
                            "steps": [
                                {
                                    "step_idx": 0,
                                    "ob_len": 50,
                                    "ac_len": 20,
                                    "reward": g * 0.5,
                                    "episode_done": True,
                                    "metrics": {"correct": float(g)},
                                    "logs": {},
                                }
                            ],
                        }
                    )
                )
        (iter_dir / "train_rollout_summaries.jsonl").write_text("\n".join(rollouts) + "\n")
        (iter_dir / "train_logtree.json").write_text(
            json.dumps(
                {
                    "title": f"Iteration {iteration}",
                    "root": {"tag": "div", "children": []},
                }
            )
        )

    return d


class TestTrainingRunStore:
    def test_read_config(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        config = store.read_config()
        assert config is not None
        assert config["model_name"] == "Llama-3.1-8B"

    def test_config_cached(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        c1 = store.read_config()
        c2 = store.read_config()
        assert c1 is c2

    def test_read_metrics(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        metrics = store.read_metrics()
        assert len(metrics) == 5
        assert metrics[0]["step"] == 0

    def test_incremental_metrics(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        store.read_metrics()
        # Append new data
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 5, "new": 1.0}) + "\n")
        new = store.read_new_metrics()
        assert len(new) == 1
        assert new[0]["step"] == 5

    def test_metric_keys(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        store.read_metrics()
        keys = store.metric_keys()
        assert "env/all/reward/total" in keys
        assert "step" not in keys

    def test_read_rollouts(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        rollouts = store.read_rollouts(0)
        assert len(rollouts) == 4
        assert rollouts[0]["group_idx"] == 0

    def test_rollouts_cached(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        r1 = store.read_rollouts(0)
        r2 = store.read_rollouts(0)
        assert r1 is r2

    def test_read_single_rollout(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        r = store.read_single_rollout(0, 1, 0)
        assert r is not None
        assert r["group_idx"] == 1

    def test_read_checkpoints(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        ckpts = store.read_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0]["name"] == "000004"

    def test_read_timing(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        timing = store.read_timing()
        assert len(timing) == 1
        assert timing[0]["step"] == 0

    def test_read_logtree(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        lt = store.read_logtree(0)
        assert lt is not None
        assert lt["title"] == "Iteration 0"

    def test_list_iterations(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        iters = store.list_iterations()
        assert len(iters) == 2
        assert iters[0].iteration == 0
        assert iters[0].has_train_rollouts is True

    def test_pickle_serializable(self, run_dir: Path) -> None:
        store = TrainingRunStore(LocalStorage(run_dir))
        restored = pickle.loads(pickle.dumps(store))
        config = restored.read_config()
        assert config is not None
        assert config["model_name"] == "Llama-3.1-8B"

    def test_nonexistent(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path), "nonexistent")
        assert store.read_config() is None
        assert store.read_metrics() == []


class TestTrainingRunStoreWrites:
    """Tests for TrainingRunStore write methods."""

    def test_write_config(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_config({"model_name": "test-model", "lr": 1e-4})
        config = store.read_config()
        assert config is not None
        assert config["model_name"] == "test-model"

    def test_write_config_updates_cache(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_config({"v": 1})
        assert store.read_config() == {"v": 1}
        store.write_config({"v": 2})
        assert store.read_config() == {"v": 2}

    def test_write_metrics(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_metrics({"loss": 2.5, "reward": 0.1}, step=0)
        store.write_metrics({"loss": 2.0, "reward": 0.3}, step=1)
        metrics = store.read_metrics()
        assert len(metrics) == 2
        assert metrics[0]["step"] == 0
        assert metrics[1]["loss"] == 2.0

    def test_write_metrics_no_step(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_metrics({"lr": 1e-4})
        metrics = store.read_metrics()
        assert len(metrics) == 1
        assert "step" not in metrics[0]
        assert metrics[0]["lr"] == 1e-4

    def test_write_timing_spans(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        spans = [
            {"name": "sampling", "duration": 1.0, "wall_start": 0.0, "wall_end": 1.0},
            {"name": "train", "duration": 0.5, "wall_start": 1.0, "wall_end": 1.5},
        ]
        store.write_timing_spans(step=0, spans=spans)
        timing = store.read_timing()
        assert len(timing) == 1
        assert timing[0]["step"] == 0
        assert len(timing[0]["spans"]) == 2

    def test_write_timing_spans_empty(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_timing_spans(step=0, spans=[])
        # Empty spans should not write anything
        assert store.read_timing() == []

    def test_write_checkpoint(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_checkpoint({"name": "ckpt_0", "batch": 10, "state_path": "tinker:///ckpt/0"})
        store.write_checkpoint({"name": "ckpt_final", "batch": 100, "final": True})
        ckpts = store.read_checkpoints()
        assert len(ckpts) == 2
        assert ckpts[0]["name"] == "ckpt_0"
        assert ckpts[1]["final"] is True

    def test_write_rollouts(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        records = [
            {"group_idx": 0, "traj_idx": 0, "total_reward": 1.0},
            {"group_idx": 0, "traj_idx": 1, "total_reward": 0.5},
        ]
        store.write_rollouts(iteration=0, records=records)
        read = store.read_rollouts(0)
        assert len(read) == 2
        assert read[0]["total_reward"] == 1.0

    def test_write_rollouts_invalidates_cache(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_rollouts(0, [{"v": 1}])
        assert store.read_rollouts(0)[0]["v"] == 1
        store.write_rollouts(0, [{"v": 2}])
        assert store.read_rollouts(0)[0]["v"] == 2

    def test_write_rollouts_eval(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_rollouts(0, [{"score": 0.8}], base_name="eval_gsm8k")
        read = store.read_rollouts(0, base_name="eval_gsm8k")
        assert len(read) == 1

    def test_write_logtree(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        tree = {"title": "Step 0", "root": {"tag": "div", "children": []}}
        store.write_logtree(iteration=0, data=tree)
        read = store.read_logtree(0)
        assert read is not None
        assert read["title"] == "Step 0"

    def test_write_code_diff(self, tmp_path: Path) -> None:
        store = TrainingRunStore(LocalStorage(tmp_path))
        store.write_code_diff("--- a/file.py\n+++ b/file.py\n")
        data = store.storage.read("code.diff")
        assert b"file.py" in data

    def test_roundtrip_write_read(self, tmp_path: Path) -> None:
        """Full roundtrip: write all data types, then read them back."""
        store = TrainingRunStore(LocalStorage(tmp_path))

        store.write_config({"model": "llama", "lr": 1e-5})
        store.write_metrics({"loss": 3.0}, step=0)
        store.write_metrics({"loss": 2.5}, step=1)
        store.write_timing_spans(
            0, [{"name": "sample", "duration": 1.0, "wall_start": 0, "wall_end": 1}]
        )
        store.write_checkpoint({"name": "000001", "batch": 1})
        store.write_rollouts(0, [{"group_idx": 0, "traj_idx": 0, "reward": 1.0}])
        store.write_logtree(0, {"title": "iter0", "root": {}})
        store.write_code_diff("diff content")

        config = store.read_config()
        assert config is not None
        assert config["model"] == "llama"
        assert len(store.read_metrics()) == 2
        assert len(store.read_timing()) == 1
        assert len(store.read_checkpoints()) == 1
        assert len(store.read_rollouts(0)) == 1
        logtree = store.read_logtree(0)
        assert logtree is not None
        assert logtree["title"] == "iter0"


class TestRunRegistry:
    def test_discover_single_run(self, run_dir: Path) -> None:
        registry = RunRegistry([LocalStorage(run_dir)])
        runs = registry.get_runs()
        assert len(runs) == 1
        assert runs[0].has_config is True
        assert runs[0].has_metrics is True
        assert runs[0].training_type == "rl"

    def test_get_training_store(self, run_dir: Path) -> None:
        registry = RunRegistry([LocalStorage(run_dir)])
        run_id = registry.get_runs()[0].run_id
        store = registry.get_training_store(run_id)
        config = store.read_config()
        assert config is not None
        assert config["model_name"] == "Llama-3.1-8B"

    def test_multi_storage(self, tmp_path: Path) -> None:
        # Create two storage roots with different runs
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for root, name in [(root_a, "run_a"), (root_b, "run_b")]:
            d = root / name
            d.mkdir(parents=True)
            (d / "metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n")

        registry = RunRegistry([LocalStorage(root_a), LocalStorage(root_b)])
        runs = registry.get_runs()
        assert len(runs) == 2
        ids = {r.run_id for r in runs}
        assert "run_a" in ids
        assert "run_b" in ids

    def test_refresh_clears_caches(self, run_dir: Path) -> None:
        registry = RunRegistry([LocalStorage(run_dir)])
        store1 = registry.get_training_store(registry.get_runs()[0].run_id)
        registry.refresh()
        store2 = registry.get_training_store(registry.get_runs()[0].run_id)
        assert store1 is not store2

    def test_registry_pickle_preserves_eval_cache(self, tmp_path: Path) -> None:
        """_EVAL_UNSET sentinel survives pickle so eval cache works."""
        (tmp_path / "run" / "metrics.jsonl").parent.mkdir(parents=True)
        (tmp_path / "run" / "metrics.jsonl").write_text('{"step":0}\n')
        registry = RunRegistry([LocalStorage(tmp_path)])
        registry.refresh()
        # Eval store not found (no runs.jsonl)
        assert registry.get_eval_store() is None
        # Pickle round-trip
        restored = pickle.loads(pickle.dumps(registry))
        # After unpickle, should re-probe (cache was reset by pickle)
        assert restored.get_eval_store() is None

    def test_custom_eval_prefix(self, tmp_path: Path) -> None:
        """Registry finds eval data under a custom prefix."""
        eval_dir = tmp_path / "my_evals" / "runs" / "eval_001"
        eval_dir.mkdir(parents=True)
        (eval_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "eval_001",
                    "model_name": "test",
                    "checkpoint_path": None,
                    "checkpoint_name": None,
                    "benchmarks": [],
                    "timestamp": "2024-01-01",
                }
            )
        )
        (tmp_path / "my_evals" / "runs.jsonl").write_text(json.dumps({"run_id": "eval_001"}) + "\n")
        # Default prefixes won't find it
        registry = RunRegistry([LocalStorage(tmp_path)])
        assert registry.get_eval_store() is None
        # Custom prefix finds it
        registry2 = RunRegistry([LocalStorage(tmp_path)], eval_prefixes=("my_evals",))
        store = registry2.get_eval_store()
        assert store is not None


class TestIncrementalReader:
    def test_concurrent_reads(self, tmp_path: Path) -> None:
        """Multiple threads reading concurrently should not corrupt state."""
        storage = LocalStorage(tmp_path)
        # Write 100 records
        lines = [json.dumps({"step": i, "value": i * 0.1}) for i in range(100)]
        storage.write("data.jsonl", ("\n".join(lines) + "\n").encode())

        reader = IncrementalReader(storage, "data.jsonl")
        errors: list[Exception] = []

        def _read():
            try:
                reader.read()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent reads: {errors}"
        # All 100 records should be present (no duplicates, no missing)
        assert len(reader.records) == 100

    def test_pickle_with_lock(self, tmp_path: Path) -> None:
        """IncrementalReader survives pickle (lock excluded and recreated)."""
        storage = LocalStorage(tmp_path)
        storage.write("data.jsonl", b'{"step":0}\n')
        reader = IncrementalReader(storage, "data.jsonl")
        reader.read()
        assert len(reader.records) == 1

        restored = pickle.loads(pickle.dumps(reader))
        assert len(restored.records) == 1
        # Can still read after unpickle (lock recreated)
        restored.read()
