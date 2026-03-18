"""Tests for sweep result collection."""

import json
from pathlib import Path

import pytest

from tinker_cookbook.sweep.results import _extract_config_value, _read_final_metrics, collect


@pytest.fixture
def sweep_dir(tmp_path: Path) -> Path:
    """Create a mock sweep directory with two completed runs."""
    for i, (lr, rank) in enumerate([(1e-4, 32), (3e-4, 64)]):
        run_dir = tmp_path / f"run_{i}"
        run_dir.mkdir()

        # Write config.json
        config = {"learning_rate": lr, "lora_rank": rank, "model_name": "test-model"}
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Write metrics.jsonl (multiple lines, last line has final metrics)
        metrics_lines = [
            {"step": 0, "train_mean_nll": 2.5, "progress": 0.0, "learning_rate": lr},
            {"step": 50, "train_mean_nll": 2.0 - i * 0.3, "progress": 0.5, "learning_rate": lr},
            {
                "step": 100,
                "train_mean_nll": 1.8 - i * 0.3,
                "progress": 1.0,
                "learning_rate": lr,
            },
        ]
        with open(run_dir / "metrics.jsonl", "w") as f:
            for line in metrics_lines:
                f.write(json.dumps(line) + "\n")

    return tmp_path


class TestReadFinalMetrics:
    def test_reads_last_line(self, sweep_dir: Path):
        metrics = _read_final_metrics(sweep_dir / "run_0")
        assert metrics["step"] == 100
        assert metrics["progress"] == 1.0

    def test_missing_file(self, tmp_path: Path):
        metrics = _read_final_metrics(tmp_path / "nonexistent")
        assert metrics == {}

    def test_empty_file(self, tmp_path: Path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").touch()
        metrics = _read_final_metrics(run_dir)
        assert metrics == {}


class TestExtractConfigValue:
    def test_top_level(self):
        config = {"learning_rate": 1e-4, "model_name": "test"}
        assert _extract_config_value(config, "learning_rate") == 1e-4

    def test_nested(self):
        config = {"dataset_builder": {"common_config": {"batch_size": 128}}}
        assert _extract_config_value(config, "dataset_builder.common_config.batch_size") == 128

    def test_missing_key(self):
        config = {"learning_rate": 1e-4}
        assert _extract_config_value(config, "nonexistent") is None

    def test_missing_nested(self):
        config = {"a": {"b": 1}}
        assert _extract_config_value(config, "a.c.d") is None


class TestCollect:
    def test_basic_collect(self, sweep_dir: Path):
        df = collect(str(sweep_dir))
        assert len(df) == 2
        assert "train_mean_nll" in df.columns
        assert "learning_rate" in df.columns
        assert "log_path" in df.columns

    def test_config_keys_filter(self, sweep_dir: Path):
        df = collect(str(sweep_dir), config_keys=["learning_rate", "lora_rank"])
        assert "learning_rate" in df.columns
        assert "lora_rank" in df.columns

    def test_require_complete_filters_incomplete(self, sweep_dir: Path):
        # Add an incomplete run
        run_dir = sweep_dir / "run_incomplete"
        run_dir.mkdir()
        with open(run_dir / "config.json", "w") as f:
            json.dump({"learning_rate": 5e-4}, f)
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"step": 10, "train_mean_nll": 2.0, "progress": 0.1}) + "\n")

        df = collect(str(sweep_dir), require_complete=True)
        assert len(df) == 2  # incomplete run excluded

        df_all = collect(str(sweep_dir), require_complete=False)
        assert len(df_all) == 3  # incomplete run included

    def test_empty_directory(self, tmp_path: Path):
        df = collect(str(tmp_path))
        assert len(df) == 0

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            collect("/nonexistent/path")

    def test_skips_non_directories(self, sweep_dir: Path):
        # Add a file (not a directory) — should be skipped
        (sweep_dir / "README.md").write_text("hello")
        df = collect(str(sweep_dir))
        assert len(df) == 2
