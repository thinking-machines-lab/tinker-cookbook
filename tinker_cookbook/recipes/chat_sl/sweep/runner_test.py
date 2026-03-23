"""Tests for sweep runner."""

import json
from pathlib import Path

import chz
import pytest

from tinker_cookbook.recipes.chat_sl.sweep.runner import (
    _validate_axes,
    _validate_config_has_log_path,
    run,
)


@chz.chz
class MockConfig:
    log_path: str | None = None
    learning_rate: float = 1e-4
    lora_rank: int = 32
    model_name: str = "test-model"


def _mock_main(config: MockConfig) -> None:
    """A mock training function that writes metrics.jsonl + config.json."""
    assert config.log_path is not None
    log_dir = Path(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write config.json
    with open(log_dir / "config.json", "w") as f:
        json.dump(
            {
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "model_name": config.model_name,
            },
            f,
        )

    # Write metrics.jsonl
    with open(log_dir / "metrics.jsonl", "w") as f:
        loss = 2.0 - config.learning_rate * 1000  # Fake: lower LR = higher loss
        f.write(
            json.dumps(
                {
                    "step": 100,
                    "train_mean_nll": loss,
                    "progress": 1.0,
                    "learning_rate": config.learning_rate,
                }
            )
            + "\n"
        )


class TestValidateAxes:
    def test_valid_axes(self):
        _validate_axes(MockConfig, {"learning_rate": [1e-4], "lora_rank": [32]})

    def test_invalid_axis_raises(self):
        with pytest.raises(TypeError, match="not a field"):
            _validate_axes(MockConfig, {"bogus_param": [1, 2]})

    def test_error_shows_available_fields(self):
        with pytest.raises(TypeError, match="learning_rate"):
            _validate_axes(MockConfig, {"bogus": [1]})


class TestRun:
    def test_basic_sweep(self, tmp_path: Path):
        results = run(
            _mock_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            learning_rate=[1e-4, 3e-4],
        )
        assert len(results) == 2
        assert "train_mean_nll" in results.columns
        assert "learning_rate" in results.columns

    def test_two_axes(self, tmp_path: Path):
        results = run(
            _mock_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            learning_rate=[1e-4, 3e-4],
            lora_rank=[32, 128],
        )
        assert len(results) == 4

    def test_no_axes_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="At least one sweep axis"):
            run(_mock_main, MockConfig(), sweep_dir=str(tmp_path))

    def test_invalid_axis_raises(self, tmp_path: Path):
        with pytest.raises(TypeError, match="not a field"):
            run(
                _mock_main,
                MockConfig(),
                sweep_dir=str(tmp_path),
                bogus=[1, 2],
            )

    def test_non_list_axis_raises(self, tmp_path: Path):
        with pytest.raises(TypeError, match="must be a list"):
            run(
                _mock_main,
                MockConfig(),
                sweep_dir=str(tmp_path),
                learning_rate=1e-4,  # type: ignore[arg-type]
            )

    def test_skip_existing(self, tmp_path: Path):
        # Run once
        run(
            _mock_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            learning_rate=[1e-4, 3e-4],
        )

        # Run again with skip_existing — should not fail
        call_count = 0
        original_main = _mock_main

        def counting_main(config: MockConfig) -> None:
            nonlocal call_count
            call_count += 1
            original_main(config)

        run(
            counting_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            skip_existing=True,
            learning_rate=[1e-4, 3e-4],
        )
        assert call_count == 0  # All skipped

    def test_failed_run_continues(self, tmp_path: Path):
        call_count = 0

        def sometimes_failing_main(config: MockConfig) -> None:
            nonlocal call_count
            call_count += 1
            if config.learning_rate == 1e-4:
                raise RuntimeError("Simulated failure")
            _mock_main(config)

        results = run(
            sometimes_failing_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            learning_rate=[1e-4, 3e-4],
        )
        assert call_count == 2  # Both were attempted
        assert len(results) == 1  # Only the successful run has results

    def test_custom_name_fn(self, tmp_path: Path):
        def short_name(overrides: dict) -> str:
            return f"lr{overrides['learning_rate']}"

        results = run(
            _mock_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            name_fn=short_name,
            learning_rate=[1e-4, 3e-4],
        )
        assert len(results) == 2
        assert (tmp_path / "lr0.0001").is_dir()
        assert (tmp_path / "lr0.0003").is_dir()

    def test_max_parallel(self, tmp_path: Path):
        results = run(
            _mock_main,
            MockConfig(),
            sweep_dir=str(tmp_path),
            max_parallel=2,
            learning_rate=[1e-4, 3e-4, 5e-4],
        )
        assert len(results) == 3


class TestValidateConfigHasLogPath:
    def test_config_with_log_path(self):
        _validate_config_has_log_path(MockConfig)  # Should not raise

    def test_config_without_log_path(self):
        @chz.chz
        class NoLogPathConfig:
            learning_rate: float = 1e-4

        with pytest.raises(TypeError, match="log_path"):
            _validate_config_has_log_path(NoLogPathConfig)

    def test_run_rejects_config_without_log_path(self, tmp_path: Path):
        @chz.chz
        class BadConfig:
            learning_rate: float = 1e-4

        with pytest.raises(TypeError, match="log_path"):
            run(
                lambda c: None,
                BadConfig(),
                sweep_dir=str(tmp_path),
                learning_rate=[1e-4],
            )
