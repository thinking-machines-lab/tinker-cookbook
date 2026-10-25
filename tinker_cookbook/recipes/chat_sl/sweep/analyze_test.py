"""Tests for the markdown generation in sweep/analyze.py (no W&B needed)."""

from typing import Any

from tinker_cookbook.recipes.chat_sl.sweep.analyze import RunResult, generate_model_section


def _run(**overrides: Any) -> RunResult:
    base: dict[str, Any] = {
        "model": "Qwen-Qwen3-8B",
        "rank": 32,
        "lr": 3e-4,
        "test_nll": 0.5,
        "train_nll": 0.4,
        "wall_time_min": 10.0,
        "run_name": "tulu3-Qwen-Qwen3-8B-32rank-0.0003lr-128batch-local",
    }
    base.update(overrides)
    return RunResult(**base)


def test_model_section_includes_bpb_columns_when_present():
    section = generate_model_section("Qwen-Qwen3-8B", [_run(test_bpb=0.8, train_bpb=0.7)], None)
    assert "Test BPB" in section
    assert "Train BPB" in section
    assert "0.8000" in section  # test_bpb rendered
    assert "0.7000" in section  # train_bpb rendered


def test_model_section_omits_bpb_columns_for_legacy_runs():
    section = generate_model_section("Qwen-Qwen3-8B", [_run()], None)
    assert "BPB" not in section
    assert "| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |" in section


def test_model_section_shows_dash_for_missing_bpb_in_mixed_grid():
    # One run has BPB, another doesn't -> columns shown, missing cell is an em dash.
    runs = [_run(rank=32, test_bpb=0.8, train_bpb=0.7), _run(rank=64)]
    section = generate_model_section("Qwen-Qwen3-8B", runs, None)
    assert "Test BPB" in section
    assert "—" in section  # the rank=64 run has no BPB
