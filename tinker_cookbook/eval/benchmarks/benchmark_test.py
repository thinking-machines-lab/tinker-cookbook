"""Tests for benchmark runner helpers."""

from __future__ import annotations

from tinker_cookbook.eval.benchmarks._runner import _compute_token_turn_summary
from tinker_cookbook.eval.benchmarks._types import Metrics


class TestComputeTokenTurnSummary:
    """Tests for _compute_token_turn_summary."""

    def test_basic(self) -> None:
        metrics_list: list[Metrics] = [
            {"_eval_turns": 2, "_eval_ac_tokens": 100, "_eval_ob_tokens": 50},
            {"_eval_turns": 3, "_eval_ac_tokens": 150, "_eval_ob_tokens": 75},
        ]
        result = _compute_token_turn_summary(metrics_list)
        assert result["total_ac_tokens"] == 250
        assert result["total_ob_tokens"] == 125
        assert result["total_turns"] == 5
        assert result["turns_per_episode"] == 2.5
        assert result["ac_tokens_per_turn"] == 50.0
        assert result["ob_tokens_per_turn"] == 25.0

    def test_empty_list(self) -> None:
        result = _compute_token_turn_summary([])
        assert result["total_ac_tokens"] == 0
        assert result["total_ob_tokens"] == 0
        assert result["total_turns"] == 0
        assert result["turns_per_episode"] == 0
        assert "ac_tokens_per_turn" not in result
        assert "ob_tokens_per_turn" not in result

    def test_missing_keys_default_to_zero(self) -> None:
        metrics_list: list[Metrics] = [
            {"some_other_metric": 1.0},
            {"_eval_turns": 1, "_eval_ac_tokens": 10, "_eval_ob_tokens": 5},
        ]
        result = _compute_token_turn_summary(metrics_list)
        assert result["total_ac_tokens"] == 10
        assert result["total_ob_tokens"] == 5
        assert result["total_turns"] == 1
        assert result["turns_per_episode"] == 0.5

    def test_zero_turns_no_per_turn_stats(self) -> None:
        metrics_list: list[Metrics] = [
            {"_eval_turns": 0, "_eval_ac_tokens": 0, "_eval_ob_tokens": 0},
        ]
        result = _compute_token_turn_summary(metrics_list)
        assert result["total_turns"] == 0
        assert "ac_tokens_per_turn" not in result
        assert "ob_tokens_per_turn" not in result

    def test_single_example(self) -> None:
        metrics_list: list[Metrics] = [
            {"_eval_turns": 1, "_eval_ac_tokens": 200, "_eval_ob_tokens": 100},
        ]
        result = _compute_token_turn_summary(metrics_list)
        assert result["total_ac_tokens"] == 200
        assert result["total_ob_tokens"] == 100
        assert result["total_turns"] == 1
        assert result["turns_per_episode"] == 1.0
        assert result["ac_tokens_per_turn"] == 200.0
        assert result["ob_tokens_per_turn"] == 100.0
