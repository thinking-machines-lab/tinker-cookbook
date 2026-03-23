"""Tests for sweep grid generation."""

import pytest

from tinker_cookbook.recipes.chat_sl.sweep.grid import default_run_name, grid


class TestGrid:
    def test_single_axis(self):
        result = grid(learning_rate=[1e-4, 3e-4])
        assert result == [{"learning_rate": 1e-4}, {"learning_rate": 3e-4}]

    def test_two_axes(self):
        result = grid(learning_rate=[1e-4, 3e-4], lora_rank=[32, 128])
        assert len(result) == 4
        assert {"learning_rate": 1e-4, "lora_rank": 32} in result
        assert {"learning_rate": 1e-4, "lora_rank": 128} in result
        assert {"learning_rate": 3e-4, "lora_rank": 32} in result
        assert {"learning_rate": 3e-4, "lora_rank": 128} in result

    def test_dict_form(self):
        result = grid({"learning_rate": [1e-4], "lora_rank": [32]})
        assert result == [{"learning_rate": 1e-4, "lora_rank": 32}]

    def test_dict_and_kwargs_merged(self):
        result = grid({"learning_rate": [1e-4]}, lora_rank=[32])
        assert result == [{"learning_rate": 1e-4, "lora_rank": 32}]

    def test_empty_grid(self):
        result = grid()
        assert result == [{}]

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            grid(learning_rate=1e-4)  # type: ignore[arg-type]

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one value"):
            grid(learning_rate=[])

    def test_single_value_per_axis(self):
        result = grid(learning_rate=[1e-4], lora_rank=[32])
        assert result == [{"learning_rate": 1e-4, "lora_rank": 32}]

    def test_three_axes(self):
        result = grid(a=[1, 2], b=[3, 4], c=[5, 6])
        assert len(result) == 8

    def test_preserves_order(self):
        result = grid(learning_rate=[1e-4, 3e-4], lora_rank=[32, 128])
        # First axis varies slowest (outer loop)
        assert result[0]["learning_rate"] == 1e-4
        assert result[1]["learning_rate"] == 1e-4
        assert result[2]["learning_rate"] == 3e-4
        assert result[3]["learning_rate"] == 3e-4


class TestDefaultRunName:
    def test_basic(self):
        name = default_run_name({"learning_rate": 3e-4, "lora_rank": 32})
        assert name == "learning_rate=0.0003_lora_rank=32"

    def test_integer_values(self):
        name = default_run_name({"batch_size": 128})
        assert name == "batch_size=128"

    def test_string_values(self):
        name = default_run_name({"dataset": "tulu3"})
        assert name == "dataset=tulu3"

    def test_scientific_notation(self):
        name = default_run_name({"lr": 1e-5})
        assert name == "lr=1e-05"

    def test_empty(self):
        name = default_run_name({})
        assert name == ""
