"""Tests for sequence packing utilities in sft_datasets.py."""

import numpy as np
import tinker
import torch

from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import (
    PackedSupervisedDataset,
    pack_rendered_examples,
)


def _make_rendered(
    n_tokens: int, n_completion: int
) -> tuple[tinker.ModelInput, torch.Tensor]:
    """Create a fake rendered example with *n_tokens* total, last *n_completion* trainable."""
    tokens = list(range(1, n_tokens + 1))
    weights = torch.zeros(n_tokens, dtype=torch.float32)
    weights[-n_completion:] = 1.0
    return tinker.ModelInput.from_ints(tokens), weights


class TestPackRenderedExamples:
    def test_single_example_packs_correctly(self) -> None:
        rendered = [_make_rendered(10, 5)]
        datums = pack_rendered_examples(rendered, max_packed_length=100)
        assert len(datums) == 1
        datum = datums[0]
        # After right-shift: input has 9 tokens, targets/weights have 9 tokens.
        weights = datum.loss_fn_inputs["weights"]
        targets = datum.loss_fn_inputs["target_tokens"]
        assert len(weights.data) == 9
        assert len(targets.data) == 9

    def test_two_examples_fit_in_one_pack(self) -> None:
        rendered = [_make_rendered(10, 5), _make_rendered(10, 3)]
        datums = pack_rendered_examples(rendered, max_packed_length=20)
        assert len(datums) == 1
        datum = datums[0]
        # 20 total tokens -> 19 after right-shift.
        weights = datum.loss_fn_inputs["weights"]
        assert len(weights.data) == 19

    def test_examples_split_across_packs(self) -> None:
        rendered = [_make_rendered(10, 5), _make_rendered(10, 3)]
        datums = pack_rendered_examples(rendered, max_packed_length=15)
        assert len(datums) == 2

    def test_exact_fit(self) -> None:
        rendered = [_make_rendered(8, 4), _make_rendered(8, 4)]
        datums = pack_rendered_examples(rendered, max_packed_length=16)
        assert len(datums) == 1
        weights = datums[0].loss_fn_inputs["weights"]
        assert len(weights.data) == 15  # 16 - 1 for right-shift

    def test_oversized_example_truncated(self) -> None:
        rendered = [_make_rendered(20, 10)]
        datums = pack_rendered_examples(rendered, max_packed_length=10)
        assert len(datums) == 1
        weights = datums[0].loss_fn_inputs["weights"]
        assert len(weights.data) == 9  # truncated to 10, then right-shift -> 9

    def test_empty_examples_skipped(self) -> None:
        empty = (tinker.ModelInput.from_ints([]), torch.zeros(0, dtype=torch.float32))
        rendered = [empty, _make_rendered(5, 3), empty]
        datums = pack_rendered_examples(rendered, max_packed_length=100)
        assert len(datums) == 1

    def test_weights_preserve_boundaries(self) -> None:
        """Weights from each example are concatenated, preserving per-token masking."""
        # Example 1: [0, 0, 1, 1] (2 prompt + 2 completion)
        # Example 2: [0, 1, 1, 1] (1 prompt + 3 completion)
        ex1_input = tinker.ModelInput.from_ints([10, 20, 30, 40])
        ex1_weights = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        ex2_input = tinker.ModelInput.from_ints([50, 60, 70, 80])
        ex2_weights = torch.tensor([0, 1, 1, 1], dtype=torch.float32)

        datums = pack_rendered_examples(
            [(ex1_input, ex1_weights), (ex2_input, ex2_weights)],
            max_packed_length=8,
        )
        assert len(datums) == 1
        # Combined pre-shift weights: [0, 0, 1, 1, 0, 1, 1, 1]
        # After right-shift (drop first): [0, 1, 1, 0, 1, 1, 1]
        weights_data = datums[0].loss_fn_inputs["weights"].data
        assert weights_data == [0, 1, 1, 0, 1, 1, 1]

    def test_many_small_examples(self) -> None:
        """Many small examples should be efficiently packed."""
        rendered = [_make_rendered(5, 3) for _ in range(20)]
        datums = pack_rendered_examples(rendered, max_packed_length=50)
        # 20 * 5 = 100 tokens, max 50 per pack -> at least 2 packs
        assert len(datums) == 2
        # Each pack should contain 10 examples (10 * 5 = 50 tokens).
        for datum in datums:
            weights = datum.loss_fn_inputs["weights"]
            assert len(weights.data) == 49  # 50 - 1 for right-shift

    def test_packing_ratio(self) -> None:
        """Verify packing efficiency: short examples waste less space when packed."""
        rendered = [_make_rendered(100, 50) for _ in range(10)]
        max_len = 500

        # Without packing: 10 separate datums, each 100 tokens out of 500 -> 20% usage.
        # With packing: ~2 datums, each 500 tokens -> 100% usage.
        datums = pack_rendered_examples(rendered, max_packed_length=max_len)
        assert len(datums) == 2
        total_tokens = sum(len(d.loss_fn_inputs["weights"].data) for d in datums)
        # 2 packs of 500 tokens each -> 998 tokens after shift (500-1 each = 499 * 2).
        assert total_tokens == 998
