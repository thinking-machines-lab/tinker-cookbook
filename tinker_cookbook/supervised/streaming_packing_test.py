"""Tests for streaming_packing.py — pure packing algorithm and streaming dataset."""

import threading
import time

import pytest
import tinker
import torch

from tinker_cookbook.supervised.streaming_packing import (
    StreamingPackedSupervisedDataset,
    _incremental_pack,
    pack_rendered_examples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rendered(
    n_tokens: int, n_completion: int
) -> tuple[tinker.ModelInput, torch.Tensor]:
    """Create a fake rendered example with *n_tokens* total, last *n_completion* trainable."""
    tokens = list(range(1, n_tokens + 1))
    weights = torch.zeros(n_tokens, dtype=torch.float32)
    weights[-n_completion:] = 1.0
    return tinker.ModelInput.from_ints(tokens), weights


def _simple_render_fn(
    example: dict,
) -> tuple[tinker.ModelInput, torch.Tensor]:
    """A trivial render function for testing: expects {"tokens": [...], "n_completion": int}."""
    tokens = example["tokens"]
    n_completion = example["n_completion"]
    model_input = tinker.ModelInput.from_ints(tokens)
    weights = torch.zeros(len(tokens), dtype=torch.float32)
    weights[-n_completion:] = 1.0
    return model_input, weights


# ---------------------------------------------------------------------------
# Tests for pack_rendered_examples (pure algorithm, relocated from recipe)
# ---------------------------------------------------------------------------


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
        assert len(datums) == 2
        for datum in datums:
            weights = datum.loss_fn_inputs["weights"]
            assert len(weights.data) == 49  # 50 - 1 for right-shift

    def test_packing_ratio(self) -> None:
        """Verify packing efficiency."""
        rendered = [_make_rendered(100, 50) for _ in range(10)]
        max_len = 500
        datums = pack_rendered_examples(rendered, max_packed_length=max_len)
        assert len(datums) == 2
        total_tokens = sum(len(d.loss_fn_inputs["weights"].data) for d in datums)
        assert total_tokens == 998

    def test_accepts_iterator(self) -> None:
        """pack_rendered_examples should accept any iterable, not just lists."""
        rendered = iter([_make_rendered(10, 5), _make_rendered(10, 3)])
        datums = pack_rendered_examples(rendered, max_packed_length=20)
        assert len(datums) == 1

    def test_empty_input(self) -> None:
        """Empty iterable produces no datums."""
        datums = pack_rendered_examples([], max_packed_length=100)
        assert len(datums) == 0

    def test_all_empty_examples(self) -> None:
        """All-empty examples produce no datums."""
        empty = (tinker.ModelInput.from_ints([]), torch.zeros(0, dtype=torch.float32))
        datums = pack_rendered_examples([empty, empty], max_packed_length=100)
        assert len(datums) == 0


# ---------------------------------------------------------------------------
# Tests for _incremental_pack
# ---------------------------------------------------------------------------


class TestIncrementalPack:
    def test_yields_packed_datums(self) -> None:
        rendered = [_make_rendered(10, 5), _make_rendered(10, 3)]
        datums = list(_incremental_pack(iter(rendered), max_packed_length=15))
        assert len(datums) == 2

    def test_carry_over_across_boundary(self) -> None:
        """When examples span pack boundaries, leftover carries to next pack."""
        # 3 examples of 10 tokens each, max 25 -> 2 fit in first pack, 1 in second
        rendered = [_make_rendered(10, 5) for _ in range(3)]
        datums = list(_incremental_pack(iter(rendered), max_packed_length=25))
        assert len(datums) == 2
        # First pack: 20 tokens (2 examples), second pack: 10 tokens (1 example)
        assert len(datums[0].loss_fn_inputs["weights"].data) == 19  # 20 - 1
        assert len(datums[1].loss_fn_inputs["weights"].data) == 9  # 10 - 1

    def test_empty_iterator(self) -> None:
        datums = list(_incremental_pack(iter([]), max_packed_length=100))
        assert len(datums) == 0

    def test_matches_batch_pack(self) -> None:
        """Incremental pack should produce the same result as batch pack."""
        rendered = [_make_rendered(7, 3) for _ in range(15)]
        batch_datums = pack_rendered_examples(rendered, max_packed_length=50)
        incremental_datums = list(_incremental_pack(iter(rendered), max_packed_length=50))
        assert len(batch_datums) == len(incremental_datums)
        for b, i in zip(batch_datums, incremental_datums):
            assert b.loss_fn_inputs["weights"].data == i.loss_fn_inputs["weights"].data


# ---------------------------------------------------------------------------
# Tests for StreamingPackedSupervisedDataset
# ---------------------------------------------------------------------------


class TestStreamingPackedSupervisedDataset:
    def _make_examples(self, n: int, tokens_per: int = 10, completion_per: int = 5) -> list[dict]:
        return [
            {"tokens": list(range(1, tokens_per + 1)), "n_completion": completion_per}
            for _ in range(n)
        ]

    def test_basic_iteration(self) -> None:
        """Dataset produces batches that the training loop can consume sequentially."""
        examples = self._make_examples(20, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=2,
            max_packed_length=50,
            total_examples=20,
        )

        batches = []
        for i in range(len(ds)):
            try:
                batch = ds.get_batch(i)
                batches.append(batch)
            except StopIteration:
                break

        assert len(batches) > 0
        for batch in batches:
            assert len(batch) == 2
            for datum in batch:
                assert "weights" in datum.loss_fn_inputs
                assert "target_tokens" in datum.loss_fn_inputs

    def test_forward_only_access(self) -> None:
        """Backward seek raises ValueError."""
        examples = self._make_examples(20, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=2,
            max_packed_length=50,
            total_examples=20,
        )
        ds.get_batch(0)
        with pytest.raises(ValueError, match="forward iteration"):
            ds.get_batch(0)

    def test_skip_forward(self) -> None:
        """Skipping batches should work (intermediate batches consumed silently)."""
        examples = self._make_examples(100, tokens_per=5)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=50,
            total_examples=100,
        )
        # Skip to batch 2 directly
        batch = ds.get_batch(2)
        assert len(batch) == 1

    def test_exhaustion_raises_stop_iteration(self) -> None:
        """Reading past the end of the dataset raises StopIteration."""
        examples = self._make_examples(5, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=100,
            total_examples=5,
        )
        # Consume all batches
        consumed = 0
        for i in range(100):  # more than enough
            try:
                ds.get_batch(i)
                consumed += 1
            except StopIteration:
                break
        assert consumed > 0
        # One more should also raise
        with pytest.raises(StopIteration):
            ds.get_batch(consumed)

    def test_set_epoch_restarts(self) -> None:
        """set_epoch should allow iterating from the beginning again."""
        examples = self._make_examples(10, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=50,
            total_examples=10,
        )
        # Consume some batches
        batch0_epoch0 = ds.get_batch(0)
        assert len(batch0_epoch0) > 0

        # Reset
        ds.set_epoch(seed=1)

        # Should be able to start from 0 again
        batch0_epoch1 = ds.get_batch(0)
        assert len(batch0_epoch1) > 0

    def test_callable_examples_source(self) -> None:
        """When examples is a callable, it should be called each epoch for a fresh iterator."""
        call_count = 0

        def make_examples():
            nonlocal call_count
            call_count += 1
            return iter(self._make_examples(10, tokens_per=10))

        ds = StreamingPackedSupervisedDataset(
            examples=make_examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=50,
            total_examples=10,
        )
        ds.get_batch(0)
        assert call_count == 1

        ds.set_epoch(seed=1)
        ds.get_batch(0)
        assert call_count == 2

    def test_error_propagation(self) -> None:
        """Errors in the render function propagate to the consumer."""

        def bad_render_fn(example: dict) -> tuple[tinker.ModelInput, torch.Tensor]:
            raise ValueError("render failed")

        examples = self._make_examples(5)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=bad_render_fn,
            batch_size=1,
            max_packed_length=100,
            total_examples=5,
        )
        with pytest.raises(RuntimeError, match="Background rendering thread failed"):
            ds.get_batch(0)

    def test_empty_dataset(self) -> None:
        """Empty dataset should raise StopIteration on first get_batch."""
        ds = StreamingPackedSupervisedDataset(
            examples=[],
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=100,
            total_examples=0,
        )
        with pytest.raises(StopIteration):
            ds.get_batch(0)

    def test_single_example(self) -> None:
        """Dataset with a single example should produce one batch."""
        examples = self._make_examples(1, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=100,
            total_examples=1,
        )
        batch = ds.get_batch(0)
        assert len(batch) == 1

    def test_oversized_examples(self) -> None:
        """Examples exceeding max_packed_length should be truncated, not crash."""
        examples = [{"tokens": list(range(1, 101)), "n_completion": 50}]
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=20,
            total_examples=1,
        )
        batch = ds.get_batch(0)
        assert len(batch) == 1
        # Should be truncated to max_packed_length then right-shifted
        weights = batch[0].loss_fn_inputs["weights"]
        assert len(weights.data) == 19  # 20 - 1

    def test_len_returns_positive_estimate(self) -> None:
        """__len__ should return a reasonable positive estimate."""
        examples = self._make_examples(100, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=2,
            max_packed_length=50,
            total_examples=100,
            avg_example_tokens=10.0,
        )
        length = len(ds)
        assert length > 0

    def test_partial_final_batch(self) -> None:
        """If the last batch has fewer datums than batch_size, it should still be returned."""
        # 3 examples of 10 tokens each, max 25 -> 2 packs
        # With batch_size=3, we cannot fill a full batch, so we get a partial one.
        examples = self._make_examples(3, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=3,
            max_packed_length=25,
            total_examples=3,
        )
        # There should be 2 packed sequences, but batch_size is 3, so we get
        # one partial batch with 2 datums.
        batch = ds.get_batch(0)
        assert len(batch) == 2

    def test_chunk_boundary_carry_over(self) -> None:
        """Verify that examples at chunk boundaries are packed correctly.

        When the last example does not fill a pack, it carries over into the
        next pack.  This test checks that no tokens are lost.
        """
        # 7 examples of 10 tokens, max 30 -> packs of [30, 30, 10]
        examples = self._make_examples(7, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=1,
            max_packed_length=30,
            total_examples=7,
        )
        all_datums: list[tinker.Datum] = []
        for i in range(10):
            try:
                batch = ds.get_batch(i)
                all_datums.extend(batch)
            except StopIteration:
                break

        # 7 * 10 = 70 tokens. Packs of 30, 30, 10. Each loses 1 for right-shift.
        assert len(all_datums) == 3
        total_weight_tokens = sum(
            len(d.loss_fn_inputs["weights"].data) for d in all_datums
        )
        assert total_weight_tokens == 29 + 29 + 9  # 67

    def test_daemon_thread(self) -> None:
        """The background thread should be a daemon thread."""
        examples = self._make_examples(100, tokens_per=10)
        ds = StreamingPackedSupervisedDataset(
            examples=examples,
            render_fn=_simple_render_fn,
            batch_size=2,
            max_packed_length=50,
            total_examples=100,
        )
        assert ds._thread is not None
        assert ds._thread.daemon is True
        ds._stop_background()
