"""Tests for the packing algorithm (packing.py)."""

from __future__ import annotations

import tinker

from tinker_cookbook.supervised.packing import greedy_pack, make_datum, pack_to_datums


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(token_ids: list[int], weight_val: float = 1.0) -> tuple[list[int], list[float]]:
    """Create a (tokens, weights) pair for testing."""
    return (token_ids, [weight_val] * len(token_ids))


# ---------------------------------------------------------------------------
# greedy_pack
# ---------------------------------------------------------------------------


class TestGreedyPack:
    def test_empty_input(self):
        result = greedy_pack([], max_length=100)
        assert result == []

    def test_single_item_fits(self):
        items = [_make_item([1, 2, 3, 4, 5])]
        result = greedy_pack(items, max_length=100)
        assert len(result) == 1
        assert result[0][0] == [1, 2, 3, 4, 5]

    def test_two_items_fit_in_one_bin(self):
        items = [
            _make_item([1, 2, 3]),
            _make_item([4, 5, 6]),
        ]
        result = greedy_pack(items, max_length=100)
        assert len(result) == 1
        assert result[0][0] == [1, 2, 3, 4, 5, 6]

    def test_two_items_overflow_into_two_bins(self):
        items = [
            _make_item([1, 2, 3]),
            _make_item([4, 5, 6]),
        ]
        # Max 4 tokens per bin: first item (3) fits, second (3) overflows.
        result = greedy_pack(items, max_length=4)
        assert len(result) == 2
        assert result[0][0] == [1, 2, 3]
        assert result[1][0] == [4, 5, 6]

    def test_oversized_item_packed_alone_and_truncated(self):
        items = [
            _make_item([1, 2]),
            _make_item(list(range(100))),  # oversized
            _make_item([3, 4]),
        ]
        result = greedy_pack(items, max_length=10)
        assert len(result) == 3
        # The oversized item is truncated to max_length.
        assert len(result[1][0]) == 10

    def test_skips_empty_items(self):
        items = [
            _make_item([]),
            _make_item([1, 2, 3]),
            _make_item([]),
        ]
        result = greedy_pack(items, max_length=100)
        assert len(result) == 1

    def test_weights_concatenated_correctly(self):
        items = [
            ([10, 20], [0.5, 0.5]),
            ([30, 40], [1.0, 1.0]),
        ]
        result = greedy_pack(items, max_length=100)
        assert len(result) == 1
        assert result[0][0] == [10, 20, 30, 40]
        assert result[0][1] == [0.5, 0.5, 1.0, 1.0]

    def test_exact_fit(self):
        items = [
            _make_item([1, 2, 3, 4, 5]),
            _make_item([6, 7, 8, 9, 10]),
        ]
        result = greedy_pack(items, max_length=10)
        assert len(result) == 1
        assert len(result[0][0]) == 10

    def test_accepts_iterator(self):
        """greedy_pack works with any Iterable, not just lists."""

        def gen():
            yield _make_item([1, 2])
            yield _make_item([3, 4])

        result = greedy_pack(gen(), max_length=100)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# make_datum
# ---------------------------------------------------------------------------


class TestMakeDatum:
    def test_returns_datum(self):
        datum = make_datum([1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], max_length=100)
        assert isinstance(datum, tinker.Datum)
        assert datum.model_input.length > 0

    def test_truncation(self):
        tokens = list(range(20))
        weights = [1.0] * 20
        datum = make_datum(tokens, weights, max_length=10)
        # After make_datum, the model_input length should be <= max_length.
        # (datum_from_model_input_weights does input/target split, so actual
        # model_input length is max_length - 1.)
        assert datum.model_input.length <= 10


# ---------------------------------------------------------------------------
# pack_to_datums
# ---------------------------------------------------------------------------


class TestPackToDatums:
    def test_convenience_wrapper(self):
        items = [
            _make_item([10, 20, 30, 40]),
            _make_item([50, 60, 70, 80]),
        ]
        datums = pack_to_datums(items, max_length=100)
        assert len(datums) >= 1
        assert all(isinstance(d, tinker.Datum) for d in datums)

    def test_empty_input(self):
        assert pack_to_datums([], max_length=100) == []

    def test_multiple_bins(self):
        items = [_make_item([i, i + 1]) for i in range(10)]
        datums = pack_to_datums(items, max_length=5)
        # Each item is 2 tokens. At max 5, we fit 2 items per bin (4 tokens).
        # 10 items -> 5 bins.
        assert len(datums) == 5
