"""Tests for _InterleavedRLDataset and InterleavedRLDatasetBuilder."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import cast

import pytest

from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder, _InterleavedRLDataset
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockEnvGroupBuilder(EnvGroupBuilder):
    """Mock builder tagged with a source name and unique id for testing."""

    def __init__(self, source_name: str, idx: int):
        self.source_name = source_name
        self.idx = idx

    async def make_envs(self):
        return []

    def logging_tags(self) -> list[str]:
        return [self.source_name]

    def __repr__(self) -> str:
        return f"MockEnvGroupBuilder({self.source_name!r}, {self.idx})"


class MockRLDataset(RLDataset):
    """Mock dataset that returns tagged builders.

    Each batch contains ``groups_per_batch`` uniquely-indexed builders.
    Builder indices are globally unique within the dataset: batch i
    contains builders with indices ``[i*gpb, i*gpb+1, ..., i*gpb+gpb-1]``.
    """

    def __init__(self, name: str, num_batches: int, groups_per_batch: int = 1):
        self.name = name
        self._num_batches = num_batches
        self._groups_per_batch = groups_per_batch
        self._batches = [
            [MockEnvGroupBuilder(name, i * groups_per_batch + j) for j in range(groups_per_batch)]
            for i in range(num_batches)
        ]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self._batches[index % len(self._batches)]

    def __len__(self) -> int:
        return self._num_batches


class EmptyBatchDataset(RLDataset):
    """Dataset whose get_batch returns an empty list (for error testing)."""

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return []

    def __len__(self) -> int:
        return 5


class EmptyDataset(RLDataset):
    """Dataset with 0 batches (for error testing)."""

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return []

    def __len__(self) -> int:
        return 0


class RaggedDataset(RLDataset):
    """Dataset where batch 0 has 4 groups but later batches have fewer."""

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index == 0:
            return [MockEnvGroupBuilder("ragged", i) for i in range(4)]
        # Simulate a partial/filtered batch
        return [MockEnvGroupBuilder("ragged", 100)]

    def __len__(self) -> int:
        return 5


class RaggedLastBatchDataset(RLDataset):
    """Dataset where every batch has ``groups_per_batch`` groups except
    the last one, which has fewer — the legitimate ragged-tail case
    that occurs when the total problem count isn't evenly divisible.

    Builder indices are globally unique so tests can verify coverage.
    """

    def __init__(
        self, name: str, num_batches: int, groups_per_batch: int, last_batch_groups: int
    ):
        self.name = name
        self._num_batches = num_batches
        self._gpb = groups_per_batch
        self._last_gpb = last_batch_groups
        assert 0 < last_batch_groups < groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index == self._num_batches - 1:
            gpb = self._last_gpb
        else:
            gpb = self._gpb
        base = index * self._gpb
        return [MockEnvGroupBuilder(self.name, base + j) for j in range(gpb)]

    def __len__(self) -> int:
        return self._num_batches

    @property
    def total_groups(self) -> int:
        return (self._num_batches - 1) * self._gpb + self._last_gpb


# ---------------------------------------------------------------------------
# _InterleavedRLDataset tests
# ---------------------------------------------------------------------------


class Test_InterleavedRLDataset:
    def test_basic_blending(self):
        """Groups are drawn from multiple sources according to weights."""
        src_a = MockRLDataset("A", num_batches=100)
        src_b = MockRLDataset("B", num_batches=100)
        ds = _InterleavedRLDataset(
            sources=[src_a, src_b],
            weights=[0.7, 0.3],
            groups_per_batch=10,
            total_batches=50,
            seed=42,
        )
        assert len(ds) == 50

        counts = Counter()
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            assert len(batch) == 10
            for builder in batch:
                counts[builder.logging_tags()[0]] += 1

        total = sum(counts.values())
        assert total == 50 * 10
        assert abs(counts["A"] / total - 0.7) < 0.1
        assert abs(counts["B"] / total - 0.3) < 0.1

    def test_deterministic_schedule(self):
        """Same seed produces same schedule."""
        src_a = MockRLDataset("A", num_batches=50)
        src_b = MockRLDataset("B", num_batches=50)

        ds1 = _InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 4, 20, seed=123)
        ds2 = _InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 4, 20, seed=123)

        for i in range(20):
            batch1 = ds1.get_batch(i)
            batch2 = ds2.get_batch(i)
            tags1 = [b.logging_tags()[0] for b in batch1]
            tags2 = [b.logging_tags()[0] for b in batch2]
            assert tags1 == tags2, f"Batch {i} differs"

    def test_different_seed_different_schedule(self):
        """Different seeds produce different schedules."""
        src_a = MockRLDataset("A", num_batches=50)
        src_b = MockRLDataset("B", num_batches=50)

        ds1 = _InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 10, 20, seed=1)
        ds2 = _InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 10, 20, seed=2)

        any_different = False
        for i in range(20):
            tags1 = [b.logging_tags()[0] for b in ds1.get_batch(i)]
            tags2 = [b.logging_tags()[0] for b in ds2.get_batch(i)]
            if tags1 != tags2:
                any_different = True
                break
        assert any_different

    def test_natural_length_smallest_exhaustion(self):
        """Without total_batches, stops when smallest source is exhausted."""
        src_small = MockRLDataset("small", num_batches=10)
        src_large = MockRLDataset("large", num_batches=100)

        ds = _InterleavedRLDataset(
            sources=[src_small, src_large],
            weights=[0.5, 0.5],
            groups_per_batch=1,
            total_batches=None,
            seed=42,
        )
        # Small source has 10 groups, weight 0.5 → exhausted after ~20 batches
        assert len(ds) == 20

    def test_cycling_with_total_batches(self):
        """When total_batches exceeds source size, cycling works."""
        src = MockRLDataset("A", num_batches=5)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=20,
            seed=42,
        )
        assert len(ds) == 20
        for i in range(20):
            batch = ds.get_batch(i)
            assert len(batch) == 1

    def test_out_of_range_raises(self):
        """Accessing beyond total_batches raises IndexError."""
        src = MockRLDataset("A", num_batches=10)
        ds = _InterleavedRLDataset([src], [1.0], 1, 5, seed=0)
        with pytest.raises(IndexError):
            ds.get_batch(5)

    def test_negative_index_raises(self):
        """Negative batch index raises IndexError."""
        src = MockRLDataset("A", num_batches=10)
        ds = _InterleavedRLDataset([src], [1.0], 1, 5, seed=0)
        with pytest.raises(IndexError):
            ds.get_batch(-1)

    def test_validation_mismatched_lengths(self):
        """Mismatched sources and weights raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="same length"):
            _InterleavedRLDataset([src], [0.5, 0.5], 1, 10)

    def test_validation_empty_sources(self):
        """Empty sources raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            _InterleavedRLDataset([], [], 1, 10)

    def test_validation_zero_weight(self):
        """Zero weight raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="positive"):
            _InterleavedRLDataset([src], [0.0], 1, 10)

    def test_validation_empty_source_dataset(self):
        """Source with 0 batches raises ValueError."""
        with pytest.raises(ValueError, match="0 batches"):
            _InterleavedRLDataset([EmptyDataset()], [1.0], 1, 10)

    def test_validation_empty_batch(self):
        """Source that returns 0 groups per batch raises ValueError."""
        with pytest.raises(ValueError, match="0 groups"):
            _InterleavedRLDataset([EmptyBatchDataset()], [1.0], 1, 10)

    def test_ragged_batch_error(self):
        """Source with variable-length batches raises clear IndexError."""
        ds = _InterleavedRLDataset(
            sources=[RaggedDataset()],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=20,  # enough to hit a non-zero batch
            seed=0,
        )
        # Batch 0 is fine (4 groups), but later batches only have 1 group.
        # When the schedule tries to access within_idx >= 1 on those batches,
        # it should raise with a clear message.
        with pytest.raises(IndexError, match="same number of groups"):
            for i in range(20):
                ds.get_batch(i)

    def test_three_sources(self):
        """Three-source blend matches paper's multi-domain setup."""
        mcqa = MockRLDataset("mcqa", num_batches=200)
        workbench = MockRLDataset("workbench", num_batches=200)
        struct = MockRLDataset("structured", num_batches=200)

        ds = _InterleavedRLDataset(
            sources=[mcqa, workbench, struct],
            weights=[0.55, 0.30, 0.15],
            groups_per_batch=8,
            total_batches=70,
            seed=42,
        )

        counts = Counter()
        for i in range(70):
            for b in ds.get_batch(i):
                counts[b.logging_tags()[0]] += 1

        total = sum(counts.values())
        assert total == 70 * 8
        assert counts["mcqa"] > counts["workbench"] > counts["structured"]

    def test_multi_group_sources_use_all_groups(self):
        """Sources with groups_per_batch > 1 have ALL groups accessible."""
        # Source with 5 batches × 4 groups = 20 total groups
        src = MockRLDataset("A", num_batches=5, groups_per_batch=4)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=20,
            seed=42,
        )

        seen_indices = set()
        for i in range(20):
            batch = ds.get_batch(i)
            for builder in batch:
                seen_indices.add(cast(MockEnvGroupBuilder, builder).idx)

        # All 20 groups should be reachable (indices 0-19)
        assert len(seen_indices) == 20, (
            f"Expected 20 unique groups, got {len(seen_indices)}: {sorted(seen_indices)}"
        )

    def test_multi_group_sources_no_data_waste(self):
        """Verifies groups beyond index [0] in source batches are used."""
        # 3 batches × 3 groups = 9 total groups with indices 0..8
        src = MockRLDataset("A", num_batches=3, groups_per_batch=3)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=9,
            seed=0,
        )

        seen = set()
        for i in range(9):
            for builder in ds.get_batch(i):
                seen.add(cast(MockEnvGroupBuilder, builder).idx)

        # Should see all 9 groups, not just indices 0, 3, 6 (first of each batch)
        assert seen == set(range(9))

    def test_cycle_reshuffles_order(self):
        """When cycling, the second pass sees problems in a different order."""
        # 5 groups total, request 10 batches of 1 → must cycle once
        src = MockRLDataset("A", num_batches=5, groups_per_batch=1)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=10,
            seed=42,
        )

        # Collect group indices for cycle 1 (batches 0-4) and cycle 2 (batches 5-9)
        cycle1 = [cast(MockEnvGroupBuilder, ds.get_batch(i)[0]).idx for i in range(5)]
        cycle2 = [cast(MockEnvGroupBuilder, ds.get_batch(i)[0]).idx for i in range(5, 10)]

        # Both cycles should cover all 5 groups
        assert set(cycle1) == set(range(5))
        assert set(cycle2) == set(range(5))
        # But in a different order
        assert cycle1 != cycle2

    def test_sources_with_different_batch_sizes(self):
        """Sources with different groups_per_batch work correctly."""
        src_a = MockRLDataset("A", num_batches=10, groups_per_batch=2)  # 20 groups
        src_b = MockRLDataset("B", num_batches=5, groups_per_batch=8)  # 40 groups

        ds = _InterleavedRLDataset(
            sources=[src_a, src_b],
            weights=[0.5, 0.5],
            groups_per_batch=4,
            total_batches=10,
            seed=42,
        )

        # Every batch should have exactly 4 builders
        for i in range(10):
            batch = ds.get_batch(i)
            assert len(batch) == 4

        # Both sources should be represented
        all_tags = set()
        for i in range(10):
            for b in ds.get_batch(i):
                all_tags.update(b.logging_tags())
        assert all_tags == {"A", "B"}

    # -----------------------------------------------------------------------
    # Ragged last-batch sampling simulation
    # -----------------------------------------------------------------------

    def test_ragged_last_batch_single_source_completes(self):
        """A single ragged source can be iterated to completion without IndexError."""
        # 158 batches × 64 groups, last batch has 17 groups → 10065 total
        src = RaggedLastBatchDataset("mcqa", num_batches=158, groups_per_batch=64, last_batch_groups=17)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=8,
            total_batches=200,
            seed=42,
        )
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            assert len(batch) == 8

    def test_ragged_last_batch_multi_source_completes(self):
        """Multiple ragged sources blend without IndexError."""
        src_a = RaggedLastBatchDataset("A", num_batches=50, groups_per_batch=8, last_batch_groups=3)
        src_b = RaggedLastBatchDataset("B", num_batches=30, groups_per_batch=16, last_batch_groups=7)
        src_c = MockRLDataset("C", num_batches=40, groups_per_batch=4)  # uniform for contrast

        ds = _InterleavedRLDataset(
            sources=[src_a, src_b, src_c],
            weights=[0.5, 0.3, 0.2],
            groups_per_batch=6,
            total_batches=100,
            seed=99,
        )
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            assert len(batch) == 6

    def test_ragged_last_batch_total_groups_correct(self):
        """The dataset computes the correct total group count for ragged sources."""
        src = RaggedLastBatchDataset("X", num_batches=10, groups_per_batch=4, last_batch_groups=2)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=1,
            seed=0,
        )
        # 9 full batches × 4 + 1 last batch × 2 = 38
        assert ds._source_total_groups[0] == 38

    def test_ragged_last_batch_all_groups_reachable(self):
        """Every group in a ragged source can be reached (no wasted data)."""
        src = RaggedLastBatchDataset("X", num_batches=5, groups_per_batch=4, last_batch_groups=2)
        # total = 4*4 + 2 = 18 groups
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=18,
            seed=0,
        )
        seen = set()
        for i in range(len(ds)):
            for b in ds.get_batch(i):
                seen.add(cast(MockEnvGroupBuilder, b).idx)
        assert seen == set(range(18)), f"Missing groups: {set(range(18)) - seen}"

    def test_ragged_last_batch_natural_length(self):
        """Natural length computation accounts for ragged last batch."""
        # 10 batches × 4 groups, last has 2 → 38 total
        src = RaggedLastBatchDataset("X", num_batches=10, groups_per_batch=4, last_batch_groups=2)
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=None,
            seed=0,
        )
        # weight=1.0, gpb=1 → 1 group/batch → 38 batches
        assert len(ds) == 38

    def test_ragged_last_batch_cycling_completes(self):
        """Ragged source cycles correctly when total_batches exceeds one pass."""
        src = RaggedLastBatchDataset("X", num_batches=4, groups_per_batch=8, last_batch_groups=3)
        # total = 3*8 + 3 = 27 groups, request 60 batches → needs >2 full cycles
        ds = _InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=60,
            seed=7,
        )
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            assert len(batch) == 1

    @pytest.mark.parametrize(
        "num_batches, gpb, last_gpb, seeds",
        [
            (3, 4, 1, range(5)),  # minimal: 9 groups
            (10, 8, 5, range(5)),  # moderate
            (158, 64, 17, [0, 42]),  # real-world Nemotron scale
            (1, 10, 10, [0]),  # single batch (not actually ragged, but edge case)
        ],
        ids=["minimal", "moderate", "nemotron-scale", "single-batch"],
    )
    def test_ragged_last_batch_parametrized(self, num_batches, gpb, last_gpb, seeds):
        """Parametrized sampling simulation across configs and seeds."""
        for seed in seeds:
            if last_gpb == gpb:
                src = MockRLDataset("P", num_batches=num_batches, groups_per_batch=gpb)
            else:
                src = RaggedLastBatchDataset("P", num_batches=num_batches, groups_per_batch=gpb, last_batch_groups=last_gpb)
            ds = _InterleavedRLDataset(
                sources=[src],
                weights=[1.0],
                groups_per_batch=4,
                total_batches=50,
                seed=seed,
            )
            for i in range(len(ds)):
                batch = ds.get_batch(i)
                assert len(batch) == 4


# ---------------------------------------------------------------------------
# InterleavedRLDatasetBuilder tests
# ---------------------------------------------------------------------------


class TestInterleavedRLDatasetBuilder:
    def test_builder_call(self):
        """Builder's async __call__ correctly wires up sources."""
        import chz

        @chz.chz
        class MockRLDatasetBuilder(RLDatasetBuilder):
            name: str
            num_batches: int
            groups_per_batch: int = 4

            async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
                train = MockRLDataset(self.name, self.num_batches, self.groups_per_batch)
                test = MockRLDataset(self.name + "_test", 5, self.groups_per_batch)
                return train, test

        builder = InterleavedRLDatasetBuilder(
            sources=[
                MockRLDatasetBuilder(name="A", num_batches=50),
                MockRLDatasetBuilder(name="B", num_batches=50),
            ],
            weights=[0.6, 0.4],
            groups_per_batch=4,
            total_batches=20,
            seed=42,
        )

        train_ds, test_ds = asyncio.run(builder())

        assert isinstance(train_ds, _InterleavedRLDataset)
        assert len(train_ds) == 20
        assert test_ds is None

        all_tags = set()
        for i in range(20):
            batch = train_ds.get_batch(i)
            assert len(batch) == 4
            for b in batch:
                all_tags.update(b.logging_tags())
        assert "A" in all_tags
        assert "B" in all_tags

    def test_mismatched_lengths_raises(self):
        """Mismatched sources/weights raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="same length"):
            _InterleavedRLDataset([src], [0.5, 0.5], 1, 10)
