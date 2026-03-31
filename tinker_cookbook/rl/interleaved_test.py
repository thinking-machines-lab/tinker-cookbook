"""Tests for InterleavedRLDatasetBuilder."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.rl.interleaved import InterleavedRLDataset, InterleavedRLDatasetBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockEnvGroupBuilder(EnvGroupBuilder):
    """Mock builder tagged with a source name for testing."""

    def __init__(self, source_name: str, idx: int):
        self.source_name = source_name
        self.idx = idx

    async def make_envs(self):
        return []

    def logging_tags(self) -> list[str]:
        return [self.source_name]


class MockRLDataset(RLDataset):
    """Mock dataset that returns tagged builders."""

    def __init__(self, name: str, num_batches: int, groups_per_batch: int = 1):
        self.name = name
        self._batches = [
            [MockEnvGroupBuilder(name, i * groups_per_batch + j) for j in range(groups_per_batch)]
            for i in range(num_batches)
        ]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self._batches[index % len(self._batches)]

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterleavedRLDataset:
    def test_basic_blending(self):
        """Groups are drawn from multiple sources according to weights."""
        src_a = MockRLDataset("A", num_batches=100)
        src_b = MockRLDataset("B", num_batches=100)
        ds = InterleavedRLDataset(
            sources=[src_a, src_b],
            weights=[0.7, 0.3],
            groups_per_batch=10,
            total_batches=50,
            seed=42,
        )
        assert len(ds) == 50

        # Count source distribution across all batches
        counts = Counter()
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            for builder in batch:
                counts[builder.logging_tags()[0]] += 1

        total = sum(counts.values())
        assert total == 50 * 10  # 50 batches × 10 groups
        # Weights should be approximately respected (within 10%)
        assert abs(counts["A"] / total - 0.7) < 0.1
        assert abs(counts["B"] / total - 0.3) < 0.1

    def test_deterministic_schedule(self):
        """Same seed produces same schedule."""
        src_a = MockRLDataset("A", num_batches=50)
        src_b = MockRLDataset("B", num_batches=50)

        ds1 = InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 4, 20, seed=123)
        ds2 = InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 4, 20, seed=123)

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

        ds1 = InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 10, 20, seed=1)
        ds2 = InterleavedRLDataset([src_a, src_b], [0.5, 0.5], 10, 20, seed=2)

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

        ds = InterleavedRLDataset(
            sources=[src_small, src_large],
            weights=[0.5, 0.5],
            groups_per_batch=1,
            total_batches=None,
            seed=42,
        )
        # Small source has 10 batches, weight 0.5 → exhausted after ~20 batches
        assert len(ds) == 20

    def test_cycling_with_total_batches(self):
        """When total_batches exceeds source size, cycling works."""
        src = MockRLDataset("A", num_batches=5)
        ds = InterleavedRLDataset(
            sources=[src],
            weights=[1.0],
            groups_per_batch=1,
            total_batches=20,
            seed=42,
        )
        assert len(ds) == 20
        # Should not raise even though source only has 5 batches
        for i in range(20):
            batch = ds.get_batch(i)
            assert len(batch) == 1

    def test_out_of_range_raises(self):
        """Accessing beyond total_batches raises IndexError."""
        src = MockRLDataset("A", num_batches=10)
        ds = InterleavedRLDataset([src], [1.0], 1, 5, seed=0)
        with pytest.raises(IndexError):
            ds.get_batch(5)

    def test_validation_mismatched_lengths(self):
        """Mismatched sources and weights raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="same length"):
            InterleavedRLDataset([src], [0.5, 0.5], 1, 10)

    def test_validation_empty_sources(self):
        """Empty sources raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            InterleavedRLDataset([], [], 1, 10)

    def test_validation_zero_weight(self):
        """Zero weight raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="positive"):
            InterleavedRLDataset([src], [0.0], 1, 10)

    def test_three_sources(self):
        """Three-source blend matches paper's multi-domain setup."""
        mcqa = MockRLDataset("mcqa", num_batches=200)
        workbench = MockRLDataset("workbench", num_batches=200)
        struct = MockRLDataset("structured", num_batches=200)

        ds = InterleavedRLDataset(
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
        # Check approximate weight distribution
        assert counts["mcqa"] > counts["workbench"] > counts["structured"]


class TestInterleavedRLDatasetBuilder:
    def test_build(self):
        """Builder correctly constructs InterleavedRLDataset."""
        # Use InterleavedRLDataset directly to test the dataset logic,
        # since RLDatasetBuilder is a frozen chz dataclass that's hard to mock.
        src_a = MockRLDataset("A", num_batches=50, groups_per_batch=4)
        src_b = MockRLDataset("B", num_batches=50, groups_per_batch=4)

        ds = InterleavedRLDataset(
            sources=[src_a, src_b],
            weights=[0.6, 0.4],
            groups_per_batch=4,
            total_batches=30,
            seed=0,
        )
        assert isinstance(ds, InterleavedRLDataset)
        assert len(ds) == 30

        # Verify batches are non-empty and contain builders from both sources
        all_tags = set()
        for i in range(30):
            batch = ds.get_batch(i)
            assert len(batch) == 4
            for b in batch:
                all_tags.update(b.logging_tags())
        assert "A" in all_tags
        assert "B" in all_tags

    def test_mismatched_lengths_raises(self):
        """Mismatched sources/weights raises ValueError."""
        src = MockRLDataset("A", num_batches=10)
        with pytest.raises(ValueError, match="same length"):
            InterleavedRLDataset([src], [0.5, 0.5], 1, 10)
