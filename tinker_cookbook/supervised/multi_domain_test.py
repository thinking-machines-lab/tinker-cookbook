"""Tests for multi-domain supervised dataset loader."""

import collections

import datasets
import numpy as np
import pytest
import tinker

from tinker_cookbook.supervised.multi_domain import (
    DomainConfig,
    DomainMixer,
    MultiDomainSupervisedDataset,
)


def _make_dataset(n: int, prefix: str = "row") -> datasets.Dataset:
    """Create a simple in-memory HuggingFace dataset with n rows."""
    return datasets.Dataset.from_dict(
        {"text": [f"{prefix}_{i}" for i in range(n)], "id": list(range(n))}
    )


def _dummy_render_fn(row: dict) -> tinker.Datum:
    """Minimal render function that creates a Datum from a text field."""
    tokens = [ord(c) for c in row["text"][:10]]  # Simple token encoding
    model_input = tinker.ModelInput.from_ints(tokens)
    weights = np.ones(len(tokens), dtype=np.float32)
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={"weights": tinker.TensorData.from_numpy(weights)},
    )


class TestDeterminism:
    """Same seed + index = same batch, regardless of access pattern."""

    def test_same_seed_same_batch(self) -> None:
        domains = [
            DomainConfig("a", 1.0, _make_dataset(100, "a")),
            DomainConfig("b", 1.0, _make_dataset(100, "b")),
        ]
        ds1 = MultiDomainSupervisedDataset(domains, batch_size=4, render_fn=_dummy_render_fn, seed=42)
        ds2 = MultiDomainSupervisedDataset(domains, batch_size=4, render_fn=_dummy_render_fn, seed=42)

        for i in [0, 5, 10, 25]:
            batch1 = ds1.get_batch(i)
            batch2 = ds2.get_batch(i)
            assert len(batch1) == len(batch2)
            for d1, d2 in zip(batch1, batch2):
                assert d1.model_input == d2.model_input

    def test_different_seed_different_batch(self) -> None:
        domains = [
            DomainConfig("a", 1.0, _make_dataset(200, "a")),
            DomainConfig("b", 1.0, _make_dataset(200, "b")),
        ]
        ds1 = MultiDomainSupervisedDataset(domains, batch_size=4, render_fn=_dummy_render_fn, seed=42)
        ds2 = MultiDomainSupervisedDataset(domains, batch_size=4, render_fn=_dummy_render_fn, seed=99)

        # At least some batches should differ
        differences = 0
        for i in range(10):
            b1 = ds1.get_batch(i)
            b2 = ds2.get_batch(i)
            for d1, d2 in zip(b1, b2):
                if d1.model_input != d2.model_input:
                    differences += 1
        assert differences > 0, "Different seeds should produce different batches"


class TestResume:
    """get_batch(50) gives same result whether called directly or after getting 0-49."""

    def test_random_access_matches_sequential(self) -> None:
        domains = [
            DomainConfig("a", 2.0, _make_dataset(500, "a")),
            DomainConfig("b", 1.0, _make_dataset(500, "b")),
        ]

        # Sequential access
        ds_seq = MultiDomainSupervisedDataset(
            domains, batch_size=4, render_fn=_dummy_render_fn, seed=42
        )
        for i in range(50):
            ds_seq.get_batch(i)
        batch_seq = ds_seq.get_batch(50)

        # Direct access (simulating resume)
        ds_direct = MultiDomainSupervisedDataset(
            domains, batch_size=4, render_fn=_dummy_render_fn, seed=42
        )
        batch_direct = ds_direct.get_batch(50)

        assert len(batch_seq) == len(batch_direct)
        for d1, d2 in zip(batch_seq, batch_direct):
            assert d1.model_input == d2.model_input


class TestDomainDistribution:
    """Over many batches, domain frequencies should match configured weights."""

    def test_weight_distribution(self) -> None:
        domains = [
            DomainConfig("heavy", 3.0, _make_dataset(5000, "heavy")),
            DomainConfig("light", 1.0, _make_dataset(5000, "light")),
        ]

        mixer = DomainMixer(domains, seed=42)
        positions = mixer.get_positions(0, 4000)

        counts = collections.Counter(domain_idx for domain_idx, _ in positions)
        total = sum(counts.values())
        heavy_frac = counts[0] / total
        light_frac = counts[1] / total

        # Expected: 0.75 heavy, 0.25 light. Allow 5% tolerance.
        assert abs(heavy_frac - 0.75) < 0.05, f"Heavy fraction {heavy_frac} not near 0.75"
        assert abs(light_frac - 0.25) < 0.05, f"Light fraction {light_frac} not near 0.25"


class TestEpoch:
    """Different epochs produce different orderings."""

    def test_different_epochs_different_order(self) -> None:
        domains = [
            DomainConfig("a", 1.0, _make_dataset(200, "a")),
        ]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=4, render_fn=_dummy_render_fn, seed=42
        )

        ds.set_epoch(seed=0)
        batches_epoch0 = [ds.get_batch(i) for i in range(10)]

        ds.set_epoch(seed=1)
        batches_epoch1 = [ds.get_batch(i) for i in range(10)]

        # Collect all model inputs
        inputs_0 = [d.model_input for batch in batches_epoch0 for d in batch]
        inputs_1 = [d.model_input for batch in batches_epoch1 for d in batch]

        differences = sum(1 for a, b in zip(inputs_0, inputs_1) if a != b)
        assert differences > 0, "Different epochs should produce different orderings"


class TestSmallDomainCycling:
    """A domain with few examples should cycle correctly over many batches."""

    def test_small_domain_cycles(self) -> None:
        # 10-row domain used over 100 batches of size 1 = 100 draws
        domains = [
            DomainConfig("small", 1.0, _make_dataset(10, "small")),
        ]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=1, render_fn=_dummy_render_fn, seed=42
        )

        # Should not raise, even though we request more items than domain size
        seen_texts: set[str] = set()
        n_batches = min(len(ds), 100)
        for i in range(n_batches):
            batch = ds.get_batch(i)
            assert len(batch) == 1

        # All 10 rows should appear at some point (with high probability over 10 batches)
        mixer = DomainMixer(domains, seed=42)
        positions = mixer.get_positions(0, 10)
        row_indices = {row for _, row in positions}
        assert len(row_indices) == 10, "All rows of small domain should appear in first cycle"


class TestEdgeCases:
    """Edge cases: single domain, zero-weight domain, empty dataset."""

    def test_single_domain(self) -> None:
        domains = [DomainConfig("only", 1.0, _make_dataset(50, "only"))]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=5, render_fn=_dummy_render_fn, seed=42
        )
        assert len(ds) == 10
        batch = ds.get_batch(0)
        assert len(batch) == 5

    def test_zero_weight_domain_excluded(self) -> None:
        domains = [
            DomainConfig("active", 1.0, _make_dataset(50, "active")),
            DomainConfig("inactive", 0.0, _make_dataset(50, "inactive")),
        ]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=5, render_fn=_dummy_render_fn, seed=42
        )
        # Length should only reflect active domain
        assert len(ds) == 10

        # All items should come from the active domain
        mixer = DomainMixer(
            [d for d in domains if d.weight > 0 and len(d.dataset) > 0], seed=42
        )
        positions = mixer.get_positions(0, 50)
        domain_indices = {d for d, _ in positions}
        assert domain_indices == {0}, "Only active domain should appear"

    def test_empty_dataset_filtered(self) -> None:
        domains = [
            DomainConfig("real", 1.0, _make_dataset(50, "real")),
            DomainConfig("empty", 1.0, _make_dataset(0, "empty")),
        ]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=5, render_fn=_dummy_render_fn, seed=42
        )
        assert len(ds) == 10
        batch = ds.get_batch(0)
        assert len(batch) == 5

    def test_all_zero_weight_raises(self) -> None:
        domains = [
            DomainConfig("zero", 0.0, _make_dataset(50, "zero")),
        ]
        with pytest.raises(ValueError, match="positive weight"):
            MultiDomainSupervisedDataset(
                domains, batch_size=5, render_fn=_dummy_render_fn, seed=42
            )

    def test_len_correct(self) -> None:
        domains = [
            DomainConfig("a", 1.0, _make_dataset(100, "a")),
            DomainConfig("b", 1.0, _make_dataset(50, "b")),
        ]
        ds = MultiDomainSupervisedDataset(
            domains, batch_size=10, render_fn=_dummy_render_fn, seed=42
        )
        # Total rows = 150, batch_size = 10 -> 15 batches
        assert len(ds) == 15
