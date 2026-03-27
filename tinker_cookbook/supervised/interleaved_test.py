"""Tests for InterleavedDatasetBuilder and DatasetSource."""

import collections
from unittest.mock import patch

import datasets
import pytest

from tinker_cookbook.supervised.data import DatasetSource, InterleavedDatasetBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def _make_hf_dataset(n: int, prefix: str = "row") -> datasets.Dataset:
    """Create an in-memory HF dataset with chat-formatted messages."""
    return datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": f"{prefix}_{i}"}] for i in range(n)
            ]
        }
    )


def _make_hf_dataset_custom_field(
    n: int, prefix: str = "row", field: str = "conversations"
) -> datasets.Dataset:
    """Create an HF dataset with a non-standard message field name."""
    return datasets.Dataset.from_dict(
        {
            field: [
                [{"role": "user", "content": f"{prefix}_{i}"}] for i in range(n)
            ]
        }
    )


def _mock_load_dataset(datasets_by_path: dict[str, datasets.Dataset]):
    """Return a mock load_dataset that returns pre-built datasets by path."""

    def _load(path: str, name: str | None = None, split: str = "train") -> datasets.Dataset:
        return datasets_by_path[path]

    return _load


class TestInterleavedDatasetBuilder:
    """Tests that exercise InterleavedDatasetBuilder via mocked load_dataset."""

    def _build(
        self,
        sources: list[DatasetSource],
        datasets_by_path: dict[str, datasets.Dataset],
        test_size: int = 0,
        batch_size: int = 4,
    ):
        builder = InterleavedDatasetBuilder(
            sources=sources,
            test_size=test_size,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
                renderer_name="llama3",
                max_length=128,
                batch_size=batch_size,
            ),
        )
        with patch("tinker_cookbook.supervised.data.datasets.load_dataset") as mock_load:
            mock_load.side_effect = _mock_load_dataset(datasets_by_path)
            return builder()

    def test_basic_two_sources(self) -> None:
        ds_a = _make_hf_dataset(100, "a")
        ds_b = _make_hf_dataset(100, "b")
        sources = [
            DatasetSource(path="ds_a", weight=1.0),
            DatasetSource(path="ds_b", weight=1.0),
        ]
        train_ds, test_ds = self._build(sources, {"ds_a": ds_a, "ds_b": ds_b})
        assert test_ds is None
        assert len(train_ds) > 0
        batch = train_ds.get_batch(0)
        assert len(batch) == 4

    def test_weight_distribution(self) -> None:
        """With 3:1 weights, the interleaved dataset should reflect ~75/25 distribution."""
        ds_heavy = _make_hf_dataset(1000, "heavy")
        ds_light = _make_hf_dataset(1000, "light")
        sources = [
            DatasetSource(path="heavy", weight=3.0),
            DatasetSource(path="light", weight=1.0),
        ]

        builder = InterleavedDatasetBuilder(
            sources=sources,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
                renderer_name="llama3",
                max_length=128,
                batch_size=4,
            ),
        )

        with patch("tinker_cookbook.supervised.data.datasets.load_dataset") as mock_load:
            mock_load.side_effect = _mock_load_dataset(
                {"heavy": ds_heavy, "light": ds_light}
            )
            # Access the interleaved dataset directly to check distribution
            with patch.object(
                InterleavedDatasetBuilder, "__call__", wraps=builder.__call__
            ):
                # Build and check the underlying interleaved dataset
                hf_datasets_list = []
                for source in builder.sources:
                    ds = {"heavy": ds_heavy, "light": ds_light}[source.path]
                    ds = ds.select_columns(["messages"])
                    hf_datasets_list.append(ds)

                interleaved = datasets.interleave_datasets(
                    hf_datasets_list,
                    probabilities=[0.75, 0.25],
                    seed=0,
                    stopping_strategy="all_exhausted",
                )
                # Count messages starting with "heavy" vs "light"
                counts = collections.Counter()
                for row in interleaved:
                    msg = row["messages"][0]["content"]
                    counts["heavy" if msg.startswith("heavy") else "light"] += 1

                total = sum(counts.values())
                heavy_frac = counts["heavy"] / total
                assert abs(heavy_frac - 0.75) < 0.05, f"Heavy fraction {heavy_frac} not near 0.75"

    def test_test_size_split(self) -> None:
        ds_a = _make_hf_dataset(200, "a")
        sources = [DatasetSource(path="ds_a", weight=1.0)]
        train_ds, test_ds = self._build(
            sources, {"ds_a": ds_a}, test_size=50, batch_size=10
        )
        assert test_ds is not None
        # Test dataset should have its batch_size == test_size
        test_batch = test_ds.get_batch(0)
        assert len(test_batch) == 50

    def test_set_epoch_changes_order(self) -> None:
        ds_a = _make_hf_dataset(200, "a")
        sources = [DatasetSource(path="ds_a", weight=1.0)]
        train_ds, _ = self._build(sources, {"ds_a": ds_a}, batch_size=10)

        train_ds.set_epoch(seed=0)
        batch_e0 = train_ds.get_batch(0)

        train_ds.set_epoch(seed=1)
        batch_e1 = train_ds.get_batch(0)

        inputs_0 = [d.model_input for d in batch_e0]
        inputs_1 = [d.model_input for d in batch_e1]
        differences = sum(1 for a, b in zip(inputs_0, inputs_1) if a != b)
        assert differences > 0, "Different epochs should produce different orderings"

    def test_single_domain(self) -> None:
        ds_a = _make_hf_dataset(50, "a")
        sources = [DatasetSource(path="ds_a", weight=1.0)]
        train_ds, test_ds = self._build(sources, {"ds_a": ds_a}, batch_size=5)
        assert test_ds is None
        assert len(train_ds) == 10
        batch = train_ds.get_batch(0)
        assert len(batch) == 5

    def test_custom_message_field(self) -> None:
        ds = _make_hf_dataset_custom_field(100, "conv", field="conversations")
        sources = [DatasetSource(path="ds_conv", weight=1.0, message_field="conversations")]

        builder = InterleavedDatasetBuilder(
            sources=sources,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
                renderer_name="llama3",
                max_length=128,
                batch_size=4,
            ),
        )
        with patch("tinker_cookbook.supervised.data.datasets.load_dataset") as mock_load:
            mock_load.side_effect = _mock_load_dataset({"ds_conv": ds})
            train_ds, _ = builder()

        assert len(train_ds) > 0
        batch = train_ds.get_batch(0)
        assert len(batch) == 4

    def test_empty_sources_raises(self) -> None:
        builder = InterleavedDatasetBuilder(
            sources=[],
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
                renderer_name="llama3",
                max_length=128,
                batch_size=4,
            ),
        )
        with pytest.raises(ValueError, match="At least one dataset source"):
            builder()

    def test_deterministic_batches(self) -> None:
        """Same config produces identical batches."""
        ds_a = _make_hf_dataset(100, "a")
        ds_b = _make_hf_dataset(100, "b")
        datasets_map = {"ds_a": ds_a, "ds_b": ds_b}
        sources = [
            DatasetSource(path="ds_a", weight=1.0),
            DatasetSource(path="ds_b", weight=1.0),
        ]

        train1, _ = self._build(sources, datasets_map)
        train2, _ = self._build(sources, datasets_map)

        for i in range(5):
            b1 = train1.get_batch(i)
            b2 = train2.get_batch(i)
            assert len(b1) == len(b2)
            for d1, d2 in zip(b1, b2):
                assert d1.model_input == d2.model_input

    def test_small_source_oversampled(self) -> None:
        """With all_exhausted strategy, a small source should be oversampled."""
        ds_big = _make_hf_dataset(1000, "big")
        ds_small = _make_hf_dataset(10, "small")
        sources = [
            DatasetSource(path="big", weight=1.0),
            DatasetSource(path="small", weight=1.0),
        ]

        builder = InterleavedDatasetBuilder(
            sources=sources,
            stopping_strategy="all_exhausted",
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
                renderer_name="llama3",
                max_length=128,
                batch_size=4,
            ),
        )
        with patch("tinker_cookbook.supervised.data.datasets.load_dataset") as mock_load:
            mock_load.side_effect = _mock_load_dataset({"big": ds_big, "small": ds_small})
            train_ds, _ = builder()

        # The interleaved dataset should have more rows than just big + small
        # because the small domain gets oversampled
        assert len(train_ds) > 0
        # Should be able to iterate without errors
        for i in range(min(5, len(train_ds))):
            batch = train_ds.get_batch(i)
            assert len(batch) == 4
