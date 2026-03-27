"""Tests for InterleavedChatDatasetBuilder and HFDatasetSource."""

from typing import Literal
from unittest.mock import patch

import datasets
import pytest

from tinker_cookbook.supervised.data import HFDatasetSource, InterleavedChatDatasetBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def _make_hf_dataset(n: int, prefix: str = "row") -> datasets.Dataset:
    """Create an in-memory HF dataset with chat-formatted messages."""
    return datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": f"{prefix}_{i}"}] for i in range(n)]}
    )


def _make_hf_dataset_custom_field(
    n: int, prefix: str = "row", field: str = "conversations"
) -> datasets.Dataset:
    """Create an HF dataset with a non-standard message field name."""
    return datasets.Dataset.from_dict(
        {field: [[{"role": "user", "content": f"{prefix}_{i}"}] for i in range(n)]}
    )


def _mock_load_dataset(datasets_by_path: dict[str, datasets.Dataset]):
    """Return a mock load_dataset that returns pre-built datasets by path."""

    def _load(path: str, name: str | None = None, split: str = "train") -> datasets.Dataset:
        return datasets_by_path[path]

    return _load


class TestInterleavedChatDatasetBuilder:
    """Tests that exercise InterleavedChatDatasetBuilder via mocked load_dataset."""

    def _build(
        self,
        sources: list[HFDatasetSource],
        datasets_by_path: dict[str, datasets.Dataset],
        test_size: int = 0,
        batch_size: int = 4,
        stopping_strategy: Literal[
            "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
        ] = "all_exhausted",
    ):
        builder = InterleavedChatDatasetBuilder(
            sources=sources,
            test_size=test_size,
            stopping_strategy=stopping_strategy,
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
            HFDatasetSource(path="ds_a", weight=1.0),
            HFDatasetSource(path="ds_b", weight=1.0),
        ]
        train_ds, test_ds = self._build(sources, {"ds_a": ds_a, "ds_b": ds_b})
        assert test_ds is None
        assert len(train_ds) > 0
        batch = train_ds.get_batch(0)
        assert len(batch) == 4

    def test_default_weights_uniform_by_size(self) -> None:
        """Without explicit weights, sources are weighted by size."""
        ds_big = _make_hf_dataset(900, "big")
        ds_small = _make_hf_dataset(100, "small")
        sources = [
            HFDatasetSource(path="big"),
            HFDatasetSource(path="small"),
        ]
        train_ds, _ = self._build(sources, {"big": ds_big, "small": ds_small})
        assert len(train_ds) > 0
        batch = train_ds.get_batch(0)
        assert len(batch) == 4

    def test_mixed_weights_raises(self) -> None:
        """Mixing weighted and unweighted sources should raise."""
        sources = [
            HFDatasetSource(path="a", weight=2.0),
            HFDatasetSource(path="b"),
        ]
        with pytest.raises(ValueError, match="Either all sources must have explicit weights"):
            self._build(sources, {"a": _make_hf_dataset(50), "b": _make_hf_dataset(50)})

    def test_test_size_split(self) -> None:
        ds_a = _make_hf_dataset(200, "a")
        sources = [HFDatasetSource(path="ds_a", weight=1.0)]
        train_ds, test_ds = self._build(sources, {"ds_a": ds_a}, test_size=50, batch_size=10)
        assert test_ds is not None
        test_batch = test_ds.get_batch(0)
        assert len(test_batch) == 50

    def test_set_epoch_changes_order(self) -> None:
        ds_a = _make_hf_dataset(200, "a")
        sources = [HFDatasetSource(path="ds_a", weight=1.0)]
        train_ds, _ = self._build(sources, {"ds_a": ds_a}, batch_size=10)

        train_ds.set_epoch(seed=0)
        batch_e0 = train_ds.get_batch(0)

        train_ds.set_epoch(seed=1)
        batch_e1 = train_ds.get_batch(0)

        inputs_0 = [d.model_input for d in batch_e0]
        inputs_1 = [d.model_input for d in batch_e1]
        differences = sum(1 for a, b in zip(inputs_0, inputs_1) if a != b)
        assert differences > 0, "Different epochs should produce different orderings"

    def test_single_source(self) -> None:
        ds_a = _make_hf_dataset(50, "a")
        sources = [HFDatasetSource(path="ds_a", weight=1.0)]
        train_ds, test_ds = self._build(sources, {"ds_a": ds_a}, batch_size=5)
        assert test_ds is None
        assert len(train_ds) == 10
        batch = train_ds.get_batch(0)
        assert len(batch) == 5

    def test_custom_message_field(self) -> None:
        ds = _make_hf_dataset_custom_field(100, "conv", field="conversations")
        sources = [HFDatasetSource(path="ds_conv", weight=1.0, message_field="conversations")]
        train_ds, _ = self._build(sources, {"ds_conv": ds})
        assert len(train_ds) > 0
        batch = train_ds.get_batch(0)
        assert len(batch) == 4

    def test_empty_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one dataset source"):
            self._build([], {})

    def test_deterministic_batches(self) -> None:
        """Same config produces identical batches."""
        ds_a = _make_hf_dataset(100, "a")
        ds_b = _make_hf_dataset(100, "b")
        datasets_map = {"ds_a": ds_a, "ds_b": ds_b}
        sources = [
            HFDatasetSource(path="ds_a", weight=1.0),
            HFDatasetSource(path="ds_b", weight=1.0),
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
            HFDatasetSource(path="big", weight=1.0),
            HFDatasetSource(path="small", weight=1.0),
        ]
        train_ds, _ = self._build(
            sources, {"big": ds_big, "small": ds_small}, stopping_strategy="all_exhausted"
        )
        assert len(train_ds) > 0
        for i in range(min(5, len(train_ds))):
            batch = train_ds.get_batch(i)
            assert len(batch) == 4
