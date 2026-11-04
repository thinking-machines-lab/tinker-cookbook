"""Holdout partition contracts using real, in-memory HuggingFace datasets."""

from typing import Literal
from unittest.mock import patch

import datasets
import pytest

from tinker_cookbook.supervised.data import (
    HFDatasetSource,
    InterleavedChatDatasetBuilder,
    SupervisedDatasetFromHFDataset,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def _dataset(size: int, prefix: str) -> datasets.Dataset:
    rows = [[{"role": "user", "content": f"{prefix}{i}"}] for i in range(max(size, 1))]
    return datasets.Dataset.from_dict({"messages": rows}).select(range(size))


def _build(
    source_datasets: list[datasets.Dataset],
    *,
    weights: list[float] | None = None,
    test_size: int = 3,
    seed: int = 0,
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "all_exhausted",
) -> tuple[SupervisedDatasetFromHFDataset, SupervisedDatasetFromHFDataset | None]:
    by_path = {f"source_{i}": ds for i, ds in enumerate(source_datasets)}
    sources = [
        HFDatasetSource(path=path, weight=weights[i] if weights is not None else None)
        for i, path in enumerate(by_path)
    ]
    builder = InterleavedChatDatasetBuilder(
        sources=sources,
        test_size=test_size,
        shuffle_seed=seed,
        stopping_strategy=stopping_strategy,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="local-synthetic",
            renderer_name="role_colon",
            batch_size=1,
            max_length=128,
        ),
    )
    with patch("tinker_cookbook.supervised.data.datasets.load_dataset") as load:
        load.side_effect = lambda path, **kwargs: by_path[path]
        train, heldout = builder()
    assert isinstance(train, SupervisedDatasetFromHFDataset)
    assert heldout is None or isinstance(heldout, SupervisedDatasetFromHFDataset)
    return train, heldout


def _ids(dataset: SupervisedDatasetFromHFDataset) -> list[str]:
    return [str(row["messages"][0]["content"]) for row in dataset.hf_dataset.to_list()]


@pytest.mark.parametrize("weights", [None, [1.0, 1.0]])
@pytest.mark.parametrize(
    "stopping_strategy", ["first_exhausted", "all_exhausted", "all_exhausted_without_replacement"]
)
def test_holdout_uses_unique_original_rows_disjoint_from_training(
    weights: list[float] | None,
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ],
) -> None:
    train, heldout = _build(
        [_dataset(2, "a"), _dataset(10, "b")],
        weights=weights,
        stopping_strategy=stopping_strategy,
    )
    assert heldout is not None
    train_ids, heldout_ids = _ids(train), _ids(heldout)
    assert len(heldout_ids) == 3
    assert len(set(heldout_ids)) == 3
    assert set(train_ids).isdisjoint(heldout_ids)
    if stopping_strategy != "first_exhausted":
        assert any(row.startswith("a") for row in train_ids)
        assert any(row.startswith("b") for row in train_ids)


@pytest.mark.parametrize("weights", [None, [1.0, 1.0]])
@pytest.mark.parametrize(
    "stopping_strategy", ["first_exhausted", "all_exhausted", "all_exhausted_without_replacement"]
)
def test_no_holdout_keeps_the_existing_interleave(
    weights: list[float] | None,
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ],
) -> None:
    source_datasets = [_dataset(2, "a"), _dataset(10, "b")]
    effective_weights = weights if weights is not None else [2.0, 10.0]
    expected = datasets.interleave_datasets(
        source_datasets,
        probabilities=[w / sum(effective_weights) for w in effective_weights],
        seed=0,
        stopping_strategy=stopping_strategy,
    )
    train, heldout = _build(
        source_datasets, weights=weights, test_size=0, stopping_strategy=stopping_strategy
    )
    assert heldout is None
    assert train.hf_dataset.to_list() == expected.to_list()
    assert len(source_datasets[0]) == 2
    assert len(source_datasets[1]) == 10


def test_holdout_preserves_a_singleton_training_source() -> None:
    train, heldout = _build([_dataset(1, "a"), _dataset(10, "b")], weights=[1.0, 1.0])
    assert heldout is not None
    assert "a0" in _ids(train)
    assert len(_ids(heldout)) == len(set(_ids(heldout))) == 3
    assert all(row.startswith("b") for row in _ids(heldout))
    assert set(_ids(train)).isdisjoint(_ids(heldout))


@pytest.mark.parametrize("weights", [None, [3.0, 1.0]])
def test_training_probabilities_follow_the_remaining_training_sources(
    weights: list[float] | None,
) -> None:
    interleave = datasets.interleave_datasets
    with patch(
        "tinker_cookbook.supervised.data.datasets.interleave_datasets", wraps=interleave
    ) as mix:
        train, heldout = _build(
            [_dataset(20, "a"), _dataset(30, "b")], weights=weights, test_size=8
        )
    assert heldout is not None
    train_ids = set(_ids(train))
    remaining = [sum(row.startswith(prefix) for row in train_ids) for prefix in ("a", "b")]
    expected_weights = weights if weights is not None else remaining
    assert mix.call_args.kwargs["probabilities"] == pytest.approx(
        [weight / sum(expected_weights) for weight in expected_weights]
    )
    assert mix.call_args.kwargs["stopping_strategy"] == "all_exhausted"


def test_explicit_holdout_weights_redistribute_only_at_source_capacity() -> None:
    train, heldout = _build([_dataset(2, "a"), _dataset(10, "b")], weights=[9.0, 1.0], test_size=5)
    assert heldout is not None
    heldout_ids = _ids(heldout)
    assert sum(row.startswith("a") for row in heldout_ids) == 1
    assert sum(row.startswith("b") for row in heldout_ids) == 4
    assert len(set(heldout_ids)) == 5
    assert set(_ids(train)).isdisjoint(heldout_ids)


def test_explicit_holdout_weights_change_source_distribution() -> None:
    _, heldout = _build([_dataset(500, "a"), _dataset(500, "b")], weights=[9.0, 1.0], test_size=100)
    assert heldout is not None
    assert 80 <= sum(row.startswith("a") for row in _ids(heldout)) <= 99


def test_default_holdout_weights_sample_eligible_rows_by_size() -> None:
    _, heldout = _build([_dataset(100, "a"), _dataset(900, "b")], test_size=100)
    assert heldout is not None
    assert 1 <= sum(row.startswith("a") for row in _ids(heldout)) <= 20


def test_default_holdout_weights_count_all_still_eligible_original_rows() -> None:
    _, heldout = _build([_dataset(2, "a"), _dataset(10, "b")], test_size=1, seed=11)
    assert heldout is not None
    assert _ids(heldout)[0].startswith("a")


def test_maximum_feasible_holdout_retains_every_training_source() -> None:
    train, heldout = _build([_dataset(2, "a"), _dataset(3, "b")], test_size=3)
    assert heldout is not None
    assert len(_ids(heldout)) == len(set(_ids(heldout))) == 3
    assert len(set(_ids(train))) == 2
    assert any(row.startswith("a") for row in _ids(train))
    assert any(row.startswith("b") for row in _ids(train))
    assert set(_ids(train)).isdisjoint(_ids(heldout))


@pytest.mark.parametrize("test_size", [4, 5, 6])
def test_impossible_holdout_fails_instead_of_removing_a_training_source(test_size: int) -> None:
    with pytest.raises(ValueError, match=r"test_size.*retaining one training row"):
        _build([_dataset(2, "a"), _dataset(3, "b")], test_size=test_size)


def test_singleton_sources_cannot_supply_a_holdout() -> None:
    with pytest.raises(ValueError, match=r"maximum unique holdout size \(0\)"):
        _build([_dataset(1, "a"), _dataset(1, "b")], test_size=1)


def test_empty_positively_weighted_source_fails_clearly() -> None:
    with pytest.raises(ValueError, match="empty positively weighted source"):
        _build([_dataset(0, "a"), _dataset(10, "b")], weights=[1.0, 1.0])


@pytest.mark.parametrize("weights", [None, [0.0, 1.0]])
def test_empty_zero_weight_source_does_not_supply_holdout_rows(weights: list[float] | None) -> None:
    train, heldout = _build([_dataset(0, "a"), _dataset(10, "b")], weights=weights)
    assert heldout is not None
    assert len(_ids(heldout)) == len(set(_ids(heldout))) == 3
    assert all(row.startswith("b") for row in _ids(train) + _ids(heldout))
    assert set(_ids(train)).isdisjoint(_ids(heldout))


@pytest.mark.parametrize(
    "stopping_strategy", ["first_exhausted", "all_exhausted", "all_exhausted_without_replacement"]
)
def test_zero_weight_source_is_excluded_from_both_holdout_partitions(
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ],
) -> None:
    train, heldout = _build(
        [_dataset(5, "a"), _dataset(10, "b")],
        weights=[0.0, 1.0],
        stopping_strategy=stopping_strategy,
    )
    assert heldout is not None
    assert len(_ids(heldout)) == 3
    assert all(row.startswith("b") for row in _ids(train) + _ids(heldout))
    assert set(_ids(train)).isdisjoint(_ids(heldout))


@pytest.mark.parametrize("weights", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_holdout_rejects_non_finite_or_all_zero_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError, match=r"finite|positive"):
        _build([_dataset(2, "a"), _dataset(10, "b")], weights=weights)


def test_holdout_rejects_an_all_empty_original_pool() -> None:
    with pytest.raises(ValueError, match="positive"):
        _build([_dataset(0, "a"), _dataset(0, "b")], test_size=1)


def test_holdout_rejects_negative_source_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _build([_dataset(2, "a"), _dataset(10, "b")], weights=[-1.0, 2.0])


@pytest.mark.parametrize("empty", [False, True])
def test_no_holdout_forwards_sources_and_dependency_errors(empty: bool) -> None:
    source_datasets = [_dataset(0 if empty else 2, "a"), _dataset(10, "b")]
    weights = [1.0, 1.0] if empty else [0.0, 1.0]
    dependency_error = ValueError("interleave failed")
    with patch(
        "tinker_cookbook.supervised.data.datasets.interleave_datasets", side_effect=dependency_error
    ) as mix:
        with pytest.raises(ValueError) as actual:
            _build(source_datasets, weights=weights, test_size=0)
    assert actual.value is dependency_error
    assert [ds.to_list() for ds in mix.call_args.args[0]] == [
        ds.to_list() for ds in source_datasets
    ]
    assert mix.call_args.kwargs["probabilities"] == [weight / sum(weights) for weight in weights]


def test_distinct_original_rows_with_identical_content_are_not_deduplicated() -> None:
    ds = datasets.Dataset.from_dict({"messages": [[{"role": "user", "content": "identical"}]] * 3})
    train, heldout = _build([ds], test_size=1)
    assert heldout is not None
    assert len(train.hf_dataset) == 2
    assert len(heldout.hf_dataset) == 1
    assert train.hf_dataset[0] == heldout.hf_dataset[0]


def test_holdout_is_deterministic_for_the_same_seed() -> None:
    source_datasets = [_dataset(20, "a"), _dataset(30, "b")]
    first = _build(source_datasets, weights=[2.0, 1.0], test_size=8, seed=7)
    second = _build(source_datasets, weights=[2.0, 1.0], test_size=8, seed=7)
    other = _build(source_datasets, weights=[2.0, 1.0], test_size=8, seed=8)
    assert first[1] is not None and second[1] is not None and other[1] is not None
    assert _ids(first[0]) == _ids(second[0])
    assert _ids(first[1]) == _ids(second[1])
    assert _ids(first[1]) != _ids(other[1])
