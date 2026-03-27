"""
Supervised learning dataset implementations from HuggingFace datasets.
"""

import json
import logging
from collections.abc import Callable
from typing import Any, Literal, cast

import blobfile
import chz
import datasets
import tinker

from tinker_cookbook.exceptions import DataFormatError, DataValidationError
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tinker.Datum:
    """Common function to process a list of messages into a Datum."""
    model_input, weights = renderer.build_supervised_example(
        conversation, train_on_what=train_on_what
    )
    return datum_from_model_input_weights(model_input, weights, max_length)


def _one_of(a: Any, b: Any) -> bool:
    return (a is not None and b is None) or (a is None and b is not None)


class SupervisedDatasetFromHFDataset(SupervisedDataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        batch_size: int,
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = (
            hf_dataset  # Keep a reference to the original dataset to avoid statefulness
        )
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn

    def get_batch(self, index: int) -> list[tinker.Datum]:
        rows = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows.to_list()]
        else:
            assert self.flatmap_fn is not None
            return [datum for row in rows.to_list() for datum in self.flatmap_fn(row)]

    def set_epoch(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.hf_dataset) // self.batch_size


class StreamingSupervisedDatasetFromHFDataset(SupervisedDataset):
    def __init__(
        self,
        hf_dataset: datasets.IterableDataset,
        batch_size: int,
        length: int,
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
        buffer_size: int = 10_000,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset.shuffle(seed=0, buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_last_batch=True
        )
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn
        # We pass the length to the dataset, since streaming HF datasets don't have a length attribute
        self.length = length

    def get_batch(self, index: int) -> list[tinker.Datum]:
        # Error on backward seeks
        if index < self.index + 1:
            raise DataValidationError(
                f"StreamingSupervisedDatasetFromHFDataset only supports forward iteration. "
                f"Cannot seek backward from batch {self.index} to {index}."
            )

        # Skip forward if needed by consuming intermediate batches
        batches_to_skip = index - self.index - 1
        for _ in range(batches_to_skip):
            next(self.dataset_iterator)
            self.index += 1

        # Get the actual batch
        self.index = index
        batch = next(self.dataset_iterator)
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows]
        else:
            assert self.flatmap_fn is not None
            return [datum for row in rows for datum in self.flatmap_fn(row)]

    def set_epoch(self, seed: int = 0):
        self.hf_dataset.set_epoch(seed)
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1

    def __len__(self) -> int:
        return self.length // self.batch_size


@chz.chz
class FromConversationFileBuilder(ChatDatasetBuilder):
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise DataFormatError(
                        f"Each line in the JSONL file must contain a 'messages' field. Got: {data.keys()}"
                    )
                conversations.append(data)

        # Create HuggingFace dataset from the loaded data
        dataset = datasets.Dataset.from_list(conversations)

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split into train and test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            # If test_size is 0 or dataset is too small, use all data for training
            train_ds = dataset
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        # Create supervised dataset
        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        # Create evaluator if we have test data
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset


@chz.chz
class HFDatasetSource:
    """A single HuggingFace dataset to include in an interleaved mix.

    Attributes:
        path: HuggingFace dataset path (e.g. ``"allenai/tulu-3-sft-mixture"``).
        name: HuggingFace dataset config name (passed as ``name`` to ``load_dataset``).
        split: Dataset split to load.
        weight: Relative mixing weight. When set, weights are normalized across sources
            to determine sampling probabilities. When left as ``None`` (default) for all
            sources, each row across all sources is equally likely (i.e. simple concatenation
            with uniform sampling).
        message_field: Column name containing conversation messages.
    """

    path: str
    name: str | None = None
    split: str = "train"
    weight: float | None = None
    message_field: str = "messages"


@chz.chz
class InterleavedChatDatasetBuilder(ChatDatasetBuilder):
    """Builds an SFT dataset by interleaving multiple HuggingFace datasets.

    Uses ``datasets.interleave_datasets`` to mix rows from multiple sources according
    to the configured weights. When no weights are specified, sources are weighted by
    size so that every row is equally likely (equivalent to concatenation + shuffle).
    The resulting dataset is a standard Arrow-backed ``datasets.Dataset`` with O(1)
    random access and deterministic per-epoch shuffling.
    """

    sources: list[HFDatasetSource]
    test_size: int = 0
    shuffle_seed: int = 0
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "all_exhausted"

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        if not self.sources:
            raise ValueError("At least one dataset source must be provided")

        hf_datasets: list[datasets.Dataset] = []

        for source in self.sources:
            ds = datasets.load_dataset(source.path, name=source.name, split=source.split)
            if not isinstance(ds, datasets.Dataset):
                raise TypeError(
                    f"Expected a Dataset but got {type(ds).__name__}. "
                    f"Check that split='{source.split}' is valid for '{source.path}'."
                )
            if source.message_field != "messages":
                ds = ds.rename_column(source.message_field, "messages")
            ds = ds.select_columns(["messages"])
            hf_datasets.append(ds)
            logger.info(f"Loaded '{source.path}' ({len(ds)} rows, weight={source.weight})")

        # If all weights are None, weight by dataset size (uniform row sampling).
        # If any weight is set, all must be set.
        if all(s.weight is None for s in self.sources):
            weights: list[float] = [float(len(ds)) for ds in hf_datasets]
        elif any(s.weight is None for s in self.sources):
            raise ValueError(
                "Either all sources must have explicit weights or none of them. "
                "Got a mix of weighted and unweighted sources."
            )
        else:
            weights = [cast(float, s.weight) for s in self.sources]

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Total weight across all sources must be positive")
        probabilities = [w / total_weight for w in weights]

        interleaved = datasets.interleave_datasets(
            hf_datasets,
            probabilities=probabilities,
            seed=self.shuffle_seed,
            stopping_strategy=self.stopping_strategy,
        )
        if not isinstance(interleaved, datasets.Dataset):
            raise TypeError(
                f"Expected Dataset from interleave_datasets, got {type(interleaved).__name__}"
            )
        logger.info(f"Interleaved dataset: {len(interleaved)} rows")

        if self.test_size > 0 and len(interleaved) <= self.test_size:
            logger.warning(
                f"test_size ({self.test_size}) >= dataset size ({len(interleaved)}), skipping test split"
            )
        if self.test_size > 0 and len(interleaved) > self.test_size:
            interleaved = interleaved.shuffle(seed=self.shuffle_seed)
            test_ds = interleaved.take(self.test_size)
            train_ds = interleaved.skip(self.test_size)
        else:
            train_ds = interleaved
            test_ds = None

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        test_dataset = (
            SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), map_fn=map_fn)
            if test_ds is not None
            else None
        )

        return train_dataset, test_dataset
