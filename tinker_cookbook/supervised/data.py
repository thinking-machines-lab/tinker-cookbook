"""
Supervised learning dataset implementations from HuggingFace datasets.
"""

import json
from typing import Any, Callable

import blobfile
import chz
import datasets
import tinker
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tinker.Datum:
    """Common function to process a list of messages into a Datum."""
    tokens, weights = renderer.build_supervised_example(conversation, train_on_what=train_on_what)
    return datum_from_tokens_weights(tokens, weights, max_length)


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
        # TODO: this is a hack to make sure the index is correct
        # should maybe think about a more robust way to do this
        assert index == self.index + 1
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
                    raise ValueError(
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
