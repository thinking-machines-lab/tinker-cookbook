"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

import json
import logging
from typing import Any, Callable, cast

import chz
import datasets
import tinker.types as types
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> types.Datum:
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
        map_fn: Callable[[dict], types.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[types.Datum]] | None = None,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = (
            hf_dataset  # Keep a reference to the original dataset to avoid statefulness
        )
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn

    def get_batch(self, index: int) -> list[types.Datum]:
        rows = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows.to_list()]
        else:
            assert self.flatmap_fn is not None
            return [datum for row in rows.to_list() for datum in self.flatmap_fn(row)]

    def shuffle(self, seed: int = 0):
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.hf_dataset) // self.batch_size


@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        # take the last 1000 as test, the rest as train
        def map_fn(row: dict) -> types.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, self.train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class NoRobotsBuilder(ChatDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, None]:
        dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)

        def map_fn(row: dict) -> types.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length
            )

        return SupervisedDatasetFromHFDataset(
            dataset, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), None


@chz.chz
class FromConversationFileBuilder(ChatDatasetBuilder):
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with open(self.file_path, "r") as f:
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

        # Define mapping function
        def map_fn(row: dict) -> types.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length
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
