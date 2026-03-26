"""
Dataset builders for Nemotron-Cascade-2 SFT data.

Loads from nvidia/Nemotron-Cascade-2-SFT-Data on HuggingFace.
The dataset has subsets: math, science, chat, instruction_following, safety,
conversational_agent, swe, terminal_agent.

Conversations use standard OpenAI message format:
  [{"role": "system"|"user"|"assistant", "content": "..."}]
"""

import logging
from typing import Literal, cast

import chz
import datasets
import tinker

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import (
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/Nemotron-Cascade-2-SFT-Data"

# Subset sizes (approximate) for reference
SUBSET_SIZES = {
    "math": 5_226_364,
    "science": 2_717_163,
    "chat": 13_972_873,
    "instruction_following": 820_263,
    "safety": 3_570,
    "conversational_agent": 822_213,
    "swe": 439_610,
    "terminal_agent": 822_213,
}

SFTSubset = Literal[
    "math",
    "science",
    "chat",
    "instruction_following",
    "safety",
    "conversational_agent",
    "swe",
    "terminal_agent",
]


@chz.chz
class NemotronCascadeSFTBuilder(ChatDatasetBuilder):
    """Loads one or more subsets of Nemotron-Cascade-2-SFT-Data from HuggingFace."""

    subsets: tuple[SFTSubset, ...] = ("math",)
    max_examples: int | None = None
    test_size: int = 1024
    seed: int = 0
    streaming: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        if self.streaming:
            return self._build_streaming(map_fn)
        return self._build_eager(map_fn)

    def _build_eager(
        self, map_fn: datasets.typing.Callable
    ) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        all_datasets = []
        for subset in self.subsets:
            logger.info(f"Loading SFT subset: {subset}")
            ds = datasets.load_dataset(DATASET_NAME, name=subset, split="train")
            ds = cast(datasets.Dataset, ds)
            all_datasets.append(ds)

        if len(all_datasets) == 1:
            dataset = all_datasets[0]
        else:
            dataset = datasets.concatenate_datasets(all_datasets)

        dataset = dataset.shuffle(seed=self.seed)

        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        # Split train/test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        logger.info(f"SFT dataset: {len(train_ds)} train examples")

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset

    def _build_streaming(
        self, map_fn: datasets.typing.Callable
    ) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Use streaming for very large datasets to avoid downloading everything upfront."""
        all_streams = []
        total_size = 0
        for subset in self.subsets:
            logger.info(f"Loading SFT subset (streaming): {subset}")
            ds = datasets.load_dataset(DATASET_NAME, name=subset, split="train", streaming=True)
            ds = cast(datasets.IterableDataset, ds)
            all_streams.append(ds)
            total_size += SUBSET_SIZES.get(subset, 100_000)

        if len(all_streams) == 1:
            stream = all_streams[0]
        else:
            stream = datasets.interleave_datasets(all_streams)

        if self.max_examples is not None:
            total_size = min(total_size, self.max_examples)
            stream = stream.take(self.max_examples)

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            stream,
            batch_size=self.common_config.batch_size,
            length=total_size,
            map_fn=map_fn,
        )
        return train_dataset, None


@chz.chz
class NemotronCascadeSFTFromFileBuilder(ChatDatasetBuilder):
    """Loads SFT data from a local JSONL file (pre-downloaded and preprocessed)."""

    file_path: str
    test_size: int = 1024
    seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        import json

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        # Load JSONL
        rows = []
        with open(self.file_path) as f:
            for line in f:
                rows.append(json.loads(line))

        dataset = datasets.Dataset.from_list(rows)
        dataset = dataset.shuffle(seed=self.seed)

        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        logger.info(f"SFT dataset from file: {len(train_ds)} train examples")

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset
