"""
Dataset builders for SFT recipe.

Supports:
- NoRobots: HuggingFaceH4/no_robots dataset
- Tulu3: allenai/tulu-3-sft-mixture dataset
- Any HuggingFace dataset with a 'messages' column (e.g., parsed/phare-silver-messages)
- Custom JSONL files via FromConversationFileBuilder
"""

import logging
from typing import cast

import chz
import datasets
import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


@chz.chz
class NoRobotsBuilder(ChatDatasetBuilder):
    """Builder for HuggingFaceH4/no_robots dataset."""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        train_dataset = train_dataset.shuffle(seed=0)

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_dataset, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_dataset, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class Tulu3Builder(ChatDatasetBuilder):
    """Builder for allenai/tulu-3-sft-mixture dataset."""

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        # Use train_on_what from common_config if provided, otherwise default to LAST_ASSISTANT_MESSAGE
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )


@chz.chz
class HuggingFaceMessagesBuilder(ChatDatasetBuilder):
    """General builder for HuggingFace datasets with a 'messages' column.

    Works with any HF dataset that has conversations in a 'messages' column
    with the standard format: [{"role": "user", "content": "..."}, ...]
    """

    dataset_name: str  # HuggingFace dataset identifier (e.g., "parsed/phare-silver-messages")

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        # Load dataset - authentication is handled by huggingface_hub.login() in setup_environment()
        try:
            dataset = datasets.load_dataset(self.dataset_name, trust_remote_code=True)
        except Exception as e:
            # If it fails, print helpful error message
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load dataset '{self.dataset_name}'. Error: {e}")
            logger.error("Make sure:")
            logger.error(f"  1. The dataset exists: https://huggingface.co/datasets/{self.dataset_name}")
            logger.error("  2. Your HF_TOKEN has access to it (if it's private)")
            logger.error("  3. You're logged in to HuggingFace Hub")
            raise
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"]
        train_dataset = train_dataset.shuffle(seed=0)

        # Create test split from last 1024 examples
        test_ds = train_dataset.select(range(max(0, len(train_dataset) - 1024), len(train_dataset)))
        train_ds = train_dataset.select(range(0, max(0, len(train_dataset) - 1024)))

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        return SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        ), SupervisedDatasetFromHFDataset(
            test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
