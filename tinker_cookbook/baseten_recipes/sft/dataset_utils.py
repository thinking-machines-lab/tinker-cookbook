"""
Dataset utilities for SFT recipe.

Provides wrappers and helpers for dataset builders.
"""

from typing import Iterator

import chz
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    SupervisedDataset,
    ChatDatasetBuilderCommonConfig,
)


class LimitedSupervisedDataset(SupervisedDataset):
    """Wrapper that limits the number of examples from a dataset."""

    def __init__(self, dataset: SupervisedDataset, num_examples: int):
        self.dataset = dataset
        self.num_examples = num_examples

    def __iter__(self) -> Iterator:
        count = 0
        for batch in self.dataset:
            if count >= self.num_examples:
                break
            # Yield full batches until we hit the limit
            batch_size = len(batch)
            if count + batch_size <= self.num_examples:
                yield batch
                count += batch_size
            else:
                # Partial batch to reach exact limit
                remaining = self.num_examples - count
                yield batch[:remaining]
                break


@chz.chz
class LimitedDatasetBuilder(ChatDatasetBuilder):
    """Wrapper builder that limits the number of training examples.

    Useful for quick experiments or debugging with smaller datasets.
    """

    builder: ChatDatasetBuilder
    num_examples: int

    @property
    def common_config(self) -> ChatDatasetBuilderCommonConfig:
        return self.builder.common_config

    @property
    def renderer(self):
        return self.builder.renderer

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_ds, test_ds = self.builder()

        # Limit training dataset
        limited_train_ds = LimitedSupervisedDataset(train_ds, self.num_examples)

        # Keep test dataset unchanged
        return limited_train_ds, test_ds
