"""
Basic interfaces and types for supervised training.
"""

import warnings

import chz
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


class SupervisedDataset:
    """
    Dataset used for supervised learning
    """

    def get_batch(self, index: int) -> list[types.Datum]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def shuffle(self, seed: int = 0):
        """Shuffle the dataset with the given seed."""
        warnings.warn("Shuffling the dataset is not implemented.")


@chz.chz
class SupervisedDatasetBuilder:
    """
    A config class that knows how to construct a supervised dataset. This dataset is usually a chat dataset but doesn't need to be; it could just be tokens.
    """

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        raise NotImplementedError


@chz.chz
class ChatDatasetBuilderCommonConfig:
    """
    Config that all chat dataset builders have
    Some specific datasets have additional options.
    """

    model_name_for_tokenizer: str
    renderer_name: str
    max_length: int | None
    batch_size: int


@chz.chz
class ChatDatasetBuilder(SupervisedDatasetBuilder):
    """
    Builds a chat dataset, which is a dataset that uses a renderer to convert from
    list-of-messages to tokens.
    """

    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.common_config.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(self.common_config.renderer_name, self.tokenizer)
