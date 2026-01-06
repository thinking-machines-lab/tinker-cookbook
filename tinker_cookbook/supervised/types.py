"""
Basic interfaces and types for supervised training.
"""

import logging

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


class SupervisedDataset:
    """
    Dataset used for supervised learning
    """

    def get_batch(self, index: int) -> list[tinker.Datum]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def set_epoch(self, seed: int = 0):
        """Tell the dataset that we're on the given epoch of training.
        Datasets can decide what to do with this information, but for best
        results with multi-epoch training, you might want to shuffle differently each epoch,
        though results on whether this helps are inconclusive.
        """
        logger.warning(
            "set_epoch called, but shuffling is not implemented for %s",
            self.__class__.__name__,
        )


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
    train_on_what: renderers.TrainOnWhat | None = None


@chz.chz
class ChatDatasetBuilder(SupervisedDatasetBuilder):
    """
    Builds a chat dataset, which is a dataset that uses a renderer to convert from
    list-of-messages to tokens.
    """

    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """
        Return a training dataset and optionally an evaluation dataset.
        """
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.common_config.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(
            self.common_config.renderer_name,
            self.tokenizer,
        )
