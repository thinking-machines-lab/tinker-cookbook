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
    """Abstract base for datasets used in supervised learning.

    Subclasses must implement ``get_batch`` and ``__len__``.  ``set_epoch``
    may be overridden to shuffle data between epochs.
    """

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return a batch of training Datum objects at the given index.

        Args:
            index (int): The batch index.

        Returns:
            list[tinker.Datum]: The training datums for this batch.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        raise NotImplementedError

    def set_epoch(self, seed: int = 0):
        """Notify the dataset that a new epoch is starting.

        Implementations may use this to reshuffle data.  The default emits a
        warning that shuffling is not implemented.

        Args:
            seed (int): Epoch seed that can be used for deterministic
                shuffling. Default ``0``.
        """
        logger.warning(
            "set_epoch called, but shuffling is not implemented for %s",
            self.__class__.__name__,
        )


@chz.chz
class SupervisedDatasetBuilder:
    """A ``chz`` config class that knows how to construct a supervised dataset.

    The dataset is usually a chat dataset but does not need to be; it could
    be raw tokens.  Subclasses must implement ``__call__``.
    """

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build and return (train_dataset, eval_dataset).

        Returns:
            tuple[SupervisedDataset, SupervisedDataset | None]: The training
                dataset and an optional evaluation dataset.
        """
        raise NotImplementedError


@chz.chz
class ChatDatasetBuilderCommonConfig:
    """Shared configuration for all chat-based dataset builders.

    Specific dataset builders may add extra options on top of these.

    Attributes:
        model_name_for_tokenizer (str): HuggingFace model identifier used to
            load the tokenizer.
        renderer_name (str): Name of the renderer that converts chat messages
            to token sequences (e.g. ``"qwen3"``, ``"llama3"``).
        max_length (int | None): Maximum sequence length after tokenisation.
            ``None`` means no truncation.
        batch_size (int): Number of examples per training batch.
        train_on_what (renderers.TrainOnWhat | None): Which tokens receive
            non-zero loss weight.  ``None`` falls back to the default
            (``ALL_ASSISTANT_MESSAGES``).
    """

    model_name_for_tokenizer: str

    renderer_name: str
    max_length: int | None
    batch_size: int
    train_on_what: renderers.TrainOnWhat | None = None


@chz.chz
class ChatDatasetBuilder(SupervisedDatasetBuilder):
    """Build a chat dataset that uses a renderer to tokenise message lists.

    Subclasses must implement ``__call__`` to return the concrete datasets.

    Attributes:
        common_config (ChatDatasetBuilderCommonConfig): Shared configuration
            (tokenizer, renderer, batch size, etc.).
    """

    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build and return (train_dataset, eval_dataset).

        Returns:
            tuple[SupervisedDataset, SupervisedDataset | None]: Training
                dataset and an optional evaluation dataset.
        """
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer for this dataset's model."""
        return get_tokenizer(self.common_config.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        """Get the renderer for this dataset's model."""
        return renderers.get_renderer(
            self.common_config.renderer_name,
            self.tokenizer,
        )
