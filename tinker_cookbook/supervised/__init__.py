"""Supervised fine-tuning (SFT) training and data utilities.

Provides dataset builders for constructing supervised training data from
chat conversations or HuggingFace datasets, plus the training loop entry point.

Example::

    from tinker_cookbook.supervised import (
        ChatDatasetBuilder,
        SupervisedDatasetBuilder,
        conversation_to_datum,
    )
"""

from tinker_cookbook.supervised.common import (
    compute_mean_nll,
    datum_from_model_input_weights,
)
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.train import Config
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)

__all__ = [
    # Core types
    "SupervisedDataset",
    "SupervisedDatasetBuilder",
    "ChatDatasetBuilder",
    "ChatDatasetBuilderCommonConfig",
    # Training config and entry point
    "Config",
    # Dataset implementations
    "SupervisedDatasetFromHFDataset",
    "StreamingSupervisedDatasetFromHFDataset",
    "FromConversationFileBuilder",
    # Data utilities
    "conversation_to_datum",
    "datum_from_model_input_weights",
    "compute_mean_nll",
]
