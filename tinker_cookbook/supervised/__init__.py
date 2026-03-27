"""Supervised learning: dataset builders, data utilities, and training loops."""

from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    HFDatasetSource,
    InterleavedChatDatasetBuilder,
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)

__all__ = [
    # Dataset abstractions (types.py)
    "ChatDatasetBuilder",
    "ChatDatasetBuilderCommonConfig",
    "SupervisedDataset",
    "SupervisedDatasetBuilder",
    # Dataset implementations and builders (data.py)
    "HFDatasetSource",
    "FromConversationFileBuilder",
    "InterleavedChatDatasetBuilder",
    "StreamingSupervisedDatasetFromHFDataset",
    "SupervisedDatasetFromHFDataset",
    "conversation_to_datum",
    # Helpers (common.py)
    "compute_mean_nll",
    "datum_from_model_input_weights",
]
