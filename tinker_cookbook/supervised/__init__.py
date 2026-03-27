"""Supervised learning: dataset builders, data utilities, and training loops."""

from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.prepack import (
    PrepackedSupervisedDataset,
    load_packed_datums,
    pack_token_lists,
    save_packed_datums,
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
    "FromConversationFileBuilder",
    "StreamingSupervisedDatasetFromHFDataset",
    "SupervisedDatasetFromHFDataset",
    "conversation_to_datum",
    # Pre-packing (prepack.py)
    "PrepackedSupervisedDataset",
    "load_packed_datums",
    "pack_token_lists",
    "save_packed_datums",
    # Helpers (common.py)
    "compute_mean_nll",
    "datum_from_model_input_weights",
]
