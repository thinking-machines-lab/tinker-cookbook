"""Supervised learning: dataset builders, data utilities, and training loops."""

from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.packing import greedy_pack, make_datum, pack_to_datums
from tinker_cookbook.supervised.prepack import (
    PrepackedDatasetBuilder,
    PrepackedSupervisedDataset,
    load_packed_datums,
    render_parallel,
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
    # Packing algorithm (packing.py)
    "greedy_pack",
    "make_datum",
    "pack_to_datums",
    # Pre-pack pipeline (prepack.py)
    "PrepackedDatasetBuilder",
    "PrepackedSupervisedDataset",
    "load_packed_datums",
    "render_parallel",
    "save_packed_datums",
    # Helpers (common.py)
    "compute_mean_nll",
    "datum_from_model_input_weights",
]
