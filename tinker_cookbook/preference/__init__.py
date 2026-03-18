"""Preference learning: Direct Preference Optimization (DPO) and RLHF.

Provides preference data types, comparison renderers, preference models,
and the DPO training entry point.

Example::

    from tinker_cookbook.preference import Comparison, LabeledComparison
    from tinker_cookbook.preference import Config  # DPO training config
"""

from tinker_cookbook.preference.preference_datasets import (
    ChatDatasetBuilderFromComparisons,
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.train_dpo import Config
from tinker_cookbook.preference.types import (
    Comparison,
    ComparisonRenderer,
    ComparisonRendererFromChatRenderer,
    LabeledComparison,
    PreferenceModel,
    PreferenceModelBuilder,
    PreferenceModelBuilderFromChatRenderer,
    PreferenceModelFromChatRenderer,
)

__all__ = [
    # Core preference types
    "Comparison",
    "LabeledComparison",
    # Comparison rendering
    "ComparisonRenderer",
    "ComparisonRendererFromChatRenderer",
    # Preference models
    "PreferenceModel",
    "PreferenceModelBuilder",
    "PreferenceModelFromChatRenderer",
    "PreferenceModelBuilderFromChatRenderer",
    # Dataset builders
    "ComparisonDatasetBuilder",
    "ChatDatasetBuilderFromComparisons",
    # DPO training config and entry point
    "Config",
]
