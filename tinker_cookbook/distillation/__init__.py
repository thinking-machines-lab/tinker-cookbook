"""On-policy distillation training.

Provides dataset configuration and the training loop for distilling
from a teacher model into a student model using on-policy rollouts.

Example::

    from tinker_cookbook.distillation import Config, DistillationDatasetConfig
"""

from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)
from tinker_cookbook.distillation.train_on_policy import Config

__all__ = [
    # Training config and entry point
    "Config",
    # Dataset configuration
    "DistillationDatasetConfig",
    "TeacherConfig",
    "CompositeDataset",
    "PromptOnlyDatasetBuilder",
]
