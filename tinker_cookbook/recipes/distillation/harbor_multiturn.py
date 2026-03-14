"""Harbor environment for multi-turn on-policy distillation.

Provides a DatasetBuilder that creates harbor sandbox environments with zero
reward. The only training signal comes from KL divergence against a teacher
model (computed in the training loop).
"""

from __future__ import annotations

import logging

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDataset, HarborDatasetBuilder
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.types import RLDataset

logger = logging.getLogger(__name__)


async def zero_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    """Reward function that always returns zero. KL penalty is the only signal."""
    return 0.0, {}


@chz.chz
class HarborDistillationDatasetBuilder(HarborDatasetBuilder):
    """Build a distillation dataset over Harbor tasks (zero reward, KL only)."""

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_dataset = HarborDataset(
            env_group_builders=self._make_env_group_builders(self.group_size),
            batch_size=self.batch_size,
        )
        return train_dataset, None
