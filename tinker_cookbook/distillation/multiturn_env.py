"""Harbor environment for multi-turn on-policy distillation.

Provides a DatasetBuilder that creates harbor sandbox environments with zero
reward. The only training signal comes from KL divergence against a teacher
model (computed in the training loop).
"""

from __future__ import annotations

import logging

import chz

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborDataset,
    HarborEnvGroupBuilder,
    HarborTask,
    SandboxFactory,
)
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

logger = logging.getLogger(__name__)


async def _zero_reward(history) -> tuple[float, dict[str, float]]:
    """Reward function that always returns zero. KL penalty is the only signal."""
    return 0.0, {}


@chz.chz
class HarborDistillationDatasetBuilder(RLDatasetBuilder):
    """Build a distillation dataset over Harbor tasks (zero reward, KL only)."""

    tasks: list[HarborTask]
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str | None = None
    max_turns: int = 10
    sandbox_timeout: int = 600
    command_timeout: int = 120
    max_trajectory_tokens: int = 32 * 1024
    sandbox_factory: SandboxFactory | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        builders = [
            HarborEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                sandbox_factory=self.sandbox_factory,
                reward_fn=_zero_reward,
            )
            for task in self.tasks
        ]
        train_dataset = HarborDataset(
            env_group_builders=builders,
            batch_size=self.batch_size,
        )
        return train_dataset, None
