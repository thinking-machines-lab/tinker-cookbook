"""Harbor environment for multi-turn on-policy distillation.

Provides an EnvGroupBuilder, Dataset, and DatasetBuilder that create harbor
sandbox environments with zero reward. The only training signal comes from
KL divergence against a teacher model (computed in the training loop).
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import chz
import modal

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    _initial_messages,
    default_sandbox_factory,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import build_agent_tool_env

logger = logging.getLogger(__name__)


async def _zero_reward(history) -> tuple[float, dict[str, float]]:
    """Reward function that always returns zero. KL penalty is the only signal."""
    return 0.0, {}


class HarborDistillationEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder for harbor distillation: sandbox envs with zero reward."""

    def __init__(
        self,
        task: HarborTask,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        sandbox_timeout: int = 600,
        command_timeout: int = 120,
        max_trajectory_tokens: int = 32 * 1024,
        sandbox_factory: SandboxFactory | None = None,
    ):
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.sandbox_factory = sandbox_factory or default_sandbox_factory
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []

        env_dir = self.task.task_dir / "environment"
        dockerfile_path = env_dir / "Dockerfile"
        image = modal.Image.from_dockerfile(path=str(dockerfile_path), context_dir=str(env_dir))

        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        envs = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(image, self.sandbox_timeout)
            self._sandboxes.append(sandbox)

            bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[bash_tool.bash],
                    initial_messages=_initial_messages(self.task, renderer, bash_tool),
                    reward_fn=_zero_reward,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
            )
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        """Cleanup sandboxes after rollouts complete."""
        for sandbox in self._sandboxes:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed: %s", e)
        self._sandboxes.clear()
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["harbor_distill"]


class HarborDistillationDataset(RLDataset):
    """Dataset that produces batches of HarborDistillationEnvGroupBuilders."""

    def __init__(
        self,
        env_group_builders: list[HarborDistillationEnvGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return math.ceil(len(self.env_group_builders) / self.batch_size)


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
            HarborDistillationEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                sandbox_factory=self.sandbox_factory,
            )
            for task in self.tasks
        ]
        train_dataset = HarborDistillationDataset(
            env_group_builders=builders,
            batch_size=self.batch_size,
        )
        return train_dataset, None
