"""Harbor environment, dataset, and dataset builder for RL training."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import chz
import modal

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import Sandbox
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
from tinker_cookbook.tool_use import build_agent_tool_env

logger = logging.getLogger(__name__)

HARBOR_SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a sandboxed environment. "
    "You have access to a bash tool to execute commands. "
    "Complete the task described by the user."
)

SandboxFactory = Callable[[modal.Image, int], Awaitable[Sandbox]]


async def default_sandbox_factory(image: modal.Image, timeout: int) -> Sandbox:
    return await ModalSandbox.create(image=image, timeout=timeout)


@dataclass(frozen=True)
class HarborTask:
    """A single Harbor terminal-bench task."""

    task_name: str
    instruction: str
    task_dir: Path  # Convention: environment/Dockerfile, tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)


def _initial_messages(
    task: HarborTask,
    renderer: Renderer,
    bash_tool: HarborBashTool,
) -> list[Message]:
    """Build initial messages with tool schemas and task instruction."""
    tool_schemas = [bash_tool.bash.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=HARBOR_SYSTEM_PROMPT,
    )
    return prefix + [{"role": "user", "content": task.instruction}]


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that creates Harbor environments with Modal sandboxes."""

    def __init__(
        self,
        task: HarborTask,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        sandbox_timeout: int = 600,
        command_timeout: int = 120,
        grader_timeout: int = 60,
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
        self.grader_timeout = grader_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.sandbox_factory = sandbox_factory or default_sandbox_factory
        self._sandboxes: list[Sandbox] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []

        # Build Modal image from the task's Dockerfile
        env_dir = self.task.task_dir / "environment"
        dockerfile_path = env_dir / "Dockerfile"
        image = modal.Image.from_dockerfile(path=str(dockerfile_path), context_dir=str(env_dir))

        # Create renderer (stateless, shared across envs)
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        tests_dir = self.task.task_dir / "tests"

        envs = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(image, self.sandbox_timeout)
            self._sandboxes.append(sandbox)

            bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
            reward_fn = HarborReward(
                tests_dir=tests_dir,
                sandbox=sandbox,
                grader_timeout=self.grader_timeout,
            )
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[bash_tool.bash],
                    initial_messages=_initial_messages(self.task, renderer, bash_tool),
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
            )
        return envs

    async def compute_group_rewards(self, trajectory_group, env_group):
        """Cleanup sandboxes after rollouts + grading complete."""
        for sandbox in self._sandboxes:
            try:
                await sandbox.terminate()
            except Exception as e:
                logger.warning("Sandbox cleanup failed: %s", e)
        self._sandboxes.clear()
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["harbor", self.task.task_name]


class HarborDataset(RLDataset):
    """Dataset that produces batches of HarborEnvGroupBuilders."""

    def __init__(
        self,
        env_group_builders: list[HarborEnvGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    """Build an RL dataset over Harbor tasks."""

    tasks: list[HarborTask]
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str | None = None
    max_turns: int = 10
    sandbox_timeout: int = 600
    command_timeout: int = 120
    grader_timeout: int = 60
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
                grader_timeout=self.grader_timeout,
                sandbox_factory=self.sandbox_factory,
            )
            for task in self.tasks
        ]
        train_dataset = HarborDataset(
            env_group_builders=builders,
            batch_size=self.batch_size,
        )

        # Same tasks for eval (overfit experiment), group_size=1
        eval_builders = [
            HarborEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=1,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                sandbox_factory=self.sandbox_factory,
            )
            for task in self.tasks
        ]
        eval_dataset = HarborDataset(
            env_group_builders=eval_builders,
            batch_size=self.batch_size,
        )

        return train_dataset, eval_dataset
