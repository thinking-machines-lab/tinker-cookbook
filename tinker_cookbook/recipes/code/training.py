"""Code RL training infrastructure.

Provides environment building and RL dataset classes for training
code generation models with tool-based interaction.

Usage:
```bash
python -m tinker_cookbook.recipes.code.training model_name=<model>
```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
from tinker_cookbook import cli_utils, model_info, tokenizer_utils
from tinker_cookbook.recipes.code.task import CodeRLTask, load_deepcoder_tasks
from tinker_cookbook.recipes.code.tools import SubmissionReward, build_tools
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl import train, types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import AgentToolMessageEnv


def _system_prompt(task: CodeRLTask) -> str:
    """Build the system prompt for a code RL task."""
    tools_overview = (
        "- run_python(task_id, code): Test your code against the task's test cases.\n"
        "- submit_answer(task_id, code): Submit your final solution for grading."
    )
    return (
        "You are a Python coding assistant. Solve the problem below by writing Python code.\n"
        "You can test your code with run_python before submitting. Only submit when confident.\n"
        f"\nProblem:\n{task.problem}\n"
        f"\nTask ID: {task.task_id}\n"
        f"\nTools:\n{tools_overview}"
    )


def _initial_messages(task: CodeRLTask) -> list[Message]:
    """Build initial message history for a code RL task."""
    return [
        {"role": "system", "content": _system_prompt(task)},
        {"role": "user", "content": "Please solve this problem."},
    ]


def build_env(
    task: CodeRLTask,
    model_name: str,
    *,
    renderer_name: str | None = None,
    max_turns: int = 6,
    sandbox_backend: SandboxBackend | None = None,
    timeout: int = 6,
) -> EnvFromMessageEnv:
    """Build an RL environment for a single code RL task."""
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    chosen_renderer = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(chosen_renderer, tokenizer)

    tools = build_tools([task], sandbox_backend=sandbox_backend, timeout=timeout)
    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=_initial_messages(task),
        max_turns=max_turns,
        reward_fn=SubmissionReward(),
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=-0.1,
    )


class EnvGroupBuilder(types.EnvGroupBuilder):
    """EnvGroupBuilder that creates code RL environments with a shared sandbox backend."""

    def __init__(
        self,
        task: CodeRLTask,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        sandbox_backend: SandboxBackend | None,
        timeout: int = 6,
    ):
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_backend = sandbox_backend
        self.timeout = timeout

    async def make_envs(self) -> Sequence[types.Env]:
        return [
            build_env(
                task=self.task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.task.task_id]


class RLDataset(types.RLDataset):
    """Dataset that cycles through code RL EnvGroupBuilders."""

    def __init__(
        self,
        env_group_builders: list[EnvGroupBuilder],
        batch_size: int,
        num_batches: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size
        self.num_batches = num_batches

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.env_group_builders)
        builders: list[types.EnvGroupBuilder] = []
        for i in range(self.batch_size):
            builders.append(self.env_group_builders[(start + i) % len(self.env_group_builders)])
        return builders

    def __len__(self) -> int:
        return self.num_batches


@dataclass(frozen=True)
class RLDatasetBuilder(types.RLDatasetBuilder):
    """Build an RL dataset over DeepCoder tasks with sandbox execution."""

    model_name: str
    split: Literal["train", "test"] = "train"
    max_tasks: int | None = 100
    batch_size: int = 4
    group_size: int = 4
    renderer_name: str | None = None
    max_turns: int = 6
    num_batches: int = 50
    sandbox_backend: SandboxBackend | None = None
    timeout: int = 6
    seed: int = 0

    async def __call__(self) -> tuple[types.RLDataset, types.RLDataset | None]:
        tasks = load_deepcoder_tasks(
            split=self.split,
            max_tasks=self.max_tasks,
            seed=self.seed,
        )
        env_builders = [
            EnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
            )
            for task in tasks
        ]
        dataset = RLDataset(
            env_group_builders=env_builders,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        return dataset, None


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Create a train.Config blueprint for code RL with SandboxFusion."""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    dataset_builder = RLDatasetBuilder(
        model_name=model_name,
        renderer_name=renderer_name,
        batch_size=4,
        group_size=4,
        max_turns=6,
        max_tasks=100,
        sandbox_backend=SandboxBackend.SANDBOXFUSION,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "/tmp/tinker-examples/rl_code_rl",
            "dataset_builder": dataset_builder,
            "learning_rate": 4e-5,
            "max_tokens": 512,
            "eval_every": 0,
            "save_every": 0,
            "num_substeps": 1,
        }
    )


def main(config: train.Config) -> None:
    """Check log dir semantics then run the RL trainer."""
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    import sys

    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
