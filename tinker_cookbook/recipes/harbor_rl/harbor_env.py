"""Harbor environment, dataset, and dataset builder for RL training."""

from __future__ import annotations

import asyncio
import logging
import tomllib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxBackend, SandboxInterface
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn

logger = logging.getLogger(__name__)

HARBOR_CACHE_DIR = Path.home() / ".cache" / "harbor" / "tasks"
HARBOR_SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a sandboxed environment. "
    "You have access to a bash tool to execute commands. "
    "Complete the task described by the user."
)

SandboxFactory = Callable[[Path, int], Awaitable[SandboxInterface]]


async def default_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Create a Modal sandbox from a task environment directory.

    Args:
        env_dir: Path to the task's environment/ directory (must contain a Dockerfile).
        timeout: Sandbox lifetime in seconds.
    """
    import modal

    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    dockerfile_path = env_dir / "Dockerfile"
    image = modal.Image.from_dockerfile(path=str(dockerfile_path), context_dir=str(env_dir))
    return await ModalSandbox.create(image=image, timeout=timeout)


def prebuilt_image_ref(env_dir: Path) -> str | None:
    """Return the task's ``environment.docker_image`` from task.toml, if set.

    ``env_dir`` is the task's ``environment/`` directory, so task.toml sits
    in its parent.
    """
    task_toml = env_dir.parent / "task.toml"
    if not task_toml.is_file():
        return None
    config = tomllib.loads(task_toml.read_text())
    ref = config.get("environment", {}).get("docker_image")
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    return None


def _tensorlake_task_image(env_dir: Path) -> str:
    """Resolve a task's Tensorlake image. Blocks; run in a thread.

    Prefers the prebuilt ``docker_image`` from task.toml. Harbor publishes
    the Terminal-Bench images to Tensorlake, so this path needs no build.
    Falls back to building the task's Dockerfile.
    """
    from tinker_cookbook.sandbox.tensorlake_sandbox import (
        build_image_from_dockerfile,
        resolve_registry_image,
    )

    task_name = env_dir.parent.name
    ref = prebuilt_image_ref(env_dir)
    if ref is not None:
        try:
            name = resolve_registry_image(ref)
        except Exception as e:
            logger.warning(
                "%s: could not use prebuilt image %s (%s); building from Dockerfile",
                task_name,
                ref,
                e,
            )
        else:
            logger.info("%s: using prebuilt Tensorlake image %s (%s)", task_name, name, ref)
            return name
    name = build_image_from_dockerfile(env_dir / "Dockerfile", env_dir)
    logger.info("%s: using Tensorlake image %s built from Dockerfile", task_name, name)
    return name


# Image resolutions keyed by environment directory, so concurrent rollouts of
# one task share a single lookup or build.
_tensorlake_image_builds: dict[Path, asyncio.Task[str]] = {}


async def _get_tensorlake_image(env_dir: Path) -> str:
    build = _tensorlake_image_builds.get(env_dir)
    if build is None:
        build = asyncio.create_task(asyncio.to_thread(_tensorlake_task_image, env_dir))
        _tensorlake_image_builds[env_dir] = build
    try:
        # shield: a cancelled caller must not cancel the build that other rollouts share.
        return await asyncio.shield(build)
    except Exception:
        # Let the next call try the build again.
        if _tensorlake_image_builds.get(env_dir) is build:
            del _tensorlake_image_builds[env_dir]
        raise


async def tensorlake_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Create a Tensorlake sandbox from a task environment directory.

    Uses the task's prebuilt ``docker_image`` when Tensorlake has it (Harbor
    publishes the Terminal-Bench images). Otherwise the image is built once
    per Dockerfile and cached by Tensorlake.

    Args:
        env_dir: Path to the task's environment/ directory (must contain a Dockerfile).
        timeout: Sandbox lifetime in seconds.
    """
    from tinker_cookbook.sandbox.tensorlake_sandbox import TensorlakeSandbox

    image = await _get_tensorlake_image(env_dir)
    return await TensorlakeSandbox.create(image=image, timeout=timeout)


def get_sandbox_factory(backend: SandboxBackend) -> SandboxFactory:
    """Return the Harbor sandbox factory for a backend."""
    if backend == SandboxBackend.MODAL:
        return default_sandbox_factory
    if backend == SandboxBackend.TENSORLAKE:
        return tensorlake_sandbox_factory
    raise ValueError(f"Harbor tasks do not support sandbox backend {backend!r}")


@dataclass(frozen=True)
class HarborTask:
    """A single Harbor terminal-bench task."""

    task_name: str
    instruction: str
    task_dir: Path  # Convention: environment/Dockerfile, tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)


def parse_task_names(task_names: str | None) -> list[str] | None:
    """Split a comma-separated task name list from the CLI. None means all tasks."""
    if task_names is None:
        return None
    names = [name.strip() for name in task_names.split(",") if name.strip()]
    return names or None


def load_harbor_tasks(dataset: str, task_names: Sequence[str] | None = None) -> list[HarborTask]:
    """Load Harbor tasks from ~/.cache/harbor/tasks/<dataset>/.

    Args:
        dataset: Dataset path under the cache, e.g. ``terminal-bench-2.0/terminal-bench``.
        task_names: Load only these tasks. None loads all of them. Raises
            ``ValueError`` when a name does not exist in the dataset.
    """
    tasks_dir = HARBOR_CACHE_DIR / dataset
    if not tasks_dir.is_dir():
        raise FileNotFoundError(
            f"No Harbor tasks at {tasks_dir}. Download them first, e.g. "
            f"`uvx harbor datasets download terminal-bench@2.0 -o {HARBOR_CACHE_DIR / dataset.split('/')[0]}`"
        )
    wanted = set(task_names) if task_names is not None else None
    tasks: list[HarborTask] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if wanted is not None and task_dir.name not in wanted:
            continue
        tasks.append(
            HarborTask(
                task_name=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text(),
                task_dir=task_dir,
                config=tomllib.loads((task_dir / "task.toml").read_text()),
            )
        )
    if wanted is not None:
        missing = sorted(wanted - {t.task_name for t in tasks})
        if missing:
            raise ValueError(f"Unknown Harbor task(s) in {dataset}: {missing}")
    tasks.sort(key=lambda t: t.task_name)
    return tasks


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
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        sandbox_factory: SandboxFactory | None = None,
        reward_fn: RewardFn | None = None,
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
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.sandbox_factory = sandbox_factory or default_sandbox_factory
        self.reward_fn = reward_fn
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []

        env_dir = self.task.task_dir / "environment"

        # Create renderer (stateless, shared across envs)
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        tests_dir = self.task.task_dir / "tests"

        envs = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(env_dir, self.sandbox_timeout)
            self._sandboxes.append(sandbox)

            bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
            reward_fn = self.reward_fn or HarborReward(
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
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed: %s", e)
        self._sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["harbor"]


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
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    sandbox_factory: SandboxFactory | None = None
    reward_fn: RewardFn | None = None

    def _make_env_group_builders(self, group_size: int) -> list[HarborEnvGroupBuilder]:
        return [
            HarborEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                sandbox_factory=self.sandbox_factory,
                reward_fn=self.reward_fn,
            )
            for task in self.tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_dataset = HarborDataset(
            env_group_builders=self._make_env_group_builders(self.group_size),
            batch_size=self.batch_size,
        )
        eval_dataset = HarborDataset(
            env_group_builders=self._make_env_group_builders(group_size=1),
            batch_size=self.batch_size,
        )
        return train_dataset, eval_dataset
