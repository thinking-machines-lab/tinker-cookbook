"""Harbor environment, dataset, and dataset builder for RL training."""

from __future__ import annotations

import logging
import shlex
import tomllib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import CloudSandboxBackend, SandboxInterface
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


async def _cleanup_after_setup_failure(sandbox: SandboxInterface) -> None:
    """Terminate a sandbox its factory cannot return, since nothing else would."""
    try:
        await sandbox.cleanup()
    except Exception as e:
        logger.warning("Sandbox cleanup failed: %s", e)


def _task_config(env_dir: Path) -> dict[str, Any]:
    """Return the parsed task.toml next to env_dir, or an empty dict when there is none."""
    task_toml = env_dir.parent / "task.toml"
    if not task_toml.exists():
        return {}
    return tomllib.loads(task_toml.read_text())


_SIZE_UNITS_MB = {"K": 1 / 1024, "M": 1, "G": 1024}


def _resource_gib(environment: dict[str, Any], name: str) -> float | None:
    """Return the memory or storage a task declares, in GiB, or None when it declares none.

    Reads the ``memory_mb`` and ``storage_mb`` fields alongside the deprecated ``memory``
    and ``storage`` strings such as ``"2G"`` or ``"512M"``, which Harbor still accepts.
    """
    mb = environment.get(f"{name}_mb")
    if mb is None:
        size = environment.get(name)
        if not isinstance(size, str):
            return None
        size = size.strip().upper()
        if size[-1:] not in _SIZE_UNITS_MB:
            raise ValueError(f"{name} = {size!r} is not a size such as '2G' or '512M'.")
        mb = float(size[:-1]) * _SIZE_UNITS_MB[size[-1]]
    return mb / 1024 if mb else None


def _network_policy(table: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Return the (network_mode, allowed_hosts) a task.toml table sets, if it sets a network_mode."""
    network_mode = table.get("network_mode")
    if network_mode is None:
        return None
    return network_mode, list(table.get("allowed_hosts") or [])


async def default_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Create a Modal sandbox from a task environment directory.

    A workdir the task declares is created when the image lacks it, as Harbor does.

    Args:
        env_dir: Path to the task's environment/ directory (must contain a Dockerfile).
        timeout: Sandbox lifetime in seconds.

    Raises:
        SandboxError: If the task's workdir cannot be created.
    """
    import modal

    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    dockerfile_path = env_dir / "Dockerfile"
    image = modal.Image.from_dockerfile(path=str(dockerfile_path), context_dir=str(env_dir))
    sandbox = await ModalSandbox.create(image=image, timeout=timeout)
    workdir = _task_config(env_dir).get("environment", {}).get("workdir")
    if workdir:
        try:
            result = await sandbox.run_command(f"mkdir -p {shlex.quote(workdir)}")
            if result.exit_code != 0:
                raise SandboxError(
                    f"Failed to create the working directory {workdir}: {result.stderr}"
                )
        except BaseException:
            await _cleanup_after_setup_failure(sandbox)
            raise
    return sandbox


def _holds_runtime_files(env_dir: Path) -> bool:
    """True when env_dir holds files for the task's working directory rather than a Dockerfile."""
    return env_dir.is_dir() and not (env_dir / "Dockerfile").exists() and any(env_dir.iterdir())


async def sail_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Create a Sail sandbox from a task environment directory.

    Boots the prebuilt image named in the task's task.toml when there is one,
    and otherwise builds the Dockerfile in env_dir. A task that names a prebuilt
    image and keeps no Dockerfile gets the files in env_dir uploaded to its
    working directory, as Harbor does. The Sailbox gets the smallest size that
    covers the cpus, memory, and storage the task declares, an egress policy
    that enforces the task's network_mode, and the workdir the task declares,
    created as root when the image lacks it. Pulling or building the image gets
    the build_timeout_sec the task declares, Harbor's default of 600 seconds
    when it does not.

    Args:
        env_dir: Path to the task's environment/ directory.
        timeout: Sandbox lifetime in seconds.

    Raises:
        ValueError: If the task needs more than the factory provides. It runs
            single-container Linux CPU tasks as one episode under one egress
            policy, starts no service in them, and grades them in the agent's
            Sailbox, so it rejects a docker-compose.yaml in env_dir, a Windows
            image, a request for GPUs or a TPU, a healthcheck, steps, a verifier
            in a separate environment, and an agent or verifier network policy
            that differs from the environment's.
    """
    import sail

    from tinker_cookbook.sandbox.sail_sandbox import SailSandbox

    compose_file = env_dir / "docker-compose.yaml"
    if compose_file.exists():
        raise ValueError(
            f"{compose_file} needs Docker Compose, which sail_sandbox_factory does not run."
        )
    task_toml = env_dir.parent / "task.toml"
    config = _task_config(env_dir)
    environment = config.get("environment", {})
    if environment.get("os", "").lower() == "windows":
        raise ValueError(f"{task_toml} declares a Windows task, which Sailboxes do not run.")
    if environment.get("gpus") or environment.get("tpu"):
        raise ValueError(f"{task_toml} asks for an accelerator, which Sailboxes do not provide.")
    if "healthcheck" in environment:
        raise ValueError(
            f"{task_toml} declares a healthcheck, which sail_sandbox_factory does not run: "
            "a Sailbox runs neither the image's ENTRYPOINT nor its CMD, so no service starts "
            "for the check to wait on."
        )
    if config.get("steps"):
        raise ValueError(
            f"{task_toml} is a multi-step task, which sail_sandbox_factory does not run."
        )
    verifier = config.get("verifier", {})
    # A [verifier.environment] table implies environment_mode = "separate" in Harbor.
    default_mode = "separate" if "environment" in verifier else "shared"
    if verifier.get("environment_mode", default_mode) == "separate":
        raise ValueError(
            f"{task_toml} runs its verifier in a separate environment, "
            "which sail_sandbox_factory does not provide."
        )
    policy = _network_policy(environment)
    if policy is None:
        # Harbor still honors the deprecated allow_internet flag when a task sets no
        # network_mode.
        policy = ("no-network" if environment.get("allow_internet") is False else "public", [])
    for phase in ("agent", "verifier"):
        override = _network_policy(config.get(phase, {}))
        if override is not None and override != policy:
            raise ValueError(
                f"{task_toml} gives the {phase} phase a network policy of its own, which "
                "sail_sandbox_factory cannot switch to after the Sailbox starts."
            )
    network_mode, allowed_hosts = policy
    egress_policy = None
    if network_mode == "no-network":
        egress_policy = sail.EgressPolicy.no_network()
    elif network_mode == "allowlist":
        egress_policy = (
            sail.EgressPolicy.allow_only(*allowed_hosts)
            if allowed_hosts
            else sail.EgressPolicy.no_egress()
        )
    docker_image = environment.get("docker_image")
    if isinstance(docker_image, str):
        image = sail.Image.from_registry(docker_image)
    else:
        image = await sail.Image.from_dockerfile.aio(env_dir / "Dockerfile", context_dir=env_dir)
    sizing = sail.Sailbox.size_for(
        cpus=environment.get("cpus"),
        memory_gib=_resource_gib(environment, "memory"),
        disk_gib=_resource_gib(environment, "storage"),
    )
    sandbox = await SailSandbox.create(
        image=image,
        image_build_timeout=round(environment.get("build_timeout_sec", 600)),
        max_lifetime_seconds=timeout,
        size=sizing.size,
        memory_limit_gib=sizing.memory_limit_gib,
        disk_limit_gib=sizing.disk_limit_gib,
        egress_policy=egress_policy,
    )
    try:
        workdir = environment.get("workdir")
        if workdir:
            # fs.mkdir creates missing parents and leaves an existing directory alone.
            await sandbox.sailbox.fs.mkdir.aio(workdir)
        if isinstance(docker_image, str) and _holds_runtime_files(env_dir):
            workdir = workdir or (await sandbox.run_command("pwd")).stdout.strip()
            await sandbox.sailbox.fs.upload_dir.aio(env_dir, workdir)
    except BaseException:
        await _cleanup_after_setup_failure(sandbox)
        raise
    return sandbox


def sandbox_factory_for(backend: CloudSandboxBackend) -> SandboxFactory:
    """Return the built-in factory for a cloud sandbox backend."""
    if backend == "modal":
        return default_sandbox_factory
    return sail_sandbox_factory


@dataclass(frozen=True)
class HarborTask:
    """A single Harbor terminal-bench task."""

    task_name: str
    instruction: str
    task_dir: Path  # Convention: environment/Dockerfile, tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def workdir(self) -> str:
        """The directory the agent's commands run in.

        The workdir the task declares under [environment], as in Harbor, and "/"
        when it declares none. The built-in sandbox factories create a declared
        workdir when the image lacks it.
        """
        return self.config.get("environment", {}).get("workdir") or "/"


def load_harbor_tasks(dataset: str) -> list[HarborTask]:
    """Load Harbor tasks from ~/.cache/harbor/tasks/<dataset>/."""
    tasks_dir = HARBOR_CACHE_DIR / dataset
    tasks: list[HarborTask] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tasks.append(
            HarborTask(
                task_name=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text(),
                task_dir=task_dir,
                config=tomllib.loads((task_dir / "task.toml").read_text()),
            )
        )
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
        self.sandbox_factory = (
            sandbox_factory if sandbox_factory is not None else default_sandbox_factory
        )
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

            bash_tool = HarborBashTool(
                sandbox, command_timeout=self.command_timeout, workdir=self.task.workdir
            )
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
