from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeAlias, TypedDict

from tinker_cookbook.recipes.terminal_rl.terminal_env import AsyncTerminalGymEnv, TerminalAction


class TerminalActionParseError(Exception):
    pass


StopCondition: TypeAlias = list[str] | list[int]


class RendererProtocol(Protocol):
    def get_stop_sequences(self) -> StopCondition: ...

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> Any: ...

    def parse_response(self, tokens: list[int]) -> tuple[dict[str, Any], bool]: ...


class TerminusCommand(TypedDict, total=False):
    keystrokes: str
    duration: float


@dataclass(frozen=True)
class ParsedTerminalPolicyOutput:
    commands: list[TerminusCommand]
    task_complete: bool
    raw: dict[str, Any]


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            # Drop first line ``` or ```json
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def parse_terminus_json_plain(content: str) -> ParsedTerminalPolicyOutput:
    """Parse the JSON specified by terminus-json-plain.txt.

    Expected keys: analysis, plan, commands[{keystrokes, duration}], optional task_complete.
    """

    cleaned = _strip_code_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise TerminalActionParseError(f"Invalid JSON output: {e}") from e

    if not isinstance(data, dict):
        raise TerminalActionParseError("Top-level output must be a JSON object")

    commands = data.get("commands", [])
    if not isinstance(commands, list):
        raise TerminalActionParseError("'commands' must be a list")

    # Normalize to the expected schema to keep pyright happy.
    commands_typed: list[TerminusCommand] = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        typed: TerminusCommand = {}
        keystrokes = cmd.get("keystrokes")
        if isinstance(keystrokes, str):
            typed["keystrokes"] = keystrokes

        duration = cmd.get("duration")
        if isinstance(duration, (int, float)):
            typed["duration"] = float(duration)
        elif isinstance(duration, str):
            try:
                typed["duration"] = float(duration)
            except Exception:
                pass

        commands_typed.append(typed)

    task_complete = bool(data.get("task_complete", False))

    return ParsedTerminalPolicyOutput(
        commands=commands_typed, task_complete=task_complete, raw=data
    )


class HarborTerminalTinkerEnv:
    """Adapt Harbor's AsyncTerminalGymEnv to tinker-cookbook's RL Env interface.

    High-level idea:
    - Observation: a tinker.ModelInput created by a renderer over a message history.
    - Action: model completion tokens (list[int]) that decode into assistant text.
    - The assistant text must be JSON (terminus-json-plain format) describing terminal keystrokes.
    - We execute those keystrokes in the sandbox via AsyncTerminalGymEnv.
    """

    def __init__(
        self,
        *,
        gym_env: AsyncTerminalGymEnv,
        renderer: RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        reserved_generation_tokens: int = 4096,
    ):
        try:  # pragma: no cover
            importlib.import_module("tinker")
            importlib.import_module("tinker_cookbook")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HarborTerminalTinkerEnv requires 'tinker' + 'tinker-cookbook' to be installed."
            ) from e

        self._gym_env = gym_env
        self._renderer = renderer
        # Interpret max_trajectory_tokens as the model context window (in tokens).
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._reserved_generation_tokens = max(0, int(reserved_generation_tokens))
        self._past_messages: list[dict[str, Any]] = []

    def _fits_context_window(self, prompt_tokens: int) -> bool:
        """Return True if prompt + reserved generation tokens fits context window."""
        return (
            int(prompt_tokens) + self._reserved_generation_tokens
        ) <= self._max_trajectory_tokens

    @property
    def stop_condition(self) -> StopCondition:
        return self._renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Any, StopCondition]:
        obs, info = await self._gym_env.reset()
        initial_prompt = str(info.get("initial_prompt") or obs)
        self._past_messages = [{"role": "user", "content": initial_prompt}]

        model_input = self._renderer.build_generation_prompt(self._past_messages)
        if not self._fits_context_window(model_input.length):
            raise ValueError(
                "Initial prompt too long for context window: "
                f"{model_input.length} prompt + {self._reserved_generation_tokens} reserved "
                f"> {self._max_trajectory_tokens}"
            )
        return model_input, self.stop_condition

    async def step(self, action: list[int]) -> Any:
        tinker = importlib.import_module("tinker")
        renderers = importlib.import_module("tinker_cookbook.renderers")
        StepResult = getattr(importlib.import_module("tinker_cookbook.rl.types"), "StepResult")

        # 1) Decode assistant message using renderer.
        message, parse_success = self._renderer.parse_response(action)
        self._past_messages.append(message)

        assistant_text = renderers.get_text_content(message)

        # 2) Parse JSON commands.
        try:
            parsed = parse_terminus_json_plain(assistant_text)
            json_ok = 1.0
        except TerminalActionParseError as e:
            # Treat invalid JSON as a terminal failure.
            # Close the sandbox to avoid resource leaks.
            await self._gym_env.close()
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "parse_success": float(bool(parse_success)),
                    "json_ok": 0.0,
                },
                logs={
                    "error": str(e),
                    "assistant_text": assistant_text[:2000],
                },
            )

        # 3) Execute commands.
        num_commands = 0
        last_terminal_obs = ""
        for cmd in parsed.commands:
            if not isinstance(cmd, dict):
                continue
            keystrokes = cmd.get("keystrokes")
            duration = cmd.get("duration", 1.0)
            if not isinstance(keystrokes, str):
                continue
            try:
                duration_f = float(duration)
            except Exception:
                duration_f = 1.0

            step_res = await self._gym_env.step(
                TerminalAction(
                    keys=keystrokes,
                    block=False,
                    min_timeout_sec=max(0.0, duration_f),
                    max_timeout_sec=180.0,
                    done=False,
                )
            )
            last_terminal_obs = step_res.obs
            num_commands += 1

        # 4) Determine done + reward.
        episode_done = bool(parsed.task_complete)
        reward = 0.0
        rewards_dict: dict[str, float | int] | None = None
        if episode_done:
            rewards_dict, reduced = await self._gym_env.verify()
            reward = float(reduced) if isinstance(reduced, (int, float)) else 0.0

        # 5) Build next observation for continuing episodes.
        if episode_done:
            next_observation = tinker.ModelInput.empty()
            # Close the sandbox to avoid resource leaks.
            await self._gym_env.close()
        else:
            # New user prompt contains fresh terminal state.
            next_prompt = self._gym_env.build_initial_prompt(
                terminal_state=last_terminal_obs or "(no new terminal output)"
            )
            self._past_messages.append({"role": "user", "content": next_prompt})
            next_observation = self._renderer.build_generation_prompt(self._past_messages)
            if not self._fits_context_window(next_observation.length):
                # Close the sandbox to avoid resource leaks.
                await self._gym_env.close()
                return StepResult(
                    reward=0.0,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "parse_success": float(bool(parse_success)),
                        "json_ok": json_ok,
                        "too_long": 1.0,
                    },
                    logs={
                        "next_ob_len": int(next_observation.length),
                        "reserved_generation_tokens": int(self._reserved_generation_tokens),
                        "context_window_tokens": int(self._max_trajectory_tokens),
                    },
                )

        return StepResult(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "parse_success": float(bool(parse_success)),
                "json_ok": json_ok,
                "num_commands": num_commands,
            },
            logs={
                "task_complete": bool(parsed.task_complete),
                "terminal_obs": (last_terminal_obs or "")[:2000],
                "rewards": rewards_dict or {},
            },
        )


class HarborSingleTaskEnvGroupBuilder:
    """Build a group of Harbor terminal envs for a single fixed task.

    This currently supports exactly one fixed task directory and is intended for GRPO-style
    usage: for each rollout group, instantiate N environments for the *same* task (different
    sandbox instances), collect N trajectories, and let the algorithm center rewards within
    the group.

    TODO: Support multiple tasks (e.g., sample task dirs per group/batch).
    """

    def __init__(
        self,
        *,
        task_dir: Path | str,
        group_size: int,
        environment: Any,
        renderer: RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        gym_env_kwargs: dict[str, Any] | None = None,
    ):
        self._task_dir = Path(task_dir)
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError("group_size must be positive")

        self._environment = environment
        self._renderer = renderer
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._gym_env_kwargs = gym_env_kwargs or {}

    async def make_envs(self) -> Sequence[HarborTerminalTinkerEnv]:
        # Import check (mirrors HarborTerminalTinkerEnv behavior)
        importlib.import_module("tinker")
        importlib.import_module("tinker_cookbook")

        envs: list[HarborTerminalTinkerEnv] = []
        for _ in range(self._group_size):
            gym_env = AsyncTerminalGymEnv(
                task_dir=self._task_dir,
                environment=self._environment,
                **self._gym_env_kwargs,
            )
            envs.append(
                HarborTerminalTinkerEnv(
                    gym_env=gym_env,
                    renderer=self._renderer,
                    max_trajectory_tokens=self._max_trajectory_tokens,
                )
            )
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Any], env_group: Sequence[HarborTerminalTinkerEnv]
    ) -> list[tuple[float, dict[str, Any]]]:
        # Default: no additional group reward.
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return ["harbor-terminal", self._task_dir.name]


class HarborSingleTaskRLDataset:
    """A minimal dataset that repeatedly yields EnvGroupBuilders for one task."""

    def __init__(
        self,
        *,
        groups_per_batch: int,
        num_batches: int,
        group_builder_thunk: Callable[[], HarborSingleTaskEnvGroupBuilder],
    ):
        self._groups_per_batch = int(groups_per_batch)
        self._num_batches = int(num_batches)
        self._group_builder_thunk = group_builder_thunk

    def __len__(self) -> int:
        return self._num_batches

    def get_batch(self, index: int) -> list[HarborSingleTaskEnvGroupBuilder]:
        if index < 0 or index >= self._num_batches:
            raise IndexError("batch index out of range")
        return [self._group_builder_thunk() for _ in range(self._groups_per_batch)]


class HarborSingleTaskRLDatasetBuilder:
    """tinker-cookbook-compatible RLDatasetBuilder for a single terminal task.

    This builder currently supports exactly one fixed task directory.

    TODO: Support multiple tasks (e.g., a list of task dirs and a sampling strategy).
    """

    def __init__(
        self,
        *,
        task_dir: Path | str,
        group_size: int,
        environment: Any,
        renderer: RendererProtocol | None = None,
        renderer_name: str | None = None,
        model_name_for_tokenizer: str | None = None,
        groups_per_batch: int = 1,
        num_batches: int = 1,
        max_trajectory_tokens: int = 32 * 1024,
        gym_env_kwargs: Mapping[str, Any] | None = None,
    ):
        self._task_dir = Path(task_dir)
        self._group_size = int(group_size)
        self._environment = environment
        self._renderer = renderer
        self._renderer_name = renderer_name
        self._model_name_for_tokenizer = model_name_for_tokenizer
        self._groups_per_batch = int(groups_per_batch)
        self._num_batches = int(num_batches)
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._gym_env_kwargs: dict[str, Any] = dict(gym_env_kwargs or {})

    async def __call__(self) -> tuple[HarborSingleTaskRLDataset, None]:
        # Return (train_dataset, test_dataset|None)
        renderer = self._renderer
        if renderer is None:
            if not self._renderer_name or not self._model_name_for_tokenizer:
                raise ValueError(
                    "Provide either renderer=... or (renderer_name=..., model_name_for_tokenizer=...)"
                )
            tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
            get_tokenizer = getattr(tokenizer_utils, "get_tokenizer")
            tokenizer = get_tokenizer(self._model_name_for_tokenizer)

            renderers_mod = importlib.import_module("tinker_cookbook.renderers")
            get_renderer = getattr(renderers_mod, "get_renderer")
            renderer = get_renderer(self._renderer_name, tokenizer=tokenizer)

        def thunk() -> HarborSingleTaskEnvGroupBuilder:
            return HarborSingleTaskEnvGroupBuilder(
                task_dir=self._task_dir,
                group_size=self._group_size,
                environment=self._environment,
                renderer=renderer,
                max_trajectory_tokens=self._max_trajectory_tokens,
                gym_env_kwargs=self._gym_env_kwargs,
            )

        return (
            HarborSingleTaskRLDataset(
                groups_per_batch=self._groups_per_batch,
                num_batches=self._num_batches,
                group_builder_thunk=thunk,
            ),
            None,
        )
