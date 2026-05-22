# RL Reference — Environments, Multi-turn, Types

Complete reference for RL environments, multi-turn training, and core types.

## Core protocol: Env

Envs are **single-use** (no reset). Each env handles one episode.

```python
from tinker_cookbook.rl.types import Env, StepResult, Observation, Action, ActionExtra, StopCondition

class MyEnv(Env):
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial prompt (ModelInput) and stop condition."""
        model_input = renderer.build_generation_prompt(messages)
        return model_input, renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Process model output and return reward + next state."""
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=renderer.get_stop_sequences(),
            metrics={"accuracy": 1.0},
            logs={"detail": "..."},
        )
```

## EnvGroupBuilder

Creates a group of envs for the same prompt. Advantages are centered within each group (GRPO).

```python
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, Metrics

class MyEnvGroupBuilder(EnvGroupBuilder):
    async def make_envs(self) -> Sequence[Env]:
        return [MyEnv(problem=self.problem) for _ in range(self.group_size)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["my_task"]

    async def cleanup(self) -> None:
        pass
```

## RLDatasetBuilder

Builds train/test datasets of `EnvGroupBuilder` batches:

```python
@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 128
    group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return MyDataset(...), None
```

The dataset's `get_batch(batch_idx)` returns `Sequence[EnvGroupBuilder]`.

## ProblemEnv — single-turn answer verification

Handles rendering, reward computation (with format penalty), and logging. You implement 4 methods:

```python
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder

class MyMathEnv(ProblemEnv):
    def __init__(self, question: str, answer: str, **kwargs):
        super().__init__(**kwargs)
        self.question, self.answer = question, answer

    def get_question(self) -> str: return self.question
    def check_answer(self, sample_str: str) -> bool: return self.answer.lower() in sample_str.lower()
    def check_format(self, sample_str: str) -> bool: return True
    def get_reference_answer(self) -> str: return self.answer
```

Wrap with `ProblemGroupBuilder`:
```python
builder = ProblemGroupBuilder(
    env_thunk=lambda: MyMathEnv(question=q, answer=a, renderer=renderer),
    num_envs=group_size, dataset_name="my_math",
)
```

**Reward formula:** `format_coef * (check_format - 1) + check_answer` (format_coef default 0.1)

## MessageEnv — multi-turn conversations

For multi-turn environments (tool use, interactive tasks):

```python
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult, EnvFromMessageEnv

class MyToolEnv(MessageEnv):
    async def initial_observation(self) -> list[Message]:
        return [{"role": "user", "content": "Use the calculator to compute 123 * 456"}]

    async def step(self, message: Message) -> MessageStepResult:
        content = message.get("content", "")
        return MessageStepResult(
            reward=1.0 if "56088" in content else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float("56088" in content)},
        )
```

Bridge to Env:
```python
env = EnvFromMessageEnv(
    renderer=renderer, message_env=MyToolEnv(),
    max_trajectory_tokens=8192, failed_parse_reward=-1.0,
)
```

## Dimension conventions

- `_P` = problems, `_G` = groups, `_T` = tokens, `_D` = datums
- Example: `tokens_P_G_T[p][g][t]` = token t of group g of problem p

## Core types reference

- `Action = list[int]` — Token IDs produced by agent
- `Observation = tinker.ModelInput` — Model input fed to model
- `Logprobs = list[float]` — Per-token log-probabilities
- `Metrics = dict[str, float | int]` — Numeric values for logging
- `Logs = dict[str, str | int | float]` — Diagnostic info
- `StepResult` — reward, episode_done, next_observation, next_stop_condition, metrics, logs
- `Transition` — Single (ob, ac, reward) tuple
- `Trajectory` — Complete episode with transitions list
- `TrajectoryGroup` — Group of trajectories with final_rewards_G
- `RolloutError` — Captured error with error_type and error_message

## Full multi-turn example: Calculator tool use

Complete multi-turn RL training script with a calculator tool environment.

```python
import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class CalculatorEnv(MessageEnv):
    """Model must use calc(expr) to solve a math problem."""

    def __init__(self, question: str, answer: float):
        self.question = question
        self.answer = answer
        self.turns = 0

    async def initial_observation(self) -> list[Message]:
        return [
            {"role": "system", "content": "You can use calc(expr) to evaluate math. Give your final answer as: Answer: <number>"},
            {"role": "user", "content": self.question},
        ]

    async def step(self, message: Message) -> MessageStepResult:
        content = message.get("content", "")
        self.turns += 1

        answer_match = re.search(r"Answer:\s*([\d.]+)", content)
        if answer_match:
            correct = abs(float(answer_match.group(1)) - self.answer) < 0.01
            return MessageStepResult(
                reward=1.0 if correct else 0.0,
                episode_done=True, next_messages=[],
                metrics={"correct": float(correct), "turns": self.turns},
            )

        calc_match = re.search(r"calc\((.+?)\)", content)
        if calc_match:
            try:
                result = eval(calc_match.group(1))  # noqa: S307
            except Exception:
                result = "Error: invalid expression"
            return MessageStepResult(
                reward=0.0, episode_done=False,
                next_messages=[{"role": "user", "content": f"Result: {result}"}],
                metrics={},
            )

        return MessageStepResult(reward=0.0, episode_done=True, next_messages=[], metrics={})


@dataclass(frozen=True)
class CalculatorEnvGroupBuilder(EnvGroupBuilder):
    question: str
    answer: float
    group_size: int
    renderer_name: str
    model_name: str

    async def make_envs(self) -> Sequence[EnvFromMessageEnv]:
        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name, tokenizer)
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=CalculatorEnv(self.question, self.answer),
                max_trajectory_tokens=4096,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["calculator"]


class CalculatorDataset(RLDataset):
    def __init__(self, problems, batch_size, group_size, renderer_name, model_name):
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name, self.model_name = renderer_name, model_name

    def __len__(self): return max(1, len(self.problems) // self.batch_size)

    def get_batch(self, batch_idx):
        start = (batch_idx * self.batch_size) % len(self.problems)
        batch = self.problems[start:start + self.batch_size]
        return [CalculatorEnvGroupBuilder(q, a, self.group_size, self.renderer_name, self.model_name) for q, a in batch]


@chz.chz
class CalculatorDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 2
    group_size: int = 4
    renderer_name: str = "qwen3_5"
    model_name: str = "Qwen/Qwen3.5-9B-Base"

    async def __call__(self):
        problems = [("What is 123 * 456?", 56088), ("What is 789 + 321?", 1110)]
        return CalculatorDataset(problems, self.batch_size, self.group_size, self.renderer_name, self.model_name), None


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-9B-Base"
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    max_tokens: int = 1024
    kl_penalty_coef: float = 0.0
    lora_rank: int = 32
    log_path: str = "/tmp/tinker-examples/multiturn"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
    cli_utils.check_log_dir(cli_config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=CalculatorDatasetBuilder(
            group_size=cli_config.group_size,
            renderer_name=renderer_name,
            model_name=cli_config.model_name,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        log_path=cli_config.log_path,
        max_steps=cli_config.max_steps,
    )
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
```

## Key multi-turn patterns

**Context limits:** `EnvFromMessageEnv` handles context overflow. Set `max_trajectory_tokens` to limit total context. When exceeded, `ActionExtra["response_hit_length_limit"]` is `True`.

**Async rollouts** for expensive environments:
```python
from tinker_cookbook.rl.train import AsyncConfig
config = Config(
    ...,
    async_config=AsyncConfig(max_steps_off_policy=4, groups_per_batch=8),
)
```

**Cleanup resources:** Override `cleanup()` on EnvGroupBuilder for sandboxes, DB connections, etc.

## Built-in multi-turn recipes

- **Harbor** — Terminal tasks with sandbox: `tinker_cookbook/recipes/harbor_rl/`
- **Search-R1** — Retrieval with Chroma: `tinker_cookbook/recipes/search_tool/`
- **Multiplayer** — Two-player competitive RL: `tinker_cookbook/recipes/multiplayer_rl/`

## Code references

- `tinker_cookbook/rl/types.py` — All RL types
- `tinker_cookbook/rl/message_env.py` — MessageEnv, EnvFromMessageEnv
- `tinker_cookbook/rl/problem_env.py` — ProblemEnv, ProblemGroupBuilder
- `tinker_cookbook/rl/rollouts.py` — Rollout execution
- `tinker_cookbook/rl/rollout_strategy.py` — FailFast, RetryOnFailure
- `tinker_cookbook/rl/data_processing.py` — Advantage computation
- `tinker_cookbook/rl/metrics.py` — KL computation, metrics
