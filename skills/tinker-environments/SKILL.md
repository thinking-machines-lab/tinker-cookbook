---
name: tinker-environments
description: Guide for defining RL environments — the Env protocol, EnvGroupBuilder, RLDataset, and custom environment creation. Use when the user asks about RL environments, reward functions, or how to define custom tasks for RL training.
---

# RL Environments

RL training requires environments that provide observations and rewards. This skill covers the core protocols and high-level abstractions for building them.

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
        # action is TokensWithLogprobs; extra has metadata like response_hit_length_limit
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=renderer.get_stop_sequences(),
            metrics={"accuracy": 1.0},     # aggregated across batches
            logs={"detail": "..."},         # diagnostic only, not aggregated
        )
```

## EnvGroupBuilder

Creates a group of envs for the same prompt. Advantages are centered within each group (GRPO).

```python
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, Metrics

class MyEnvGroupBuilder(EnvGroupBuilder):
    async def make_envs(self) -> Sequence[Env]:
        """Return group_size envs for the same task."""
        return [MyEnv(problem=self.problem) for _ in range(self.group_size)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Optional final group reward (default returns 0.0 for each)."""
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["my_task"]

    async def cleanup(self) -> None:
        """Release expensive resources (sandboxes, DB connections)."""
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
        # Return (train_dataset, optional_test_dataset)
        return MyDataset(...), None
```

The dataset's `get_batch(batch_idx)` returns `Sequence[EnvGroupBuilder]`.

## ProblemEnv — single-turn answer verification

For tasks where the model answers a question and gets a correctness reward, use `ProblemEnv`. It handles rendering, reward computation (with format penalty), and logging. You only implement 4 methods:

```python
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder

class MyMathEnv(ProblemEnv):
    def __init__(self, question: str, answer: str, **kwargs):
        super().__init__(**kwargs)
        self.question = question
        self.answer = answer

    def get_question(self) -> str:
        return self.question

    def check_answer(self, sample_str: str) -> bool:
        return self.answer.lower() in sample_str.lower()

    def check_format(self, sample_str: str) -> bool:
        return True  # No format requirement

    def get_reference_answer(self) -> str:
        return self.answer
```

Use `ProblemGroupBuilder` to wrap it — it automatically creates `group_size` copies of the env:

```python
builder = ProblemGroupBuilder(
    env_thunk=lambda: MyMathEnv(question=q, answer=a, renderer=renderer),
    num_envs=group_size,
    dataset_name="my_math",
)
```

**Reward formula:** `format_coef * (check_format - 1) + check_answer`. The `format_coef` (default 0.1) penalizes bad format without dominating the correctness signal.

## MessageEnv — multi-turn conversations

For multi-turn environments (tool use, interactive tasks), use `MessageEnv`. It operates at the message level instead of tokens:

```python
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult, EnvFromMessageEnv
from tinker_cookbook.renderers.base import Message

class MyToolEnv(MessageEnv):
    async def initial_observation(self) -> list[Message]:
        """Return initial conversation as renderer messages."""
        return [{"role": "user", "content": "Use the calculator to compute 123 * 456"}]

    async def step(self, message: Message) -> MessageStepResult:
        """Process an assistant message, return reward + next messages."""
        content = message.get("content", "")
        # Parse tool calls, execute, return result
        return MessageStepResult(
            reward=1.0 if "56088" in content else 0.0,
            episode_done=True,
            next_messages=[],  # Append environment response for multi-turn
            metrics={"correct": float("56088" in content)},
        )
```

Wrap with `EnvFromMessageEnv` to bridge to the token-level `Env` interface:

```python
env = EnvFromMessageEnv(
    renderer=renderer,
    message_env=MyToolEnv(),
    max_trajectory_tokens=8192,     # Truncate if context gets too long
    failed_parse_reward=-1.0,       # Penalty for unparseable output
)
```

## Dimension conventions

- `_P` = problems, `_G` = groups, `_T` = tokens, `_D` = datums
- Example: `tokens_P_G_T[p][g][t]` = token `t` of group `g` of problem `p`

## Common pitfalls
- Envs are **single-use** — always create fresh ones via EnvGroupBuilder
- `EnvGroupBuilder` and `RLDatasetBuilder` must be **pickleable** for distributed execution
- Shared resources (sandboxes, connections) should be managed in `cleanup()`, not in env
- For multi-turn envs, use `AsyncConfig(max_steps_off_policy=...)` when env execution is slow
- For more examples, see the [tinker-cookbook recipes](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes)
