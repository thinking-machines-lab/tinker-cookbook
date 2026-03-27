---
name: tinker-multiturn-rl
description: Set up and run multi-turn RL training for interactive environments (terminal tasks, tool use, search/RAG, games) using the Tinker API. Use when the user wants multi-turn RL, agentic training, tool-use RL, or interactive environment training.
---

# Multi-Turn RL Training

Help the user set up RL training for multi-turn interactive environments using the Tinker API.

## Key concepts

**MessageEnv** is the high-level abstraction for multi-turn environments. It operates at the message level (not tokens), and `EnvFromMessageEnv` bridges it to the token-level `Env` interface used by the training loop.

```python
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult, EnvFromMessageEnv
```

**Key parameters for multi-turn RL:**
- `max_tokens`: Max tokens per generation step
- `max_trajectory_tokens`: Total context budget across all turns
- `kl_penalty_coef`: Often 0.0 for multi-turn (allow exploration of tool use)
- `AsyncConfig(max_steps_off_policy=N)`: Overlap sampling and training for slow envs

**Built-in multi-turn recipes:**
- Harbor — Terminal tasks with sandbox execution (`tinker_cookbook.recipes.harbor_rl`)
- Search-R1 — Retrieval with Chroma vector DB (`tinker_cookbook.recipes.search_tool`)
- Multiplayer games — Two-player competitive RL (`tinker_cookbook.recipes.multiplayer_rl`)

## Minimal working example

Here is a complete multi-turn RL environment and training script. This example trains a model to use a calculator tool:

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

        # Check for final answer
        answer_match = re.search(r"Answer:\s*([\d.]+)", content)
        if answer_match:
            correct = abs(float(answer_match.group(1)) - self.answer) < 0.01
            return MessageStepResult(
                reward=1.0 if correct else 0.0,
                episode_done=True,
                next_messages=[],
                metrics={"correct": float(correct), "turns": self.turns},
            )

        # Check for tool call
        calc_match = re.search(r"calc\((.+?)\)", content)
        if calc_match:
            try:
                result = eval(calc_match.group(1))  # noqa: S307
            except Exception:
                result = "Error: invalid expression"
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=[{"role": "user", "content": f"Result: {result}"}],
                metrics={},
            )

        # No tool call or answer — end with 0 reward
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
    renderer_name: str = "llama3"
    model_name: str = "meta-llama/Llama-3.1-8B"

    async def __call__(self):
        problems = [("What is 123 * 456?", 56088), ("What is 789 + 321?", 1110)]
        return CalculatorDataset(problems, self.batch_size, self.group_size, self.renderer_name, self.model_name), None


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
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

## Customization

**Context limits:** `EnvFromMessageEnv` handles context overflow automatically. Set `max_trajectory_tokens` to limit total context. When exceeded, `ActionExtra["response_hit_length_limit"]` is set to `True`.

**Async rollouts** for expensive environments (sandbox execution, API calls):
```python
from tinker_cookbook.rl.train import AsyncConfig
config = Config(
    ...,
    async_config=AsyncConfig(max_steps_off_policy=4, groups_per_batch=8),
)
```

**Cleanup resources:** Override `cleanup()` on your `EnvGroupBuilder` to release sandboxes, DB connections, etc. It runs after rollouts regardless of success/failure.

For advanced recipes (Harbor sandbox, Search-R1, multiplayer games), see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes).

## Common pitfalls
- Multi-turn envs are expensive — start with small `groups_per_batch` (4–8)
- Use `AsyncConfig` for async rollouts when env execution is slow
- `kl_penalty_coef=0.0` is common for multi-turn to allow tool-use exploration
- `EnvGroupBuilder` must be **pickleable** — use config strings + lazy construction in `make_envs()`
- Set `max_trajectory_tokens` to avoid unbounded context growth
