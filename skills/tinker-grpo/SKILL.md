---
name: tinker-grpo
description: Set up and run reinforcement learning with verifiable rewards (RLVR/GRPO) for math, code, or custom environments using the Tinker API. Use when the user wants to do RL training, GRPO, reward-based optimization, or train with verifiable rewards.
---

# Group Relative Policy Optimization (GRPO / RL)

Help the user set up and run RL training with verifiable rewards using the Tinker API.

## Key concepts

**How GRPO works:** For each problem, the model generates `group_size` responses. Rewards are computed, advantages are centered within each group, and the policy is updated.

**Key hyperparameters:**
- `group_size`: Rollouts per prompt (4–16). Advantages are centered within each group.
- `batch_size` / `groups_per_batch`: Problems per batch
- `max_tokens`: Maximum generation length
- `learning_rate`: Typically 1e-5 to 4e-5 for RL
- `kl_penalty_coef`: KL penalty against reference model (0.0 = no penalty)
- `temperature`: Sampling temperature (default 1.0)

**Loss functions:** `importance_sampling` (default), `ppo`, `cispo`, `dro` — set via `loss_fn` parameter.

**Built-in environments:**
- `Gsm8kDatasetBuilder` — Grade-school math (`tinker_cookbook.recipes.math_rl.math_env`)
- `ArithmeticDatasetBuilder` — Simple arithmetic (`tinker_cookbook.recipes.math_rl.math_env`)
- `DeepcoderDatasetBuilder` — Code generation with sandbox (`tinker_cookbook.recipes.code_rl`)

## Minimal working example

This is a complete, runnable RL script using GSM8K math:

```python
import asyncio
import sys

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = Gsm8kDatasetBuilder(
        batch_size=128,
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )
    return chz.Blueprint(train.Config).apply({
        "model_name": model_name,
        "renderer_name": renderer_name,
        "log_path": "/tmp/tinker-examples/rl_basic",
        "dataset_builder": builder,
        "learning_rate": 4e-5,
        "max_tokens": 256,
        "eval_every": 0,
    })


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
```

Run it: `python my_rl.py` or `python my_rl.py learning_rate=1e-5 group_size=8`

## Custom environment with ProblemEnv

For single-turn answer-verification tasks, subclass `ProblemEnv` — it handles rendering, reward computation, and logging. You only implement 4 methods:

```python
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder

class MyEnv(ProblemEnv):
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

Then create an `RLDatasetBuilder` that produces `ProblemGroupBuilder` batches:

```python
from collections.abc import Sequence
import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 64
    group_size: int = 8
    renderer_name: str = "llama3"
    model_name_for_tokenizer: str = "meta-llama/Llama-3.1-8B"

    async def __call__(self) -> "MyDataset":
        problems = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        return MyDataset(problems, self.batch_size, self.group_size, renderer), None

class MyDataset:
    def __init__(self, problems, batch_size, group_size, renderer):
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def __len__(self) -> int:
        return len(self.problems) // self.batch_size

    def get_batch(self, batch_idx: int) -> Sequence[EnvGroupBuilder]:
        start = (batch_idx * self.batch_size) % len(self.problems)
        batch = self.problems[start:start + self.batch_size]
        return [
            ProblemGroupBuilder(
                env_thunk=lambda q=q, a=a: MyEnv(
                    question=q, answer=a, renderer=self.renderer
                ),
                num_envs=self.group_size,
            )
            for q, a in batch
        ]
```

For multi-turn and tool-use environments, see `/tinker-multiturn-rl` and `/tinker-environments`.

## Customization

**Async training** for expensive environments (overlaps sampling and training):
```python
from tinker_cookbook.rl.train import AsyncConfig
# Add to config: async_config=AsyncConfig(max_steps_off_policy=4, groups_per_batch=128)
```

**Error tolerance** for flaky environments:
```python
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
# Add to config: rollout_error_tolerance=RetryOnFailure(max_retries=5)
```

For advanced recipes (code RL, rubric grading, multiple environments), see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes).

## Common pitfalls
- `Env` objects are single-use — always create fresh envs via builder
- Advantages are centered within each group — `group_size` matters for variance reduction
- `max_tokens` too small truncates reasoning; too large wastes compute
- Start with small `groups_per_batch` for debugging, scale up for real runs
- `EnvGroupBuilder` and `RLDatasetBuilder` must be pickleable (use config strings + lazy construction)
