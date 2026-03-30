---
name: rl
description: Set up and run reinforcement learning — GRPO with verifiable rewards, custom RL environments, and multi-turn interactive training using the Tinker API. Use when the user wants to do RL training, GRPO, reward-based optimization, custom environments, multi-turn RL, tool-use training, or agentic training — even if they don't say "RL" explicitly.
---

# Reinforcement Learning

Everything for RL training: GRPO, environments, single-turn and multi-turn.

## Minimal GRPO example

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
        batch_size=128, group_size=16,
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

## How GRPO works

For each problem, the model generates `group_size` responses. Rewards are computed, advantages are centered within each group, and the policy is updated.

**Key hyperparameters:**
- `group_size`: Rollouts per prompt (4-16). Advantages centered within each group.
- `groups_per_batch` / `batch_size`: Problems per batch
- `max_tokens`: Maximum generation length
- `learning_rate`: Typically 1e-5 to 4e-5 for RL
- `kl_penalty_coef`: KL penalty against reference model (0.0 = no penalty)
- `loss_fn`: `importance_sampling` (default), `ppo`, `cispo`, `dro`

**Built-in environments:**
- `Gsm8kDatasetBuilder` — Grade-school math (`tinker_cookbook.recipes.math_rl.math_env`)
- `ArithmeticDatasetBuilder` — Simple arithmetic (`tinker_cookbook.recipes.math_rl.math_env`)
- `DeepcoderDatasetBuilder` — Code generation with sandbox (`tinker_cookbook.recipes.code_rl`)

## Custom environments

### ProblemEnv — single-turn answer verification

For tasks where the model answers a question and gets a correctness reward. You implement 4 methods:

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
        return True

    def get_reference_answer(self) -> str:
        return self.answer
```

Wrap with `ProblemGroupBuilder`:
```python
builder = ProblemGroupBuilder(
    env_thunk=lambda: MyEnv(question=q, answer=a, renderer=renderer),
    num_envs=group_size,
)
```

**Reward formula:** `format_coef * (check_format - 1) + check_answer`

### MessageEnv — multi-turn conversations

For interactive environments (tool use, games, multi-step tasks). Operates at message level:

```python
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult, EnvFromMessageEnv
from tinker_cookbook.renderers.base import Message

class MyToolEnv(MessageEnv):
    async def initial_observation(self) -> list[Message]:
        return [{"role": "user", "content": "Use calc(expr) to solve: 123 * 456"}]

    async def step(self, message: Message) -> MessageStepResult:
        content = message.get("content", "")
        # Parse tool calls, execute, return result
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
        )
```

Bridge to token-level Env interface:
```python
env = EnvFromMessageEnv(
    renderer=renderer, message_env=MyToolEnv(),
    max_trajectory_tokens=8192, failed_parse_reward=-1.0,
)
```

For the complete Env protocol, EnvGroupBuilder, RLDatasetBuilder, and full multi-turn examples, read `references/environments.md`.

For a full multi-turn training script with calculator tool use, read `references/multiturn.md`.

## Async and concurrency (critical for throughput)

RL training has the most concurrency opportunities in Tinker. Writing sequential code where async is possible is the #1 throughput killer.

### Async training loop

The built-in `rl/train.py` already overlaps rollouts with training. For custom loops, follow the same pattern — submit async calls back-to-back before awaiting:

```python
# CORRECT: overlap training with next batch's rollouts
fb_future = tc.forward_backward_async(data=training_data, loss_fn="importance_sampling")
optim_future = tc.optim_step_async(adam_params=adam_params)
# Start rollouts for next batch while GPU trains
next_rollouts = await do_group_rollout(...)
fb_result = fb_future.result()
optim_result = optim_future.result()

# WRONG: sequential = wastes GPU cycles
fb_result = tc.forward_backward_async(data=data, loss_fn="importance_sampling").result()
tc.optim_step_async(adam_params=adam_params).result()
next_rollouts = await do_group_rollout(...)  # GPU idle during rollouts!
```

### Async evaluation

When evaluating multiple problems or computing metrics, run evaluations concurrently:

```python
# CORRECT: evaluate concurrently
import asyncio
tasks = [evaluate_problem(sc, problem) for problem in test_problems]
results = await asyncio.gather(*tasks)

# WRONG: evaluate sequentially
results = []
for problem in test_problems:
    result = await evaluate_problem(sc, problem)  # Each waits for the last
    results.append(result)
```

### AsyncConfig for expensive environments

For environments with slow execution (sandboxes, API calls, tool use), use `AsyncConfig` to overlap sampling with training:

```python
from tinker_cookbook.rl.train import AsyncConfig
config = train.Config(
    ...,
    async_config=AsyncConfig(max_steps_off_policy=4, groups_per_batch=128),
)
```

This lets the training loop use slightly off-policy data while new rollouts are being collected, dramatically improving throughput for slow environments.

### Sampler desync

After saving weights, always create a **new** SamplingClient. A stale client silently samples from outdated weights:

```python
# After training step
tc.save_weights_for_sampler(name="step_100")
sc = svc.create_sampling_client(model_path=saved_path)  # New client — old one still uses previous weights
# Do NOT reuse the old sc — it points at old weights
```

**Error tolerance** for flaky environments:
```python
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
config = train.Config(..., rollout_error_tolerance=RetryOnFailure(max_retries=5))
```

## Common pitfalls

- **Sequential API calls**: The #1 performance mistake. Always use `_async` variants and overlap GPU work with rollouts/data prep. Never chain `.result()` calls sequentially.
- **Sampler desync**: Create a **new** SamplingClient after every weight save. Reusing a stale client silently samples from old weights.
- `Env` objects are **single-use** — always create fresh envs via builder
- Advantages are centered within each group — `group_size` matters for variance reduction
- `max_tokens` too small truncates reasoning; too large wastes compute
- `EnvGroupBuilder` and `RLDatasetBuilder` must be **pickleable**
- Start with small `groups_per_batch` for debugging, scale up for real runs
- Multi-turn: use `AsyncConfig` when env execution is slow
- Multi-turn: `kl_penalty_coef=0.0` is common to allow tool-use exploration
- Set `max_trajectory_tokens` to avoid unbounded context growth

## Code references

- `tinker_cookbook/rl/train.py` — RL training loop and Config
- `tinker_cookbook/rl/types.py` — Env, EnvGroupBuilder, RLDataset, Trajectory
- `tinker_cookbook/rl/message_env.py` — MessageEnv, EnvFromMessageEnv
- `tinker_cookbook/rl/problem_env.py` — ProblemEnv, ProblemGroupBuilder
- `tinker_cookbook/rl/rollout_strategy.py` — FailFast, RetryOnFailure
- `tinker_cookbook/rl/rollouts.py` — Rollout execution
- `tinker_cookbook/recipes/math_rl/` — Math RL recipes
- `tinker_cookbook/recipes/code_rl/` — Code RL recipes
- `tinker_cookbook/recipes/harbor_rl/` — Terminal task RL
- `tinker_cookbook/recipes/multiplayer_rl/` — Multi-agent RL
- `tinker_cookbook/recipes/search_tool/` — Search/RAG RL
