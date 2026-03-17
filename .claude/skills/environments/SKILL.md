---
name: environments
description: Guide for defining RL environments — the Env protocol, EnvGroupBuilder, RLDataset, and custom environment creation. Use when the user asks about RL environments, reward functions, or how to define custom tasks for RL training.
---

# RL Environments

RL training requires environments that provide observations and rewards. This skill covers how to define and use them.

## Reference

Read these for details:
- `tinker_cookbook/rl/types.py` — Env, EnvGroupBuilder, RLDatasetBuilder, Trajectory
- `docs/rl/rl-envs.mdx` — Custom environments guide
- `tinker_cookbook/recipes/math_rl/math_env.py` — Math environment example
- `tinker_cookbook/recipes/harbor_rl/harbor_env.py` — Multi-turn sandbox environment
- `tinker_cookbook/rl/message_env.py` — Message-based environment interface
- `CONTRIBUTING.md` — Env lifecycle and design conventions

## Core types

### Env (single-use, no reset)

```python
from tinker_cookbook.rl.types import Env, Observation, Action, StepResult, StopCondition

class MyEnv(Env):
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial prompt and stop condition."""
        model_input = renderer.build_generation_prompt(messages)
        stop = renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action: Action) -> StepResult:
        """Process model output and return next observation + reward."""
        # action is TokensWithLogprobs (tokens + logprobs)
        return StepResult(
            observation=next_model_input,
            stop_condition=stop,
            reward=reward_value,
            episode_done=True,
            metrics={"accuracy": 1.0},
        )
```

**Important:** Env objects are **single-use** — no reset method. Create fresh envs via EnvGroupBuilder each batch.

### EnvGroupBuilder

Creates a group of envs for the same prompt/task. Advantages are centered within each group (GRPO).

```python
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup

class MyEnvGroupBuilder(EnvGroupBuilder):
    async def make_envs(self) -> Sequence[Env]:
        """Return group_size envs for the same task."""
        return [MyEnv(problem=self.problem) for _ in range(self.group_size)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute final rewards for each trajectory in the group."""
        return [(env.reward, {"solved": env.reward > 0}) for env in env_group]

    def logging_tags(self) -> list[str]:
        return ["my_task"]
```

### RLDatasetBuilder

Builds train/test datasets of EnvGroupBuilders:

```python
@chz.chz
class MyDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 128
    group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Return (train_dataset, optional_test_dataset)
        ...
```

## Key data types

```python
@dataclass
class Transition:
    ob: Observation       # ModelInput
    ac: TokensWithLogprobs  # Action with logprobs
    reward: float
    episode_done: bool

@dataclass
class Trajectory:
    transitions: list[Transition]
    final_ob: Observation

@dataclass
class TrajectoryGroup:
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: list[Metrics]
```

## Patterns

### Single-turn (math, classification)
Model generates one response, gets a reward. See `recipes/math_rl/math_env.py`.

### Multi-turn (tool use, sandbox)
Model generates, environment responds, repeat. See `recipes/harbor_rl/harbor_env.py` and `docs/rl/sequence-extension.mdx` for KV-cache support.

### Multiplayer (games)
Group of envs represents a game — envs within the group interact. See `recipes/multiplayer_rl/text_arena/env.py`.

### Preference-based (RLHF)
Group of envs generates completions, preference model scores pairs. See `tinker_cookbook/rl/preference_envs.py`.

## Pluggable rollout executor

For scaling rollout collection, `train.main()` accepts an optional `rollout_executor` parameter:

```python
from concurrent.futures import ProcessPoolExecutor
from tinker_cookbook.rl.train import main

await main(config, rollout_executor=ProcessPoolExecutor(max_workers=4))
```

EnvGroupBuilders must be **pickleable** for distributed execution. Test with `tinker_cookbook/rl/builder_pickle_test.py`.

## Dimension conventions

- `_P` = problems (different prompts/tasks)
- `_G` = groups (multiple rollouts per problem)
- `_T` = tokens (sequence position)
- `_D` = datums (training data items)

Example: `tokens_P_G_T[p][g][t]` = token `t` of group `g` of problem `p`.

## Common pitfalls
- Envs are **single-use** — always create fresh ones via EnvGroupBuilder
- Advantages are centered within each group — `group_size` affects variance reduction
- EnvGroupBuilders must be pickleable for distributed rollout execution
- Shared resources (DB connections, sandboxes) should be managed by the builder, not the env
- For multi-turn envs, use `max_steps_off_policy` for async rollouts when env execution is slow
