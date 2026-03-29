# RL Environments

Complete reference for the Env protocol, EnvGroupBuilder, and RLDataset.

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

## Code references

- `tinker_cookbook/rl/types.py` — All RL types
- `tinker_cookbook/rl/message_env.py` — MessageEnv, EnvFromMessageEnv
- `tinker_cookbook/rl/problem_env.py` — ProblemEnv, ProblemGroupBuilder
- `tinker_cookbook/rl/rollouts.py` — Rollout execution
- `tinker_cookbook/rl/rollout_strategy.py` — FailFast, RetryOnFailure
- `tinker_cookbook/rl/data_processing.py` — Advantage computation
- `tinker_cookbook/rl/metrics.py` — KL computation, metrics
