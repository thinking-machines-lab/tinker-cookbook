import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 05: The Cookbook's RL Abstractions

    In tutorial 04, we wrote a GRPO training loop from scratch: sample completions, grade them, compute advantages, build datums, train. That works, but every new task would repeat the same boilerplate.

    The cookbook provides standard types that separate concerns:
    - **`Env`** -- task logic (prompts and rewards)
    - **`EnvGroupBuilder`** -- batching multiple rollouts per problem
    - **`RLDataset`** -- iterating over problems
    - **`compute_advantages`** / **`assemble_training_data`** -- reusable data processing

    This tutorial shows how the same GSM8K task from tutorial 04 maps onto these types.
    """)
    return


@app.cell
def _():
    from collections.abc import Sequence
    from functools import partial

    import tinker

    from tinker_cookbook import renderers
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.hyperparam_utils import get_lr
    from tinker_cookbook.rl.data_processing import (
        assemble_training_data,
        compute_advantages,
        remove_constant_reward_groups,
    )
    from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
    from tinker_cookbook.rl.rollouts import do_group_rollout
    from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return (
        EnvGroupBuilder,
        ProblemGroupBuilder,
        RLDataset,
        Sequence,
        TinkerTokenCompleter,
        TrajectoryGroup,
        assemble_training_data,
        compute_advantages,
        do_group_rollout,
        get_lr,
        get_tokenizer,
        partial,
        remove_constant_reward_groups,
        renderers,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Env protocol

    An `Env` represents a single RL episode. It has two methods:
    - **`initial_observation()`** -- returns the prompt as `ModelInput` + stop conditions (e.g., stop sequences)
    - **`step(action)`** -- receives the model's output tokens, returns a `StepResult` with reward, done flag, and next observation

    Envs are **single-use**: one episode per instance, then discard. This keeps state management simple.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ProblemEnv -- a shortcut for single-step problems

    Most RLVR tasks (math, code, format compliance) follow one pattern: show a question, get one response, grade it. `ProblemEnv` is a base class that handles the `initial_observation`/`step` plumbing. You just implement four methods:

    ```python
    class ProblemEnv(Env):
        def get_question(self) -> str: ...
        def check_answer(self, text: str) -> bool: ...
        def check_format(self, text: str) -> bool: ...
        def get_reference_answer(self) -> str: ...
    ```

    The reward formula is: `format_coef * (correct_format - 1) + correct_answer`. This gives a small penalty for bad formatting and a large reward for correct answers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Looking at MathEnv

    `MathEnv` (in `tinker_cookbook/recipes/math_rl/math_env.py`) extends `ProblemEnv` for math problems. The implementation is compact -- just four methods plus a constructor:

    - `get_question()` -- returns the problem text with a "Write your answer in `\boxed{}` format" suffix
    - `check_format()` -- checks if the response contains a `\boxed{}` expression
    - `check_answer()` -- extracts the boxed answer and grades it against the reference
    - `get_reference_answer()` -- returns the ground truth for logging

    Let's create a `MathEnv` and walk through its lifecycle manually.
    """)
    return


@app.cell
async def _(get_tokenizer, renderers):
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    MODEL_NAME = "Qwen/Qwen3.5-4B"
    RENDERER_NAME = "qwen3_5"

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)

    # Create a single MathEnv instance
    env = MathEnv(
        problem="What is 2 + 3?",
        answer="5",
        renderer=renderer,
    )

    # initial_observation() returns the prompt tokens and stop condition
    ob, stop_cond = await env.initial_observation()
    print(f"Observation length: {ob.length} tokens")
    print(f"Stop condition: {stop_cond}")
    print(f"Question: {env.get_question()}")
    return MODEL_NAME, MathEnv, env, renderer, tokenizer


@app.cell
async def _(env, tokenizer):
    # Simulate calling step() with a correct answer (as token IDs)
    # In real training, the action comes from the model's sampled tokens.
    # Here we just encode a string to show the interface.
    fake_response = "The answer is \\boxed{5}"
    fake_action = tokenizer.encode(fake_response)

    step_result = await env.step(fake_action)
    print(f"Reward: {step_result.reward}")
    print(f"Episode done: {step_result.episode_done}")
    print(f"Metrics: {step_result.metrics}")

    # The env is now spent -- don't reuse it. Create a new one for the next episode.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EnvGroupBuilder -- creating groups for GRPO

    In tutorial 04, we sampled `group_size` completions per problem manually. `EnvGroupBuilder` formalizes this pattern:
    - **`make_envs()`** -- returns a list of fresh `Env` instances (one per rollout in the group)
    - **`compute_group_rewards()`** -- optional group-level reward (default 0.0, added on top of per-step rewards)
    - **`cleanup()`** -- release resources after rollouts complete

    For single-step problems, `ProblemGroupBuilder` is a simple wrapper that calls an env factory function N times.
    """)
    return


@app.cell
async def _(MathEnv, ProblemGroupBuilder, partial, renderer):
    GROUP_SIZE = 4
    _group_builder = ProblemGroupBuilder(
        env_thunk=partial(MathEnv, "What is 2 + 3?", "5", renderer), num_envs=GROUP_SIZE
    )
    # ProblemGroupBuilder takes an env factory (a callable that returns a fresh ProblemEnv)
    # and the number of envs to create per group.
    envs = await _group_builder.make_envs()
    print(f"Created {len(envs)} envs for this group")
    # make_envs() creates GROUP_SIZE independent env instances
    print("Each env is an independent episode of the same problem")
    return (GROUP_SIZE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rollouts -- running the sampling loop

    `do_group_rollout()` handles the full rollout lifecycle for one group:
    1. Calls `make_envs()` to create the env instances
    2. Runs `initial_observation()` on each env
    3. Samples from the policy, calls `step()` with the tokens (repeating for multi-step envs)
    4. Calls `compute_group_rewards()` for final group-level rewards
    5. Calls `cleanup()` to release resources

    It returns a `TrajectoryGroup` containing all trajectories and their rewards. Let's run a real rollout against a model.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key  # noqa: B018
    return (api_key,)


@app.cell
async def _(
    GROUP_SIZE,
    MODEL_NAME,
    MathEnv,
    ProblemGroupBuilder,
    TinkerTokenCompleter,
    TrajectoryGroup,
    api_key,
    do_group_rollout,
    mo,
    partial,
    renderer,
    tinker,
):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    LORA_RANK = 32
    MAX_TOKENS = 512
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )
    _sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    _policy = TinkerTokenCompleter(_sampling_client, max_tokens=MAX_TOKENS, temperature=1.0)
    _group_builder = ProblemGroupBuilder(
        env_thunk=partial(
            MathEnv,
            "What is 15 * 23?",
            "345",
            renderer,
            convo_prefix=MathEnv.standard_fewshot_prefix(),
        ),
        num_envs=GROUP_SIZE,
    )
    # Create a TokenCompleter -- this wraps the sampling client with max_tokens and temperature
    traj_group: TrajectoryGroup = await do_group_rollout(_group_builder, _policy)
    rewards = traj_group.get_total_rewards()
    # Build a group for a single math problem
    print(f"Rewards per trajectory: {rewards}")
    print(f"Number of trajectories: {len(traj_group.trajectories_G)}")
    for i, (traj, reward) in enumerate(zip(traj_group.trajectories_G, rewards)):
        n_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
        # Run the rollout
        # Inspect results
        print(f"  Trajectory {i}: reward={reward:.1f}, response_tokens={n_tokens}")
    return MAX_TOKENS, training_client, traj_group


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data processing -- advantages and datums

    In tutorial 04, we manually centered rewards and built `Datum` objects with padded logprobs and advantages. The cookbook provides two functions that replace all of that:

    - **`compute_advantages()`** -- centers rewards within each group (same GRPO logic as tutorial 04)
    - **`assemble_training_data()`** -- converts `TrajectoryGroup`s + advantages into `Datum` objects ready for `forward_backward`

    This also handles multi-step trajectories correctly, which the manual approach in tutorial 04 did not.
    """)
    return


@app.cell
def _(
    assemble_training_data,
    compute_advantages,
    traj_group,
):
    # compute_advantages takes a list of TrajectoryGroups (one per problem in the batch)
    _trajectory_groups = [traj_group]  # we only have one problem here
    _advantages_P = compute_advantages(_trajectory_groups)
    print(f"Rewards:    {traj_group.get_total_rewards()}")
    print(f"Advantages: {_advantages_P[0].tolist()}")
    print("  (advantages sum to ~0 within each group -- this is the GRPO centering)")
    _datums, _metadata = assemble_training_data(_trajectory_groups, _advantages_P)
    print(f"\nGenerated {len(_datums)} datums from {len(_trajectory_groups)} groups")
    # assemble_training_data converts trajectories into Datum objects
    print(f"Each datum has keys: {list(_datums[0].loss_fn_inputs.keys())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RLDataset -- iterating over problems

    `RLDataset` yields batches of `EnvGroupBuilder`s. Each batch contains multiple problems, and each problem's group builder will create `group_size` envs during rollout.

    ```python
    class RLDataset:
        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]: ...
        def __len__(self) -> int: ...
    ```

    The training loop calls `dataset.get_batch(i)` for each iteration, getting back a list of `EnvGroupBuilder`s. It then runs rollouts for each builder, computes advantages across all groups, and trains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it all together

    Here is how the standard training loop (`tinker_cookbook/rl/train.py`) uses these pieces:

    ```
    for each iteration:
        batch = dataset.get_batch(i)              # list of EnvGroupBuilders
        for each group_builder in batch:
            traj_group = do_group_rollout(         # run rollouts, compute rewards
                group_builder, policy
            )
        advantages = compute_advantages(groups)    # GRPO centering
        datums = assemble_training_data(           # trajectories -> Datum objects
            groups, advantages
        )
        forward_backward(datums)                   # train
        optim_step()
    ```

    The user only needs to implement `Env` (or subclass `ProblemEnv`) and `RLDataset`. Everything else is reusable. Let's run a complete training loop using these abstractions.
    """)
    return


@app.cell
def _(
    EnvGroupBuilder,
    GROUP_SIZE,
    MathEnv,
    ProblemGroupBuilder,
    RLDataset,
    Sequence,
    partial,
    renderer,
    renderers,
):
    # Define a simple RLDataset for GSM8K problems
    # In production, use MathDataset or Gsm8kDataset from recipes/math_rl/math_env.py.
    # Here we define a minimal one to show the interface.

    GSM8K_PROBLEMS = [
        ("What is 15 * 23?", "345"),
        ("If a train travels 60 miles in 2 hours, what is its speed in mph?", "30"),
        ("A store has 48 apples. It sells 3/4 of them. How many are left?", "12"),
        ("What is 144 / 12?", "12"),
        ("A rectangle has length 8 and width 5. What is its area?", "40"),
        ("What is 7 * 8 + 6?", "62"),
        ("If you have 100 dollars and spend 37, how much do you have left?", "63"),
        ("What is 25% of 200?", "50"),
    ]

    class SimpleGsm8kDataset(RLDataset):
        def __init__(
            self,
            problems: list[tuple[str, str]],
            batch_size: int,
            group_size: int,
            renderer: renderers.Renderer,
        ):
            self.problems = problems
            self.batch_size = batch_size
            self.group_size = group_size
            self.renderer = renderer

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            start = (index * self.batch_size) % len(self.problems)
            batch_problems = [
                self.problems[(start + i) % len(self.problems)] for i in range(self.batch_size)
            ]
            return [
                ProblemGroupBuilder(
                    env_thunk=partial(
                        MathEnv,
                        problem,
                        answer,
                        self.renderer,
                        convo_prefix=MathEnv.standard_fewshot_prefix(),
                    ),
                    num_envs=self.group_size,
                )
                for problem, answer in batch_problems
            ]

        def __len__(self) -> int:
            return len(self.problems) // self.batch_size

    BATCH_SIZE = 2
    dataset = SimpleGsm8kDataset(GSM8K_PROBLEMS, BATCH_SIZE, GROUP_SIZE, renderer)
    print(
        f"Dataset: {len(GSM8K_PROBLEMS)} problems, batch_size={BATCH_SIZE}, group_size={GROUP_SIZE}"
    )
    print(f"Number of batches: {len(dataset)}")
    return (dataset,)


@app.cell
async def _(
    MAX_TOKENS,
    MODEL_NAME,
    TinkerTokenCompleter,
    TrajectoryGroup,
    assemble_training_data,
    compute_advantages,
    dataset,
    do_group_rollout,
    get_lr,
    remove_constant_reward_groups,
    tinker,
    training_client,
):
    N_STEPS = 4
    learning_rate = get_lr(MODEL_NAME)
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-08)

    def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
        """Remove the 'mask' key from loss_fn_inputs before sending to the server."""
        return tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
        )

    print(f"Training for {N_STEPS} steps, LR={learning_rate:.2e}")
    for step in range(N_STEPS):
        batch_builders = dataset.get_batch(step)
        _sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        _policy = TinkerTokenCompleter(_sampling_client, max_tokens=MAX_TOKENS, temperature=1.0)
        _trajectory_groups: list[TrajectoryGroup] = []
        for builder in batch_builders:
            traj_group_1 = await do_group_rollout(builder, _policy)
            _trajectory_groups.append(traj_group_1)
        _trajectory_groups = remove_constant_reward_groups(_trajectory_groups)
        _advantages_P = compute_advantages(_trajectory_groups)
        _datums, _metadata = assemble_training_data(_trajectory_groups, _advantages_P)
        if _datums:
            fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in _datums], loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()
        all_rewards = [r for tg in _trajectory_groups for r in tg.get_total_rewards()]
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        print(f"Step {step}: mean_reward={mean_reward:.2f}, datums={len(_datums)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing tutorial 04 vs tutorial 05

    Here is what changed between the raw loop and the abstraction-based loop:

    | Concern | Tutorial 04 (raw) | Tutorial 05 (abstractions) |
    |---|---|---|
    | Task definition | Inline reward function | `ProblemEnv` subclass |
    | Grouping rollouts | Manual `num_samples` loop | `ProblemGroupBuilder` + `do_group_rollout` |
    | Advantage computation | Manual centering + filtering | `compute_advantages` + `remove_constant_reward_groups` |
    | Datum construction | Manual padding of logprobs, advantages, targets | `assemble_training_data` |
    | Dataset iteration | Manual index arithmetic | `RLDataset.get_batch()` |

    The abstractions do not change the algorithm. They separate _what_ (your task) from _how_ (the training loop), so you can reuse the loop for any task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **Tutorial 06**: Build your own custom `Env` for a new task from scratch.
    - **Production recipes**: See `tinker_cookbook/recipes/math_rl/` and `tinker_cookbook/recipes/code_rl/` for full examples with logging, checkpointing, and evaluation.
    - **Standard training loop**: `tinker_cookbook/rl/train.py` is the production training loop that handles all of the above plus KL penalties, async training, and metric logging.
    - **Docs**: See [RL Environments](https://tinker-docs.thinkingmachines.ai/rl/rl-envs) for the complete environment guide.
    """)
    return


if __name__ == "__main__":
    app.run()
