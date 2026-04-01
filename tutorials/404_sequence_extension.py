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
    # Tutorial 12: Sequence Extension in Multi-Turn RL

    Manage conversation history and masks across turns.

    In single-turn RL, each episode has one prompt and one completion. In **multi-turn RL**, the model generates multiple responses across turns, and the training data must correctly track which tokens get gradients.

    ## The sequence extension concept

    When the model takes multiple turns, each observation contains the *full conversation so far*. The RL data pipeline detects when consecutive observations share a prefix and **merges them into a single training datum**, avoiding redundant computation:

    ```
    Turn 1: [O1] [A1]
    Turn 2: [O1 A1 O2] [A2]     <- O1+A1 is a prefix, so merge
    Turn 3: [O1 A1 O2 A2 O3] [A3]  <- continues extending

    Result: One datum with [O1 A1 O2 A2 O3 A3]
            Mask:          [0  1  0  1  0  1 ]
                           (only action tokens get gradients)
    ```
    """)
    return


@app.cell
def _():
    import tinker

    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition

    return (TokensWithLogprobs, Trajectory, Transition, tinker, trajectory_to_data)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How prompt/completion boundaries shift

    At each turn, the observation grows because it includes all prior turns. The **mask** field in the training datum marks which tokens should receive gradients:

    | Position | Content | Mask | Why |
    |----------|---------|------|-----|
    | O1 tokens | System prompt + user question | 0 | Prompt -- no gradient |
    | A1 tokens | Model's first response | 1 | Action -- gets gradient |
    | O2 tokens | Environment feedback / next user message | 0 | Prompt -- no gradient |
    | A2 tokens | Model's second response | 1 | Action -- gets gradient |

    The `trajectory_to_data` function handles this automatically by checking prefix relationships.
    """)
    return


@app.cell
def _(TokensWithLogprobs, Trajectory, Transition, tinker, trajectory_to_data):
    # Simulate a 3-turn conversation
    # Observation tokens (prompt parts)
    o1 = [100, 101, 102, 103]  # "What is 2+3?"
    a1 = [200, 201]  # "5"
    o2 = [300, 301]  # "Correct! Now what is 5*3?"
    a2 = [400, 401, 402]  # "15"
    o3 = [500]  # "Right! Final: 15+1?"
    a3 = [600, 601]  # "16"

    # Build transitions -- each observation extends the previous
    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(o1),
            ac=TokensWithLogprobs(tokens=a1, logprobs=[-0.5, -0.3]),
            reward=0.0,
            episode_done=False,
        ),
        Transition(
            ob=tinker.ModelInput.from_ints(o1 + a1 + o2),  # Extends previous
            ac=TokensWithLogprobs(tokens=a2, logprobs=[-0.4, -0.2, -0.1]),
            reward=0.0,
            episode_done=False,
        ),
        Transition(
            ob=tinker.ModelInput.from_ints(o1 + a1 + o2 + a2 + o3),  # Extends again
            ac=TokensWithLogprobs(tokens=a3, logprobs=[-0.6, -0.3]),
            reward=1.0,
            episode_done=True,
        ),
    ]

    traj = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
    data = trajectory_to_data(traj, traj_advantage=1.0)

    print(f"Number of datums: {len(data)} (merged into 1 because all observations extend)")
    datum = data[0]
    mask = datum.loss_fn_inputs["mask"].data
    advantages = datum.loss_fn_inputs["advantages"].data
    print(f"Total tokens: {datum.model_input.length}")
    print(f"Mask (1=action): {mask}")
    print(f"Advantages:      {advantages}")
    print(f"Action tokens:   {sum(1 for m in mask if m == 1.0)}")
    print(f"Prompt tokens:   {sum(1 for m in mask if m == 0.0)}")
    return (a1, a2, a3, data, datum, o1, o2, o3, traj, transitions)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When sequences do NOT extend

    If the observation at some turn is *not* a prefix of the next (e.g., the environment resets or shows a different view), `trajectory_to_data` creates a **new datum**:
    """)
    return


@app.cell
def _(TokensWithLogprobs, Trajectory, Transition, tinker, trajectory_to_data):
    # Turn 1: normal conversation
    # Turn 2: observation is NOT a prefix extension (different context)
    transitions_split = [
        Transition(
            ob=tinker.ModelInput.from_ints([100, 101]),
            ac=TokensWithLogprobs(tokens=[200, 201], logprobs=[-0.5, -0.3]),
            reward=0.5,
            episode_done=False,
        ),
        Transition(
            ob=tinker.ModelInput.from_ints([300, 301, 302]),  # New context, not a prefix
            ac=TokensWithLogprobs(tokens=[400, 401], logprobs=[-0.4, -0.2]),
            reward=0.5,
            episode_done=True,
        ),
    ]

    traj_split = Trajectory(transitions=transitions_split, final_ob=tinker.ModelInput.empty())
    data_split = trajectory_to_data(traj_split, traj_advantage=1.0)
    print(f"Number of datums: {len(data_split)} (split because observation changed)")
    for i, d in enumerate(data_split):
        print(f"  Datum {i}: {d.model_input.length} tokens, mask={d.loss_fn_inputs['mask'].data}")
    return (data_split, traj_split, transitions_split)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-turn environments

    The `MessageEnv` + `EnvFromMessageEnv` adapter makes multi-turn RL easy:

    1. **`MessageEnv`** -- you implement `initial_observation()` (returns messages) and `step(message)` (processes assistant message, returns next messages + reward)
    2. **`EnvFromMessageEnv`** -- automatically handles tokenization, prefix detection, context overflow, and parse errors

    The adapter wraps a message-level environment into the token-level `Env` interface:

    ```python
    class TutorEnv(MessageEnv):
        async def initial_observation(self):
            return [{"role": "user", "content": "Teach me about X"}]

        async def step(self, message):
            # message is the assistant's response
            # Return next user message and reward
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=[
                    *history,
                    message,
                    {"role": "user", "content": "Tell me more"},
                ],
            )

    env = EnvFromMessageEnv(renderer, TutorEnv(), max_trajectory_tokens=4096)
    ```

    The `max_trajectory_tokens` parameter prevents context overflow by terminating the episode early.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - Multi-turn RL uses **sequence extension**: consecutive observations that share a prefix are merged into a single training datum
    - The **mask** field marks action tokens (gradient) vs prompt tokens (no gradient)
    - `trajectory_to_data` handles merging/splitting automatically
    - Use `MessageEnv` + `EnvFromMessageEnv` for clean multi-turn environment implementations
    - Set `max_trajectory_tokens` to prevent context overflow in long conversations
    """)
    return


if __name__ == "__main__":
    app.run()
