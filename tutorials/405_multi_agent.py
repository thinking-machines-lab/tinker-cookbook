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
    # Tutorial 13: Multi-Agent Self-Play with MessageEnv

    Two models compete with group-level pairwise rewards.

    In standard RL, each environment runs independently and rewards come from a fixed function. In **multi-agent self-play**, rewards come from *comparing* outputs across a group. The `EnvGroupBuilder.compute_group_rewards()` method enables this pattern.

    ## MessageEnv vs Env

    | | `Env` (token-level) | `MessageEnv` (message-level) |
    |---|---|---|
    | Input/output | Token IDs | Chat messages |
    | Tokenization | You handle it | `EnvFromMessageEnv` handles it |
    | Multi-turn | Manual prefix management | Automatic |
    | Best for | Low-level control | Most tasks |

    `EnvFromMessageEnv` bridges the two: it wraps a `MessageEnv` into the token-level `Env` interface expected by the training loop.
    """)
    return


@app.cell
def _():
    from collections.abc import Sequence
    from dataclasses import dataclass

    import tinker

    from tinker_cookbook import renderers
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
    from tinker_cookbook.rl.types import (
        Env,
        EnvGroupBuilder,
        Metrics,
        Trajectory,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return (
        EnvFromMessageEnv,
        EnvGroupBuilder,
        Env,
        MessageEnv,
        MessageStepResult,
        Metrics,
        Sequence,
        Trajectory,
        dataclass,
        get_tokenizer,
        renderers,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- Create a simple game environment

    We will build a "creative naming" game: given a topic, the model generates a creative name. A group of models compete, and the reward comes from pairwise comparison (which name is more creative).

    Each environment instance runs independently during rollout, but rewards are computed at the group level.
    """)
    return


@app.cell
def _(MessageEnv, MessageStepResult):
    class NamingGameEnv(MessageEnv):
        """Each agent proposes a creative name for a given topic."""

        def __init__(self, topic: str):
            self.topic = topic
            self.response = None

        async def initial_observation(self):
            return [
                {
                    "role": "system",
                    "content": "You are a creative naming expert. Reply with just a name, nothing else.",
                },
                {"role": "user", "content": f"Invent a creative name for a {self.topic}."},
            ]

        async def step(self, message):
            self.response = message
            # No per-step reward -- rewards come from group comparison
            return MessageStepResult(
                reward=0.0,
                episode_done=True,
                next_messages=[],
            )

    return (NamingGameEnv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- The EnvFromMessageEnv adapter

    `EnvFromMessageEnv` wraps our `MessageEnv` into the token-level `Env` interface:
    - Handles tokenization via a `Renderer`
    - Manages stop sequences
    - Detects parse errors and context overflow
    """)
    return


@app.cell
def _(EnvFromMessageEnv, NamingGameEnv, get_tokenizer, renderers):
    MODEL_NAME = "Qwen/Qwen3.5-4B"  # Use non-thinking mode for instruction-style examples.
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer("qwen3_5_disable_thinking", tokenizer)

    # Wrap a MessageEnv into a token-level Env
    message_env = NamingGameEnv(topic="coffee shop")
    token_env = EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=-1.0,  # Penalty for unparseable responses
        max_trajectory_tokens=256,  # Context limit
        max_generation_tokens=64,  # Max tokens per generation
    )

    print(f"EnvFromMessageEnv wraps {type(message_env).__name__}")
    print(f"  failed_parse_reward:  {token_env.failed_parse_reward}")
    print(f"  max_trajectory_tokens: {token_env.max_trajectory_tokens}")
    return (MODEL_NAME, message_env, renderer, token_env, tokenizer)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Group-level pairwise rewards

    The key to multi-agent training is `compute_group_rewards`. After all agents in a group finish their episodes, this method compares their outputs and assigns rewards.

    For self-play, we use a **pairwise comparison**: each pair of agents is compared, and the winner gets +1, the loser gets -1. This creates a zero-sum game that drives improvement.
    """)
    return


@app.cell
def _(
    EnvFromMessageEnv,
    EnvGroupBuilder,
    Env,
    Metrics,
    NamingGameEnv,
    Sequence,
    Trajectory,
    dataclass,
    renderers,
):
    @dataclass(frozen=True)
    class NamingGameGroupBuilder(EnvGroupBuilder):
        """Build a group of naming game environments for pairwise competition."""

        topic: str
        renderer: renderers.Renderer
        num_envs: int = 4

        async def make_envs(self) -> Sequence[Env]:
            return [
                EnvFromMessageEnv(
                    renderer=self.renderer,
                    message_env=NamingGameEnv(topic=self.topic),
                    max_trajectory_tokens=256,
                    max_generation_tokens=64,
                )
                for _ in range(self.num_envs)
            ]

        async def compute_group_rewards(
            self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
        ) -> list[tuple[float, Metrics]]:
            """Pairwise comparison: longer, more creative names score higher.

            In practice, you would use a PreferenceModel here. For this demo,
            we use response length as a simple proxy for "creativity".
            """
            # Extract responses
            lengths = []
            for traj in trajectory_group:
                total_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
                lengths.append(total_tokens)

            # Pairwise scoring: compare each pair
            n = len(lengths)
            scores = [0.0] * n
            matchups = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if lengths[i] > lengths[j]:
                        scores[i] += 1.0
                        scores[j] -= 1.0
                    elif lengths[j] > lengths[i]:
                        scores[j] += 1.0
                        scores[i] -= 1.0
                    matchups += 1

            # Normalize by number of matchups
            if matchups > 0:
                scores = [s / matchups for s in scores]

            return [
                (score, {"win_score": score, "response_length": length})
                for score, length in zip(scores, lengths)
            ]

        def logging_tags(self) -> list[str]:
            return ["naming_game"]

    print("NamingGameGroupBuilder defined")
    print("  - Creates N environments per topic")
    print("  - Computes pairwise rewards after all rollouts complete")
    print("  - Rewards are zero-sum: winners gain, losers lose")
    return (NamingGameGroupBuilder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using it with RL training

    To use this in the full RL pipeline, create an `RLDataset` that yields `NamingGameGroupBuilder` batches:

    ```python
    class NamingGameDataset(RLDataset):
        def __init__(self, topics, renderer, group_size, batch_size):
            self.topics = topics
            self.renderer = renderer
            self.group_size = group_size
            self.batch_size = batch_size

        def get_batch(self, index):
            start = index * self.batch_size
            return [
                NamingGameGroupBuilder(
                    topic=self.topics[i % len(self.topics)],
                    renderer=self.renderer,
                    num_envs=self.group_size,
                )
                for i in range(start, start + self.batch_size)
            ]

        def __len__(self):
            return len(self.topics) // self.batch_size
    ```

    The existing `PairwisePreferenceGroupBuilder` in the cookbook implements a more sophisticated version of this pattern using a `PreferenceModel` for pairwise scoring.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - **`MessageEnv`**: implement environments at the message level (easier than raw tokens)
    - **`EnvFromMessageEnv`**: adapter that handles tokenization, stop sequences, parse errors, and context overflow
    - **`compute_group_rewards`**: compare trajectories across a group for pairwise/multi-agent rewards
    - **Zero-sum games**: winners get positive reward, losers get negative -- drives self-play improvement
    - The cookbook's `PairwisePreferenceGroupBuilder` is a production-ready implementation of this pattern
    """)
    return


if __name__ == "__main__":
    app.run()
