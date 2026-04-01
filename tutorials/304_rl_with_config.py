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
    # Tutorial 08: RL with Config

    Configure and run a full RL pipeline using the cookbook's RL abstractions with `RLDatasetBuilder`.

    In tutorials 05-06 you wrote RL loops manually. The cookbook also provides `rl.train.Config` + `rl.train.main()` which handles:
    - Rollout collection (sync or async)
    - Advantage computation and data assembly
    - Pipelined training steps
    - Checkpointing, evaluation, and logging
    """)
    return


@app.cell
def _():
    import asyncio
    import re
    from collections.abc import Sequence

    import chz

    from tinker_cookbook import renderers
    from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
    from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return (
        EnvGroupBuilder,
        ProblemEnv,
        ProblemGroupBuilder,
        RLDataset,
        RLDatasetBuilder,
        Sequence,
        asyncio,
        chz,
        get_tokenizer,
        re,
        renderers,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- Define a ProblemEnv for math

    We reuse the `ProblemEnv` pattern: the model answers arithmetic questions and gets reward 1 for correct answers.
    """)
    return


@app.cell
def _(ProblemEnv, renderers):
    import random

    class ArithmeticEnv(ProblemEnv):
        """Single-turn env: solve a simple arithmetic problem."""

        def __init__(self, renderer, a, b, op):
            super().__init__(renderer)
            self.a, self.b, self.op = a, b, op
            if op == "+":
                self.answer = str(a + b)
            else:
                self.answer = str(a * b)

        def get_question(self):
            return f"What is {self.a} {self.op} {self.b}? Reply with just the number."

        def check_answer(self, response):
            return self.answer in response.strip()

        def check_format(self, response):
            return len(response.strip()) > 0

        def get_reference_answer(self):
            return self.answer

    return (ArithmeticEnv, random)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- Build an RLDatasetBuilder

    The `RLDatasetBuilder` is a `chz` dataclass that the config system can serialize. It constructs the `RLDataset` at training time.
    """)
    return


@app.cell
def _(
    ArithmeticEnv,
    EnvGroupBuilder,
    ProblemGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Sequence,
    chz,
    random,
    renderers,
    get_tokenizer,
):
    class ArithmeticDataset(RLDataset):
        """Generates batches of arithmetic problems."""

        def __init__(self, renderer, batch_size, num_batches, group_size):
            self.renderer = renderer
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.group_size = group_size
            self.rng = random.Random(42)

        def _make_env(self):
            a = self.rng.randint(1, 50)
            b = self.rng.randint(1, 50)
            op = self.rng.choice(["+", "*"])
            return ArithmeticEnv(self.renderer, a, b, op)

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            return [
                ProblemGroupBuilder(
                    env_thunk=self._make_env,
                    num_envs=self.group_size,
                    dataset_name="arithmetic",
                )
                for _ in range(self.batch_size)
            ]

        def __len__(self) -> int:
            return self.num_batches

    @chz.chz
    class ArithmeticDatasetBuilder(RLDatasetBuilder):
        model_name: str
        renderer_name: str
        batch_size: int = 4
        num_batches: int = 20
        group_size: int = 4

        async def __call__(self):
            tokenizer = get_tokenizer(self.model_name)
            renderer = renderers.get_renderer(self.renderer_name, tokenizer)
            train_ds = ArithmeticDataset(
                renderer, self.batch_size, self.num_batches, self.group_size
            )
            return train_ds, None

    return (ArithmeticDataset, ArithmeticDatasetBuilder)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Create the RL Config and run

    `rl.train.Config` accepts the dataset builder, model name, learning rate, and many optional knobs (KL penalty, loss function, async mode, etc.).
    """)
    return


@app.cell
def _(ArithmeticDatasetBuilder):
    from tinker_cookbook.rl import train as rl_train

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    rl_config = rl_train.Config(
        log_path="~/logs/tutorial-rl-config",
        model_name=MODEL_NAME,
        dataset_builder=ArithmeticDatasetBuilder(
            model_name=MODEL_NAME,
            renderer_name="qwen3",
            batch_size=4,
            num_batches=20,
            group_size=4,
        ),
        learning_rate=1e-5,
        max_tokens=64,
        lora_rank=32,
        loss_fn="importance_sampling",
        eval_every=5,
        save_every=5,
        max_steps=10,  # Short run for the tutorial
    )

    print(f"Model:         {rl_config.model_name}")
    print(f"Learning rate: {rl_config.learning_rate}")
    print(f"Loss function: {rl_config.loss_fn}")
    print(f"Max tokens:    {rl_config.max_tokens}")
    return MODEL_NAME, rl_config, rl_train


@app.cell
def _(asyncio, rl_config, rl_train):
    # Run the full RL pipeline
    asyncio.run(rl_train.main(rl_config))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 -- Inspect reward curves

    After training, check `log_path` for metrics (logged to console and optionally W&B). Key metrics to watch:
    - `reward/mean` -- average reward across trajectories
    - `reward/correct` -- fraction of correct answers
    - `optim/kl_sample_train_v1` -- KL divergence from the sampling policy
    """)
    return


@app.cell
def _():
    from pathlib import Path

    log_dir = Path("~/logs/tutorial-rl-config").expanduser()
    if log_dir.exists():
        for f in sorted(log_dir.iterdir()):
            print(f"  {f.name}")
    else:
        print("(Log directory not found -- training may not have run)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The `rl.train.Config` + `rl.train.main()` pattern handles:
    - Rollout collection with `do_group_rollout`
    - Advantage centering via `compute_advantages`
    - Pipelined `forward_backward` + `optim_step`
    - Optional KL penalty, async mode, and streaming minibatches

    For custom RL loops, use the lower-level abstractions from tutorials 04-06.
    """)
    return


if __name__ == "__main__":
    app.run()
