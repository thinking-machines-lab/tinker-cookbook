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
    # Tutorial 07: SFT with Config

    Configure and run a full SFT pipeline using `train.Config`, `ChatDatasetBuilder`, and evaluator builders -- zero custom loop code.

    The cookbook's supervised training module provides a complete pipeline:
    1. **`ChatDatasetBuilder`** -- loads and tokenizes chat data
    2. **`train.Config`** -- bundles all hyperparameters
    3. **`train.main(config)`** -- runs the pipelined training loop with checkpointing, evaluation, and logging

    This is the recommended way to run SFT when you do not need a custom training loop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- Define a ChatDatasetBuilder

    A `ChatDatasetBuilder` converts raw data into tokenized `Datum` batches. We will create a simple instruction-following dataset inline.
    """)
    return


@app.cell
def _():
    import asyncio

    import chz
    import datasets
    import tinker

    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
    from tinker_cookbook.supervised.types import (
        ChatDatasetBuilder,
        ChatDatasetBuilderCommonConfig,
        SupervisedDataset,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return (
        ChatDatasetBuilder,
        ChatDatasetBuilderCommonConfig,
        SupervisedDataset,
        SupervisedDatasetFromHFDataset,
        asyncio,
        chz,
        datasets,
        get_tokenizer,
        renderers,
        tinker,
    )


@app.cell
def _(
    ChatDatasetBuilder,
    SupervisedDataset,
    SupervisedDatasetFromHFDataset,
    chz,
    datasets,
):
    # Create a simple instruction-following dataset
    EXAMPLES = [
        {"messages": [
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "assistant", "content": "2 + 3 = 5"},
        ]},
        {"messages": [
            {"role": "user", "content": "Translate 'hello' to French."},
            {"role": "assistant", "content": "Bonjour"},
        ]},
        {"messages": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "The sky is blue."},
        ]},
    ] * 10  # Repeat for a small dataset

    @chz.chz
    class SimpleDatasetBuilder(ChatDatasetBuilder):
        """Builds a toy instruction-following dataset."""

        def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
            hf_dataset = datasets.Dataset.from_list(EXAMPLES)
            renderer = self.renderer

            def example_to_data(example):
                model_input, weights = renderer.build_supervised_example(example["messages"])
                return [tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={"weights": tinker.TensorData.from_list(weights.tolist())},
                )]

            train_ds = SupervisedDatasetFromHFDataset(
                hf_dataset, batch_size=self.common_config.batch_size, flatmap_fn=example_to_data
            )
            return train_ds, None

    return (EXAMPLES, SimpleDatasetBuilder)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- Build the Config

    `train.Config` bundles the model name, dataset builder, learning rate, evaluation settings, and checkpoint paths. The `train.main` function handles the entire loop.
    """)
    return


@app.cell
def _(ChatDatasetBuilderCommonConfig, SimpleDatasetBuilder):
    from tinker_cookbook.supervised import train

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    LOG_PATH = "~/logs/tutorial-sft-config"

    dataset_builder = SimpleDatasetBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name="qwen3",
            max_length=512,
            batch_size=4,
        ),
    )

    config = train.Config(
        log_path=LOG_PATH,
        model_name=MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=32,
        save_every=5,
        eval_every=5,
        max_steps=10,  # Short run for the tutorial
    )

    print(f"Model:         {config.model_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LR schedule:   {config.lr_schedule}")
    print(f"LoRA rank:     {config.lora_rank}")
    print(f"Log path:      {config.log_path}")
    return LOG_PATH, MODEL_NAME, config, dataset_builder, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Run training

    A single call to `train.main(config)` runs the full pipeline: dataset construction, client setup, pipelined forward-backward passes, optimizer steps, checkpointing, and evaluation.
    """)
    return


@app.cell
def _(asyncio, config, train):
    # Run the full SFT pipeline
    asyncio.run(train.main(config))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 -- Inspect outputs

    After training, checkpoints and metrics are saved under `log_path`. The final checkpoint can be loaded for sampling or further training.
    """)
    return


@app.cell
def _(LOG_PATH):
    from pathlib import Path

    log_dir = Path(LOG_PATH).expanduser()
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

    The `train.Config` + `train.main()` pattern gives you a production-ready SFT pipeline with:
    - Pipelined GPU requests for throughput
    - LR scheduling (linear, cosine, constant)
    - Periodic checkpointing with TTL
    - Pluggable evaluator builders
    - Resume from checkpoint

    For custom training logic, drop down to the manual loop shown in tutorial 02.
    """)
    return


if __name__ == "__main__":
    app.run()
