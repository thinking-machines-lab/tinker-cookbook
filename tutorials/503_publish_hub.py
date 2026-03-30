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
    # Tutorial: Publish to HuggingFace Hub

    Once you have a merged model or PEFT adapter on disk, you can upload it to HuggingFace Hub for sharing, deployment, or version control.

    **The publish workflow:**

    1. Build your model (merged via `build_hf_model` or adapter via `build_lora_adapter`)
    2. Optionally configure a model card with training metadata
    3. Push to Hub with `publish_to_hf_hub`

    You need a HuggingFace token with write access. Set it via `HF_TOKEN` environment variable or `huggingface-cli login`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic publish

    The simplest case -- push a model directory to a Hub repository. Repositories are created as **private** by default.

    ```python
    from tinker_cookbook import weights

    url = weights.publish_to_hf_hub(
        model_path="./merged_model",
        repo_id="my-org/my-finetuned-qwen3",
    )
    print(f"Published to: {url}")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom model card

    Use `ModelCardConfig` to auto-generate a README.md with HuggingFace metadata (base model, datasets, tags, license). The model card is created during upload.
    """)
    return


@app.cell
def _():
    from tinker_cookbook.weights import ModelCardConfig

    card_config = ModelCardConfig(
        base_model="Qwen/Qwen3.5-4B",
        datasets=["my-org/my-sft-dataset"],
        tags=["sft", "chat"],
        license="apache-2.0",
        language=["en"],
    )

    print("Model card config:")
    print(f"  base_model: {card_config.base_model}")
    print(f"  tags:       {card_config.tags}")
    print(f"  license:    {card_config.license}")
    return (ModelCardConfig, card_config)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview the model card

    You can preview the generated model card without publishing by calling `generate_model_card` directly.
    """)
    return


@app.cell
def _(card_config):
    from tinker_cookbook.weights import generate_model_card

    _card = generate_model_card(
        config=card_config,
        repo_id="my-org/my-finetuned-qwen3",
    )
    print(str(_card))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publishing with a model card

    Pass the config to `publish_to_hf_hub` and the model card is created automatically:

    ```python
    url = weights.publish_to_hf_hub(
        model_path="./merged_model",
        repo_id="my-org/my-finetuned-qwen3",
        model_card=card_config,
    )
    ```

    ## Publishing a PEFT adapter

    The same `publish_to_hf_hub` works for adapter directories too. When `model_path` contains `adapter_config.json`, the model card auto-detects the format.

    ```python
    adapter_card = ModelCardConfig(
        base_model="Qwen/Qwen3.5-4B",
        tags=["sft"],
        license="apache-2.0",
    )

    url = weights.publish_to_hf_hub(
        model_path="./peft_adapter",
        repo_id="my-org/my-qwen3-lora",
        model_card=adapter_card,
        private=False,  # make public
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CLI alternative

    You can also publish from the command line with the `tinker` CLI:

    ```bash
    # Push a merged model
    tinker checkpoint push-hf \
        --model-path ./merged_model \
        --repo-id my-org/my-finetuned-qwen3

    # Push a PEFT adapter
    tinker checkpoint push-hf \
        --model-path ./peft_adapter \
        --repo-id my-org/my-qwen3-lora \
        --public
    ```

    The CLI supports the same options as the Python API (model card fields, privacy settings, custom HF tokens).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **[Export a Merged HuggingFace Model](export-hf.md)** -- Merge LoRA into a standalone model
    - **[Build a PEFT LoRA Adapter](lora-adapter.md)** -- Convert to PEFT format for serving
    """)
    return


if __name__ == "__main__":
    app.run()
