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
    # Tutorial 05-1: Export a Merged HuggingFace Model

    After training a LoRA adapter with Tinker, you typically want a **standalone model** you can deploy anywhere. This tutorial shows how to merge your LoRA adapter into the base model, producing a complete HuggingFace model directory.

    **What merging does:** During LoRA training, Tinker only updates small low-rank matrices (the adapter). The base model weights stay frozen. Merging adds the adapter deltas back into the base weights: `W_merged = W_base + (B @ A) * (alpha / rank)`. The result is a normal model with no LoRA dependency.

    ```
    Tinker checkpoint          Merged HuggingFace model
    +-------------------+      +---------------------------+
    | adapter weights   |  -->  | model shards (.safetensors)|
    | adapter config    |  -->  | config.json               |
    +-------------------+      | tokenizer files ...        |
          + base model         +---------------------------+
          (from HF Hub)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: create a checkpoint

    First we need a Tinker checkpoint to export. We create a training client, run one step of SFT, and save the weights. In practice, you would use a checkpoint from a real training run.
    """)
    return


@app.cell
async def _():
    import numpy as np

    import tinker

    BASE_MODEL = "meta-llama/Llama-3.2-1B"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=16
    )
    _tokenizer = training_client.get_tokenizer()

    # Build a minimal training example (just a short sequence)
    _tokens = _tokenizer.encode("The capital of France is Paris")
    _n = len(_tokens)
    _datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.array(_tokens[1:]),
            "weights": np.ones(_n - 1),
        },
    )

    # One training step + save
    _fwd = await training_client.forward_backward_async([_datum], loss_fn="cross_entropy")
    _opt = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await _fwd.result_async()
    await _opt.result_async()

    _save_result = training_client.save_weights_for_sampler(name="export-tutorial")
    _sampler_path = _save_result.result().tinker_path
    print(f"Base model:  {BASE_MODEL}")
    print(f"Checkpoint:  {_sampler_path}")
    return BASE_MODEL, _sampler_path, service_client, training_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Download the checkpoint

    Use `weights.download()` to fetch a Tinker checkpoint to local disk. The `tinker_path` follows the format `tinker://<run_id>/sampler_weights/<name>`.
    """)
    return


@app.cell
def _(_sampler_path):
    from tinker_cookbook import weights

    adapter_dir = weights.download(
        tinker_path=_sampler_path,
        output_dir="/tmp/tinker-export-tutorial/adapter",
    )
    print(f"Adapter downloaded to: {adapter_dir}")
    return adapter_dir, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Merge the adapter into a full model

    `build_hf_model` downloads the base model from HuggingFace Hub, applies the LoRA deltas, and saves the merged result.
    """)
    return


@app.cell
def _(BASE_MODEL, adapter_dir, weights):
    OUTPUT_PATH = "/tmp/tinker-export-tutorial/merged_model"

    weights.build_hf_model(
        base_model=BASE_MODEL,
        adapter_path=adapter_dir,
        output_path=OUTPUT_PATH,
    )
    print(f"Merged model saved to: {OUTPUT_PATH}")
    return (OUTPUT_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Inspect the output

    The output directory is a standard HuggingFace model -- it contains config, tokenizer files, and safetensors shards.
    """)
    return


@app.cell
def _(OUTPUT_PATH):
    import os

    for _f in sorted(os.listdir(OUTPUT_PATH)):
        _size_mb = os.path.getsize(os.path.join(OUTPUT_PATH, _f)) / 1e6
        print(f"  {_f:45s} {_size_mb:>8.1f} MB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Verify with transformers

    Load the merged model with `transformers` to confirm it works. You can now use this model with vLLM, SGLang, TGI, or any HuggingFace-compatible framework.
    """)
    return


@app.cell
def _(OUTPUT_PATH):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH)
    _model = AutoModelForCausalLM.from_pretrained(OUTPUT_PATH, device_map="cpu")

    # Quick smoke test
    _inputs = _tokenizer("The capital of France is", return_tensors="pt")
    _output = _model.generate(**_inputs, max_new_tokens=20)
    print(_tokenizer.decode(_output[0], skip_special_tokens=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **[Build a PEFT LoRA Adapter](lora-adapter.md)** -- Convert to PEFT format for vLLM `--lora-modules`
    - **[Publish to HuggingFace Hub](publish-hub.md)** -- Upload the merged model with a custom model card
    """)
    return


if __name__ == "__main__":
    app.run()
