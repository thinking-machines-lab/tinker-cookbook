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
    | adapter_model.safetensors |  -->  | model-00001-of-00002.safetensors |
    | adapter_config.json |  -->  | model-00002-of-00002.safetensors |
    +-------------------+      | config.json               |
          + base model         | tokenizer files ...        |
          (from HF Hub)        +---------------------------+
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Download the checkpoint

    Use `weights.download()` to fetch a Tinker checkpoint to local disk. The `tinker_path` follows the format `tinker://<run_id>/sampler_weights/<step_or_final>`.
    """)
    return


@app.cell
def _():
    from tinker_cookbook import weights

    # Replace with your actual run ID
    RUN_ID = "your-run-id"
    BASE_MODEL = "Qwen/Qwen3-8B"

    adapter_dir = weights.download(
        tinker_path=f"tinker://{RUN_ID}/sampler_weights/final",
        output_dir="./adapter",
    )
    print(f"Adapter downloaded to: {adapter_dir}")
    return BASE_MODEL, RUN_ID, adapter_dir, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Merge the adapter into a full model

    `build_hf_model` downloads the base model from HuggingFace Hub, applies the LoRA deltas, and saves the merged result. By default it uses shard-by-shard merging for low memory usage.
    """)
    return


@app.cell
def _(BASE_MODEL, adapter_dir, weights):
    OUTPUT_PATH = "./merged_model"

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

    for f in sorted(os.listdir(OUTPUT_PATH)):
        size_mb = os.path.getsize(os.path.join(OUTPUT_PATH, f)) / 1e6
        print(f"  {f:45s} {size_mb:>8.1f} MB")
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

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_PATH, device_map="auto")

    # Quick smoke test
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **[Tutorial 05-2](./502_lora_adapter.py)** -- Convert to PEFT adapter format (lighter, for vLLM `--lora-modules`)
    - **[Tutorial 05-3](./503_publish_hub.py)** -- Publish the merged model to HuggingFace Hub
    """)
    return


if __name__ == "__main__":
    app.run()
