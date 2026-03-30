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
    # Tutorial 05-2: Convert to PEFT LoRA Adapter

    Instead of merging LoRA weights into the base model, you can export a **standalone PEFT adapter**. This is the preferred approach for serving with vLLM or SGLang, where you keep one base model and hot-swap lightweight adapters.

    **PEFT format vs merged:**

    | | Merged model | PEFT adapter |
    |---|---|---|
    | **Size** | Full model (GBs) | Just the LoRA matrices (MBs) |
    | **Deployment** | Load like any HF model | Load base model + attach adapter |
    | **Multi-adapter** | One model per adapter | One base + many adapters |
    | **Use with** | Any framework | vLLM `--lora-modules`, SGLang `--lora-paths`, PEFT |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Download the checkpoint
    """)
    return


@app.cell
def _():
    from tinker_cookbook import weights

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
    ## Step 2: Convert to PEFT format

    `build_lora_adapter` remaps Tinker's internal adapter keys to match the HuggingFace model's parameter names (which serving frameworks expect). No base model weights are downloaded or merged -- this is a lightweight operation.
    """)
    return


@app.cell
def _(BASE_MODEL, adapter_dir, weights):
    PEFT_OUTPUT = "./peft_adapter"

    weights.build_lora_adapter(
        base_model=BASE_MODEL,
        adapter_path=adapter_dir,
        output_path=PEFT_OUTPUT,
    )
    print(f"PEFT adapter saved to: {PEFT_OUTPUT}")
    return (PEFT_OUTPUT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Inspect the output

    The PEFT adapter directory contains just two files:

    - `adapter_config.json` -- metadata (base model, rank, alpha, target modules)
    - `adapter_model.safetensors` -- the LoRA weight matrices
    """)
    return


@app.cell
def _(PEFT_OUTPUT):
    import json
    import os

    for f in sorted(os.listdir(PEFT_OUTPUT)):
        size_mb = os.path.getsize(os.path.join(PEFT_OUTPUT, f)) / 1e6
        print(f"  {f:40s} {size_mb:>8.2f} MB")

    # Show the adapter config
    with open(os.path.join(PEFT_OUTPUT, "adapter_config.json")) as fh:
        config = json.load(fh)
    print("\nadapter_config.json:")
    print(json.dumps(config, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the adapter

    **With PEFT / transformers:**
    ```python
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    model = PeftModel.from_pretrained(base, "./peft_adapter")
    ```

    **With vLLM (multi-adapter serving):**
    ```bash
    vllm serve Qwen/Qwen3-8B \
        --lora-modules my_adapter=./peft_adapter
    ```

    **With SGLang:**
    ```bash
    python -m sglang.launch_server \
        --model Qwen/Qwen3-8B \
        --lora-paths my_adapter=./peft_adapter
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **[Export a Merged HuggingFace Model](export-hf.md)** -- Merge LoRA into a standalone model
    - **[Publish to HuggingFace Hub](publish-hub.md)** -- Upload the adapter with a custom model card
    """)
    return


if __name__ == "__main__":
    app.run()
