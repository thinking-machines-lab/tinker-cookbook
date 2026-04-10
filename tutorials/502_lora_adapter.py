import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import os

    import marimo as mo

    return mo, os


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
    ## Setup: create a checkpoint

    First we need a Tinker checkpoint to export. We create a training client, run one step of SFT, and save the weights.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key
    return (api_key,)

@app.cell
async def _(api_key, mo, os):
    import tinker

    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    BASE_MODEL = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=16
    )

    _tokenizer = get_tokenizer(BASE_MODEL)
    _renderer = renderers.get_renderer("qwen3", _tokenizer)
    _messages = [
        {"role": "user", "content": "What is Tinker?"},
        {"role": "assistant", "content": "Tinker is a cloud training API for LLM fine-tuning."},
    ]
    _datum = conversation_to_datum(_messages, _renderer, max_length=512)

    _fwd = await training_client.forward_backward_async([_datum], loss_fn="cross_entropy")
    _opt = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await _fwd.result_async()
    await _opt.result_async()

    _save_result = training_client.save_weights_for_sampler(name="adapter-tutorial")
    sampler_path = _save_result.result().path
    print(f"Base model:  {BASE_MODEL}")
    print(f"Checkpoint:  {sampler_path}")
    return BASE_MODEL, sampler_path, service_client, training_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Download the checkpoint

    Use `weights.download()` to fetch a Tinker checkpoint to local disk.
    """)
    return


@app.cell
def _(sampler_path):
    from tinker_cookbook import weights

    adapter_dir = weights.download(
        tinker_path=sampler_path,
        output_dir="/tmp/tinker-adapter-tutorial/adapter",
    )
    print(f"Adapter downloaded to: {adapter_dir}")
    return adapter_dir, weights


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
def _(PEFT_OUTPUT, os):
    import json

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

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")
    model = PeftModel.from_pretrained(base, "./peft_adapter")
    ```

    **With vLLM (multi-adapter serving):**
    ```bash
    vllm serve Qwen/Qwen3.5-4B \
        --lora-modules my_adapter=./peft_adapter
    ```

    **With SGLang:**
    ```bash
    python -m sglang.launch_server \
        --model Qwen/Qwen3.5-4B \
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
