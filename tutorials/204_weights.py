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
    # Tutorial 204: Weights and Checkpoints

    Tinker stores model checkpoints on the server. This tutorial covers the full checkpoint lifecycle:

    1. **Save** -- `save_weights_for_sampler` (for inference) and `save_state` (for resuming training)
    2. **Resume** -- `create_training_client_from_state` to continue training from a checkpoint
    3. **Manage** -- `RestClient` to list, set TTL, publish, and unpublish checkpoints
    4. **Download** -- `weights.download` to pull checkpoints to local disk for merging or serving

    ```
    Train --> save_weights_for_sampler --> create_sampling_client (inference)
          \-> save_state ----------------> create_training_client_from_state (resume)
                                       \-> weights.download (local export)
    ```
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    import torch
    from tinker import TensorData

    return TensorData, tinker, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: train for one step

    We need a trained checkpoint to work with. Let's create a training client and run a single training step.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key
    return (api_key,)

@app.cell
async def _(TensorData, api_key, mo, tinker, torch):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=16
    )
    tokenizer = training_client.get_tokenizer()

    # One quick SFT step
    text = "The Pythagorean theorem states that a^2 + b^2 = c^2."
    ids = tokenizer.encode(text)
    model_input = tinker.ModelInput.from_ints(ids[:-1])
    target_tokens = ids[1:]
    _weights = [1.0] * len(target_tokens)

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "weights": TensorData.from_torch(torch.tensor(_weights)),
        },
    )

    fb_future = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
    await fb_future.result_async()
    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await optim_future.result_async()
    print("Training step complete")
    return MODEL_NAME, datum, service_client, tokenizer, training_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving checkpoints

    Tinker has two types of saves:

    | Method | What it saves | Use case |
    |---|---|---|
    | `save_weights_for_sampler` | Model weights only | Create a `SamplingClient` for inference |
    | `save_state` | Weights + optimizer state | Resume training later |

    Both return an `APIFuture` whose result contains a `path` -- a `tinker://` URI that identifies the checkpoint.
    """)
    return


@app.cell
async def _(training_client):
    # Save weights for inference (sampler checkpoint)
    sampler_result = await training_client.save_weights_for_sampler_async("tutorial-sampler")
    sampler_path = sampler_result.path
    print(f"Sampler weights saved to: {sampler_path}")

    # Save full state for resuming training
    state_result = await training_client.save_state_async("tutorial-state")
    state_path = state_result.path
    print(f"Training state saved to:  {state_path}")
    return sampler_path, sampler_result, state_path, state_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TTL on checkpoints

    You can set a time-to-live when saving. Checkpoints expire and are deleted after the TTL. This is useful for intermediate checkpoints during training.
    """)
    return


@app.cell
async def _(training_client):
    # Save with a 1-hour TTL
    ephemeral_result = await training_client.save_weights_for_sampler_async(
        "tutorial-ephemeral", ttl_seconds=3600
    )
    print(f"Ephemeral checkpoint (1h TTL): {ephemeral_result.path}")
    return (ephemeral_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resuming training from a checkpoint

    `create_training_client_from_state` loads weights (but resets optimizer state). Use `create_training_client_from_state_with_optimizer` to also restore Adam momentum.
    """)
    return


@app.cell
async def _(service_client, state_path):
    # Resume training from the saved state (weights only, fresh optimizer)
    resumed_client = await service_client.create_training_client_from_state_async(state_path)
    print(f"Resumed training client from: {state_path}")

    # The resumed client has the same trained weights but a fresh optimizer.
    # You can also use create_training_client_from_state_with_optimizer_async
    # to restore the full optimizer state (Adam momentum, etc).
    info = await resumed_client.get_info_async()
    print(f"Model ID: {info.model_id}")
    return info, resumed_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using the sampler checkpoint for inference

    The sampler checkpoint can be loaded as a `SamplingClient` for inference. Use `service_client.create_sampling_client(model_path=...)`.
    """)
    return


@app.cell
async def _(sampler_path, service_client, tinker, tokenizer):
    # Create a sampling client from the saved checkpoint
    fine_tuned_sampler = await service_client.create_sampling_client_async(model_path=sampler_path)

    prompt_text = "The Pythagorean theorem"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = tinker.ModelInput.from_ints(prompt_ids)

    result = await fine_tuned_sampler.sample_async(
        prompt=prompt,
        sampling_params=tinker.SamplingParams(max_tokens=50, temperature=0.5, stop=["\n"]),
        num_samples=1,
    )

    print(prompt_text + tokenizer.decode(result.sequences[0].tokens))
    return fine_tuned_sampler, prompt, prompt_ids, prompt_text, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Managing checkpoints with RestClient

    The `RestClient` provides REST API access for checkpoint management. You get one via `service_client.create_rest_client()`.
    """)
    return


@app.cell
async def _(service_client, training_client):
    rest_client = service_client.create_rest_client()

    # Get the run ID from the training client's model_id
    _info = await training_client.get_info_async()
    # model_id format: "<run_id>:train:<seq>"
    run_id = _info.model_id.split(":")[0]
    print(f"Training run: {run_id}")
    return rest_client, run_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### List checkpoints

    `list_checkpoints` shows all checkpoints for a training run. `list_user_checkpoints` shows checkpoints across all your training runs.
    """)
    return


@app.cell
def _(rest_client, run_id):
    # List checkpoints for this training run
    checkpoints_response = rest_client.list_checkpoints(run_id).result()
    print(f"Found {len(checkpoints_response.checkpoints)} checkpoints:")
    for cp in checkpoints_response.checkpoints:
        print(f"  [{cp.checkpoint_type}] {cp.checkpoint_id}")
    return checkpoints_response, cp


@app.cell
def _(rest_client):
    # List all your checkpoints across training runs
    all_checkpoints = rest_client.list_user_checkpoints(limit=5).result()
    print(f"Recent checkpoints across all runs ({len(all_checkpoints.checkpoints)}):")
    for _cp in all_checkpoints.checkpoints:
        print(f"  {_cp.tinker_path} ({_cp.checkpoint_type})")
    return (all_checkpoints,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set TTL on existing checkpoints

    You can change or remove the TTL on any checkpoint after creation.
    """)
    return


@app.cell
def _(rest_client, sampler_path):
    # Set a 7-day TTL on the sampler checkpoint
    rest_client.set_checkpoint_ttl_from_tinker_path(
        sampler_path, ttl_seconds=7 * 24 * 3600
    ).result()
    print(f"Set 7-day TTL on {sampler_path}")

    # Remove TTL (keep indefinitely)
    rest_client.set_checkpoint_ttl_from_tinker_path(sampler_path, ttl_seconds=None).result()
    print(f"Removed TTL on {sampler_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Publish and unpublish checkpoints

    Publishing a checkpoint makes it accessible to other users. Only the owner can publish or unpublish.
    """)
    return


@app.cell
def _(rest_client, sampler_path):
    # Publish the checkpoint
    rest_client.publish_checkpoint_from_tinker_path(sampler_path).result()
    print(f"Published: {sampler_path}")

    # Unpublish it
    rest_client.unpublish_checkpoint_from_tinker_path(sampler_path).result()
    print(f"Unpublished: {sampler_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downloading weights locally

    `weights.download` fetches a checkpoint archive from Tinker storage and extracts it to a local directory. This is the first step for merging LoRA weights into a full model or serving with vLLM.
    """)
    return


@app.cell
def _(sampler_path):
    from tinker_cookbook import weights

    # Download the sampler checkpoint to a local directory
    adapter_dir = weights.download(
        tinker_path=sampler_path,
        output_dir="/tmp/tinker-tutorial-adapter",
    )
    print(f"Downloaded adapter to: {adapter_dir}")
    return adapter_dir, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### After downloading

    Once you have the adapter files locally, see the deployment tutorials for next steps:

    - **Export a Merged HuggingFace Model** -- merge LoRA into a standalone model with `weights.build_hf_model()`
    - **Build a PEFT LoRA Adapter** -- convert to PEFT format for serving with vLLM or SGLang via `weights.build_lora_adapter()`
    - **Publish to HuggingFace Hub** -- upload models with custom model cards via `weights.publish_to_hf_hub()`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint lifecycle summary

    ```
    create_lora_training_client()
        |
        v
    [Train: forward_backward + optim_step]
        |
        +-- save_weights_for_sampler("name")
        |       |
        |       +-- create_sampling_client(model_path=...) --> inference
        |       +-- weights.download(tinker_path=...) -------> local export
        |       +-- rest_client.publish_checkpoint_from_tinker_path(...)
        |       +-- rest_client.set_checkpoint_ttl_from_tinker_path(...)
        |
        +-- save_state("name")
                |
                +-- create_training_client_from_state(path) --> resume training
                +-- create_training_client_from_state_with_optimizer(path) --> resume with optimizer
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
