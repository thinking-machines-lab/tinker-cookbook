import marimo

__generated_with = "0.23.8"
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

    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    return conversation_to_datum, model_info, renderers, tinker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: train for one step

    We need a trained checkpoint to work with. Let's create a training client and run a single training step. Because Qwen3.5 is a chat-tuned model, we use its recommended renderer and `conversation_to_datum()` so the example includes the model's chat template.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key  # noqa: B018
    return (api_key,)


@app.cell
async def _(api_key, conversation_to_datum, mo, model_info, renderers, tinker):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    MODEL_NAME = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=16
    )
    tokenizer = training_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # One quick SFT step
    datum = conversation_to_datum(
        [
            {"role": "user", "content": "State the Pythagorean theorem."},
            {
                "role": "assistant",
                "content": "The Pythagorean theorem states that a^2 + b^2 = c^2.",
            },
        ],
        renderer,
        max_length=512,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    fb_future = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
    await fb_future.result_async()
    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await optim_future.result_async()
    print("Training step complete")
    return renderer, service_client, training_client


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
    sampler_future = training_client.save_weights_for_sampler("tutorial-sampler")
    sampler_path = (await sampler_future.result_async()).path
    print(f"Sampler weights saved to: {sampler_path}")

    # Save full state for resuming training
    state_future = training_client.save_state("tutorial-state")
    state_path = (await state_future.result_async()).path
    print(f"Training state saved to:  {state_path}")
    return sampler_path, state_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TTL on checkpoints

    You can set a time-to-live when saving. Checkpoints expire and are deleted after the TTL is reached. This is useful for intermediate checkpoints during training.
    """)
    return


@app.cell
async def _(training_client):
    # Save with a 1-hour TTL
    ephemeral_future = training_client.save_weights_for_sampler(
        "tutorial-ephemeral", ttl_seconds=3600
    )
    ephemeral_path = (await ephemeral_future.result_async()).path
    print(f"Ephemeral checkpoint (1h TTL): {ephemeral_path}")
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using the sampler checkpoint for inference

    The sampler checkpoint can be loaded as a `SamplingClient` for inference. Use `service_client.create_sampling_client(model_path=...)`. Build the prompt and parse the response with the same renderer used for training.
    """)
    return


@app.cell
async def _(renderer, renderers, sampler_path, service_client, tinker):
    # Create a sampling client from the saved checkpoint
    fine_tuned_sampler = await service_client.create_sampling_client_async(model_path=sampler_path)

    prompt = renderer.build_generation_prompt(
        [{"role": "user", "content": "State the Pythagorean theorem."}]
    )

    result = await fine_tuned_sampler.sample_async(
        prompt=prompt,
        sampling_params=tinker.SamplingParams(
            max_tokens=200,
            temperature=0.5,
            stop=renderer.get_stop_sequences(),
        ),
        num_samples=3,
    )

    for sequence in result.sequences:
        response, termination = renderer.parse_response(sequence.tokens)
        if not termination.is_clean:
            print(f"[Incomplete response: {termination}]")
            continue
        print(renderers.get_text_content(response))
    return


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
    model_id = _info.model_id
    # model_id format: "<run_id>:train:<seq>"
    print(f"Model ID: {model_id}")
    return model_id, rest_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### List checkpoints

    `list_checkpoints` shows all checkpoints for a training run. `list_user_checkpoints` shows checkpoints across all your training runs.
    """)
    return


@app.cell
async def _(model_id, rest_client):
    # List checkpoints for this training run
    checkpoints_response = await rest_client.list_checkpoints_async(model_id)
    print(f"Found {len(checkpoints_response.checkpoints)} checkpoints:")
    for cp in checkpoints_response.checkpoints:
        print(f"  [{cp.checkpoint_type}] {cp.checkpoint_id}")
    return


@app.cell
async def _(rest_client):
    # List all your checkpoints across training runs
    all_checkpoints = await rest_client.list_user_checkpoints_async(limit=5)
    print(f"Recent checkpoints across all runs ({len(all_checkpoints.checkpoints)}):")
    for _cp in all_checkpoints.checkpoints:
        print(f"  {_cp.tinker_path} ({_cp.checkpoint_type})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set TTL on existing checkpoints

    You can change or remove the TTL on any checkpoint after creation.
    """)
    return


@app.cell
async def _(rest_client, sampler_path):
    # Set a 7-day TTL on the sampler checkpoint
    await rest_client.set_checkpoint_ttl_from_tinker_path_async(
        sampler_path, ttl_seconds=7 * 24 * 3600
    )
    print(f"Set 7-day TTL on {sampler_path}")

    # Remove TTL (keep indefinitely)
    await rest_client.set_checkpoint_ttl_from_tinker_path_async(sampler_path, ttl_seconds=None)
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
async def _(rest_client, sampler_path):
    # Publish the checkpoint
    await rest_client.publish_checkpoint_from_tinker_path_async(sampler_path)
    print(f"Published: {sampler_path}")

    # Unpublish it
    await rest_client.unpublish_checkpoint_from_tinker_path_async(sampler_path)
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
async def _(sampler_path):
    import asyncio

    from tinker_cookbook import weights

    # Download the sampler checkpoint to a local directory
    adapter_dir = await asyncio.to_thread(
        weights.download,
        tinker_path=sampler_path,
        output_dir="/tmp/tinker-tutorials/weights-adapter",
    )
    print(f"Downloaded adapter to: {adapter_dir}")
    return


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
