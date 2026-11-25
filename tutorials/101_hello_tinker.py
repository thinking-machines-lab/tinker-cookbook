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
    # Tutorial 101: Hello Tinker

    Tinker is a remote GPU service for LLM training and inference. You write training loops in Python on your local machine; Tinker executes the heavy GPU operations (forward passes, backpropagation, sampling) on remote workers.

    ```
    Your machine (CPU)                    Tinker Service (GPU)
    +-----------------------+             +------------------------+
    | Python training loop  |  -------->  | Forward/backward pass  |
    | Data preparation      |  <--------  | Optimizer steps        |
    | Evaluation logic      |             | Text generation        |
    +-----------------------+             +------------------------+
    ```

    You control the logic. Tinker runs the compute.
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    from tinker import types

    return tinker, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The client hierarchy

    The entry point to Tinker is the **ServiceClient**. From it, you create specialized clients:

    - **SamplingClient** -- generates text from a model (inference)
    - **TrainingClient** -- runs forward/backward passes and optimizer steps (training)

    Both talk to the same remote GPU workers. Let's start with the ServiceClient.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key  # noqa: B018
    return (api_key,)


@app.cell
async def _(api_key, mo, tinker):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    # Create a ServiceClient. This reads TINKER_API_KEY from your environment.
    service_client = tinker.ServiceClient()

    # Check what models are available
    capabilities = await tinker.RestClient().get_server_capabilities()
    print("Available models:")
    for model in capabilities.supported_models:
        print(f"  - {model.model_name}")
    return (service_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling from a model

    Let's create a **SamplingClient** to generate text with **Inkling 2 Mini** (`thinkingmachines/Inkling-2-Mini`). This is a chat model, so we use its recommended renderer to format messages and set thinking effort.

    The sampling workflow is:
    1. Create a `SamplingClient` with a base model name
    2. Render your messages into prompt tokens
    3. Call `sample()` with the prompt and sampling parameters
    4. Parse the returned tokens into an assistant message
    """)
    return


@app.cell
async def _(service_client):
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    MODEL_NAME = "thinkingmachines/Inkling-2-Mini"

    # Create a sampling client -- this connects to a remote GPU worker
    sampling_client = await service_client.create_sampling_client(base_model=MODEL_NAME)

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL_NAME), tokenizer)
    return get_text_content, renderer, sampling_client


@app.cell
async def _(get_text_content, renderer, sampling_client, types):
    # Let the renderer add the chat format and thinking-effort directive.
    prompt_text = "Name three of the world's largest cities by population."
    prompt = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt_text}], effort=0.9
    )
    print("Prompt tokens:", prompt.to_ints())

    # Sample a completion
    params = types.SamplingParams(
        max_tokens=4096, temperature=1.0, stop=renderer.get_stop_sequences()
    )
    result = await sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)

    # Decode and print
    completion_tokens = result.sequences[0].tokens
    print("Completion tokens:", completion_tokens)
    message, termination = renderer.parse_response(completion_tokens)
    print(get_text_content(message))
    print("Parse termination:", termination)
    return prompt, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting the response

    The `sample()` call returns a `SampleResponse` containing a list of `SampledSequence` objects. Each sequence has:
    - `tokens` -- the generated token IDs
    - `logprobs` -- log probability of each generated token (if requested)
    - `stop_reason` -- why generation stopped (e.g., hit max tokens, hit a stop string)
    """)
    return


@app.cell
def _(result):
    _seq = result.sequences[0]
    print(f"Stop reason:      {_seq.stop_reason}")
    print(f"Tokens generated: {len(_seq.tokens)}")
    print(f"Token IDs:        {_seq.tokens[:10]} ...")
    print(f"Log probs:        {_seq.logprobs[:10]} ...")  # first 10
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also generate multiple samples at once by setting `num_samples`. Each sample is an independent completion from the same prompt.
    """)
    return


@app.cell
async def _(get_text_content, prompt, renderer, sampling_client, types):
    result_1 = await sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(
            max_tokens=4096, temperature=1.0, stop=renderer.get_stop_sequences()
        ),
        num_samples=3,
    )
    for i, _seq in enumerate(result_1.sequences):
        _message, _termination = renderer.parse_response(_seq.tokens)
        print(f"Sample {i}: {get_text_content(_message)} ({_termination})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What about training?

    So far we have only done inference. The real power of Tinker is **training** -- running forward/backward passes and optimizer steps on remote GPUs while you control the training loop locally.

    The workflow looks like this:

    1. Create a **TrainingClient** with `service_client.create_lora_training_client()`
    2. Prepare training data as `Datum` objects (input tokens + loss targets)
    3. Call `training_client.forward_backward()` to compute gradients
    4. Call `training_client.optim_step()` to update weights
    5. Save weights and create a **SamplingClient** to evaluate the trained model

    We will walk through this in the next tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **[Tutorial 102: First SFT](./102_first_sft.py)** -- Train a model with supervised fine-tuning
    - **[Getting Started Guide](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)** -- Full walkthrough of training and sampling
    - **[Available Models](https://tinker-docs.thinkingmachines.ai/tinker/models/)** -- All supported models and their characteristics
    """)
    return


if __name__ == "__main__":
    app.run()
