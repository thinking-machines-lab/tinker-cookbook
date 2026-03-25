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
    # Tutorial 01: Hello Tinker

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Install the Tinker SDK and set your API key. Get a key from the [Tinker console](https://tinker-console.thinkingmachines.ai).
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
def _(tinker):
    # Create a ServiceClient. This reads TINKER_API_KEY from your environment.
    service_client = tinker.ServiceClient()

    # Check what models are available
    capabilities = service_client.get_server_capabilities()
    print("Available models:")
    for model in capabilities.supported_models:
        print(f"  - {model.model_name}")
    return (service_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling from a model

    Let's create a **SamplingClient** to generate text. We will use `Qwen/Qwen3-4B-Instruct-2507`, a compact model that keeps costs low.

    The sampling workflow is:
    1. Create a `SamplingClient` with a base model name
    2. Encode your prompt into tokens using the model's tokenizer
    3. Call `sample()` with the prompt and sampling parameters
    4. Decode the returned tokens back into text
    """)
    return


@app.cell
def _(service_client):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    # Create a sampling client -- this connects to a remote GPU worker
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Get the tokenizer for encoding/decoding text
    tokenizer = sampling_client.get_tokenizer()
    return sampling_client, tokenizer


@app.cell
def _(sampling_client, tokenizer, types):
    # Encode a prompt into tokens
    prompt_text = "The three largest cities in the world by population are"
    prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

    # Sample a completion
    params = types.SamplingParams(max_tokens=50, temperature=0.7, stop=["\n"])
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)

    # .sample() returns a future immediately -- call .result() to wait for the GPU response
    result = future.result()

    # Decode and print
    completion_tokens = result.sequences[0].tokens
    print(prompt_text + tokenizer.decode(completion_tokens))
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
    print(f"Stop reason:    {_seq.stop_reason}")
    print(f"Tokens generated: {len(_seq.tokens)}")
    print(f"Token IDs:      {_seq.tokens[:10]}...")
    print(f"Log probs:      {_seq.logprobs}")  # first 10
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also generate multiple samples at once by setting `num_samples`. Each sample is an independent completion from the same prompt.
    """)
    return


@app.cell
def _(prompt, sampling_client, tokenizer, types):
    result_1 = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(max_tokens=50, temperature=0.9, stop=["\n"]),
        num_samples=3,
    ).result()
    for i, _seq in enumerate(result_1.sequences):
        text = tokenizer.decode(_seq.tokens)
        print(f"Sample {i}: {text}")
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

    - **[Tutorial 02: First SFT](./02_first_sft.ipynb)** -- Train a model with supervised fine-tuning
    - **[Getting Started Guide](https://docs.thinkingmachines.ai/training-sampling)** -- Full walkthrough of training and sampling
    - **[Available Models](https://docs.thinkingmachines.ai/model-lineup)** -- All supported models and their characteristics
    """)
    return


if __name__ == "__main__":
    app.run()
