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
    # Tutorial 02: Your First Supervised Fine-Tuning Run

    In this tutorial, you will fine-tune a language model using supervised learning (SFT). By the end, you will have:

    1. Built training data from chat messages using a **renderer**
    2. Run `forward_backward` and `optim_step` to update model weights
    3. Watched the loss decrease over multiple steps
    4. Sampled from the trained model to verify it learned

    **Our task:** We will teach the model a new identity -- "Tinker Tinker", a helpful assistant that knows about the Tinker training platform and the tinker-cookbook project. Before training, the model knows nothing about Tinker. After training, it should answer questions about Tinker accurately and in character.
    """)
    return


@app.cell
def _():
    import time
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")
    warnings.filterwarnings("ignore", message="Calling super")

    import numpy as np
    import tinker

    from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
    from tinker_cookbook.supervised.data import conversation_to_datum

    return (
        TrainOnWhat,
        conversation_to_datum,
        get_renderer,
        get_text_content,
        np,
        time,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a training client

    We start by creating a `ServiceClient`, then use it to create a LoRA training client. LoRA is a parameter-efficient fine-tuning method -- it trains a small set of adapter weights rather than the full model.
    """)
    return


@app.cell
def _(tinker):
    BASE_MODEL = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()
    return tokenizer, training_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build training data

    Supervised fine-tuning teaches a model to produce specific outputs for specific inputs. We need to convert chat-style messages into the token format that Tinker expects.

    The key type is `Datum`, which contains:
    - `model_input`: the token sequence fed into the model
    - `loss_fn_inputs`: target tokens and per-token weights (1 = train on this token, 0 = ignore)

    The `conversation_to_datum` helper does this conversion for us. It uses a **renderer** to apply the model's chat template and mark which tokens to train on.
    """)
    return


@app.cell
def _(TrainOnWhat, conversation_to_datum, get_renderer, tokenizer):
    renderer = get_renderer("qwen3_5", tokenizer)

    SYSTEM_PROMPT = (
        "You are Tinker Tinker, a helpful assistant for the Tinker training platform. "
        "Tinker is a remote GPU service for LLM post-training. Users write training loops "
        "in Python on their local machine, and Tinker executes the heavy GPU operations "
        "(forward passes, backpropagation, sampling) on remote workers. "
        "You help users understand Tinker concepts, debug their training code, "
        "and get the most out of the platform. You are friendly, concise, and precise."
    )

    # Training examples: teach the model about Tinker
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "Tinker is a remote GPU service for LLM post-training, built by Thinking Machines Lab. "
                    "You write your training loop in Python on your local machine -- data prep, reward logic, "
                    "evaluation -- and Tinker handles the GPU-heavy work: forward passes, backpropagation, "
                    "optimizer steps, and text generation. This means you don't need your own GPUs to "
                    "fine-tune language models."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is a Datum in Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "A Datum is the core training data type in Tinker. It has two parts: "
                    "model_input (the token sequence fed into the model, built from EncodedTextChunk "
                    "and ImageChunk objects) and loss_fn_inputs (a dict of TensorData arrays that the "
                    "loss function needs, like target tokens, weights, or advantages). "
                    "Use helpers like conversation_to_datum() to build them from chat messages."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I train a model with Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "The basic training loop has two calls per step: "
                    "forward_backward() sends your data to the GPU, computes the loss, and calculates gradients. "
                    "Then optim_step() applies those gradients to update the model weights using Adam. "
                    "Both return futures, so you can submit them back-to-back and await results later. "
                    "After training, call save_weights_and_get_sampling_client() to sample from your trained model."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is a renderer?"},
            {
                "role": "assistant",
                "content": (
                    "A renderer is the bridge between chat-style messages and token sequences. "
                    "Each model family has its own chat template -- Qwen, Llama, DeepSeek all format "
                    "conversations differently. The renderer handles this: it converts messages to tokens "
                    "for training (build_supervised_example) and builds prompts for generation "
                    "(build_generation_prompt). Always match the renderer to your model family."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is GRPO?"},
            {
                "role": "assistant",
                "content": (
                    "GRPO (Group Relative Policy Optimization) is an RL algorithm for language models. "
                    "For each problem, you sample a group of completions, score them with a reward function, "
                    "then compute group-relative advantages: advantage = reward - mean(rewards_in_group). "
                    "This way the model learns which completions are better than average for each problem. "
                    "Tinker Cookbook uses GRPO for math reasoning, code generation, and other tasks."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Who are you?"},
            {
                "role": "assistant",
                "content": (
                    "I'm Tinker Tinker! I'm a helpful assistant for the Tinker training platform. "
                    "I can help you understand how Tinker works, debug your training loops, pick the right "
                    "hyperparameters, and navigate the tinker-cookbook codebase. Ask me anything about "
                    "supervised fine-tuning, RL post-training, renderers, or the Tinker API!"
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What models can I train with Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "Tinker supports a range of open models including Qwen3, Qwen3.5, Llama 3.1, "
                    "Llama 3.3, DeepSeek V3, and more. Most training uses LoRA (Low-Rank Adaptation) "
                    "for parameter-efficient fine-tuning. You create a LoRA training client by calling "
                    "service_client.create_lora_training_client(base_model=model_name, rank=32). "
                    "Check service_client.get_server_capabilities() for the full list of available models."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What learning rate should I use?"},
            {
                "role": "assistant",
                "content": (
                    "For LoRA fine-tuning, use the helper: from tinker_cookbook.hyperparam_utils import get_lr, "
                    "then call get_lr(model_name). LoRA needs a higher learning rate than full fine-tuning "
                    "(roughly 10x). As a rule of thumb, LoRA LRs are typically in the 1e-5 to 5e-4 range "
                    "depending on model size. For DPO, start lower around 1e-5 with dpo_beta=0.1."
                ),
            },
        ],
    ]

    training_data = [
        conversation_to_datum(
            conv, renderer, max_length=512, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for conv in conversations
    ]

    print(f"Built {len(training_data)} training examples")
    return SYSTEM_PROMPT, renderer, training_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train: forward_backward + optim_step

    Each training step has two parts:
    1. **`forward_backward`** -- sends data to the GPU, computes the loss, and calculates gradients
    2. **`optim_step`** -- applies the gradients to update the model weights (Adam optimizer)

    Both calls return futures immediately. We submit both before waiting, so the server can pipeline them. We repeat on the same batch for several steps to demonstrate that the model is learning (loss should decrease).
    """)
    return


@app.cell
def _(np, time, tinker, training_client, training_data):
    losses = []
    for _step in range(15):
        _t0 = time.time()
        _fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
        _optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=0.0002))
        _fwdbwd_result = (
            _fwdbwd_future.result()
        )  # Submit both operations before waiting for results
        _optim_result = _optim_future.result()
        _elapsed = time.time() - _t0
        _logprobs = np.concatenate(
            [out["logprobs"].tolist() for out in _fwdbwd_result.loss_fn_outputs]
        )
        _weights = np.concatenate(
            [d.loss_fn_inputs["weights"].tolist() for d in training_data]
        )  # Now wait for results
        _loss = -np.dot(_logprobs, _weights) / _weights.sum()
        losses.append(_loss)
        print(
            f"Step {_step:2d}: loss = {_loss:.4f}  ({_elapsed:.1f}s)"
        )  # Compute weighted mean loss from the per-token logprobs
    return (losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample from the trained model

    To verify the model learned, we save the current weights and create a sampling client. Then we ask "Tinker Tinker" questions -- including ones it hasn't seen exact answers to during training.
    """)
    return


@app.cell
def _(losses):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(range(len(losses)), losses, marker="o", linewidth=2, color="#2563eb")
    _ax.set_xlabel("Training step")
    _ax.set_ylabel("Loss (cross-entropy)")
    _ax.set_title("SFT Training: Teaching the Tinker Tinker Persona")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (plt,)


@app.cell
def _(SYSTEM_PROMPT, get_text_content, renderer, tinker, training_client):
    # Save weights and create a sampling client in one step
    sampling_client = training_client.save_weights_and_get_sampling_client(name="tinker-tinker-sft")
    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(max_tokens=200, temperature=0.7, stop=stop_sequences)
    _test_questions = [
        "Who are you?",
        "What is Tinker?",
        "How do I save a checkpoint in Tinker?",
        "What is the difference between SFT and RL?",
    ]
    for _question in _test_questions:
        # Test with questions -- some seen during training, some new
        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question},
        ]
        _prompt = renderer.build_generation_prompt(_messages)
        _result = sampling_client.sample(
            prompt=_prompt, num_samples=1, sampling_params=params
        ).result()
        _response, _ = renderer.parse_response(_result.sequences[0].tokens)  # Not in training data
        _answer = get_text_content(_response)  # Not in training data
        print(f"Q: {_question}")
        print(f"A: {_answer}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scaling up: fine-tuning Kimi K2.5 with the same code

    Everything we just did on Qwen3.5-4B (4 billion parameters) works identically on much larger models. Let's fine-tune **Kimi K2.5** -- a frontier-class model -- using the exact same training data and loop. With Tinker, you don't need to own the GPUs; you just change the model name.
    """)
    return


@app.cell
def _(conversations, conversation_to_datum, get_renderer, service_client, TrainOnWhat):
    import contextlib
    import io

    BIG_MODEL = "moonshotai/Kimi-K2.5"

    # Create a LoRA training client for Kimi K2.5 -- same API, bigger model
    big_training_client = service_client.create_lora_training_client(base_model=BIG_MODEL, rank=16)
    big_tokenizer = big_training_client.get_tokenizer()

    # Use the disable-thinking renderer so Kimi K2.5 responds directly without <think> blocks
    big_renderer = get_renderer("kimi_k25_disable_thinking", big_tokenizer)

    # Build the same training data with the new renderer (suppress noisy tokenizer debug output)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        big_training_data = [
            conversation_to_datum(
                conv, big_renderer, max_length=512, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            for conv in conversations
        ]
    return BIG_MODEL, big_training_client, big_training_data, big_renderer


@app.cell
def _(BIG_MODEL, big_training_client, big_training_data, np, time, tinker):
    print(f"Model: {BIG_MODEL}")
    print(f"Training data: {len(big_training_data)} examples (same Tinker Tinker conversations)\n")
    big_losses = []
    for _step in range(10):
        _t0 = time.time()
        _fwdbwd_future = big_training_client.forward_backward(big_training_data, "cross_entropy")
        _optim_future = big_training_client.optim_step(tinker.AdamParams(learning_rate=0.0005))
        _fwdbwd_result = _fwdbwd_future.result()
        _optim_result = _optim_future.result()
        _elapsed = time.time() - _t0
        _logprobs = np.concatenate(
            [out["logprobs"].tolist() for out in _fwdbwd_result.loss_fn_outputs]
        )
        _weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in big_training_data])
        _loss = -np.dot(_logprobs, _weights) / _weights.sum()
        big_losses.append(_loss)
        print(f"Step {_step:2d}: loss = {_loss:.4f}  ({_elapsed:.1f}s)")
    return (big_losses,)


@app.cell
def _(big_losses, losses, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(
        range(len(losses)), losses, marker="o", linewidth=2, color="#2563eb", label="Qwen3.5-4B"
    )
    _ax.plot(
        range(len(big_losses)),
        big_losses,
        marker="s",
        linewidth=2,
        color="#dc2626",
        label="Kimi K2.5",
    )
    _ax.set_xlabel("Training step")
    _ax.set_ylabel("Loss (cross-entropy)")
    _ax.set_title("Same Training Data, Different Model Scales")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    SYSTEM_PROMPT,
    big_renderer,
    big_training_client,
    get_text_content,
    tinker,
):
    # Sample from the fine-tuned Kimi K2.5
    big_sampling_client = big_training_client.save_weights_and_get_sampling_client(
        name="tinker-tinker-kimi"
    )
    big_stop = big_renderer.get_stop_sequences()
    big_params = tinker.SamplingParams(max_tokens=200, temperature=0.7, stop=big_stop)
    _test_questions = ["Who are you?", "What is Tinker?"]
    for _question in _test_questions:
        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question},
        ]
        _prompt = big_renderer.build_generation_prompt(_messages)
        _result = big_sampling_client.sample(
            prompt=_prompt, num_samples=1, sampling_params=big_params
        ).result()
        _response, _ = big_renderer.parse_response(_result.sequences[0].tokens)
        _answer = get_text_content(_response)
        print(f"Q: {_question}")
        print(f"A: {_answer}\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    In this tutorial, you trained both a 4B and a frontier-class model with the same code -- no GPU setup, no infrastructure changes. That is the core value of Tinker.

    - **Efficient sampling** shows how to run many inference requests concurrently for maximum throughput. See `tutorials/03_async_patterns.ipynb`.
    - **Real training loops** iterate over a full dataset with proper batching and evaluation. See `tinker_cookbook/recipes/sl_loop.py`.
    - **Renderers** handle chat templates, vision inputs, and per-token weight assignment. See the [Rendering docs](https://tinker-docs.thinkingmachines.ai/rendering).
    """)
    return


if __name__ == "__main__":
    app.run()
