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
    # Tutorial: Rendering

    Rendering converts a list of messages into a token sequence that a model can consume. While similar to HuggingFace chat templates, Tinker's rendering system handles the full training lifecycle: supervised learning, reinforcement learning, and deployment.

    The renderer sits between your high-level conversation data and the low-level tokens the model sees:

    ```
    Messages (list of dicts)  -->  Renderer  -->  Token IDs (list of ints)
    ```

    This tutorial covers the `Renderer` class and its key methods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    We need a tokenizer (to map between text and token IDs) and a renderer (to apply the model's chat format).
    """)
    return


@app.cell
def _():
    from tinker_cookbook import renderers, tokenizer_utils

    tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = renderers.get_renderer("qwen3", tokenizer)
    renderer  # noqa: B018
    return renderer, renderers, tokenizer, tokenizer_utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example conversation

    We will use this conversation throughout the tutorial.
    """)
    return


@app.cell
def _():
    messages = [
        {"role": "system", "content": "Answer concisely; at most one sentence per response"},
        {"role": "user", "content": "What is the longest-lived rodent species?"},
        {"role": "assistant", "content": "The naked mole rat, which can live over 30 years."},
        {"role": "user", "content": "How do they live so long?"},
        {
            "role": "assistant",
            "content": "They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.",
        },
    ]
    return (messages,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `build_generation_prompt()` -- for sampling

    Converts a conversation into a token prompt ready for the model to continue. This is used during RL rollouts and at deployment time.

    Typically you pass all messages *except* the final assistant reply, so the model generates its own response.
    """)
    return


@app.cell
def _(messages, renderer, tokenizer):
    # Remove the last assistant message so the model can generate one
    prompt = renderer.build_generation_prompt(messages[:-1])
    print("ModelInput:", prompt)
    print()
    print("Decoded tokens:")
    print(tokenizer.decode(prompt.to_ints()))
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is a `ModelInput` object containing the tokenized chat template. Notice how each message is wrapped in special tokens like `<|im_start|>` and `<|im_end|>`, and the final `<|im_start|>assistant` is left open for the model to fill in.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `get_stop_sequences()` -- stop tokens

    When sampling, we need to know when the model has finished its response. `get_stop_sequences()` returns the token IDs (or strings) that signal end-of-generation.
    """)
    return


@app.cell
def _(renderer, tokenizer):
    stop_sequences = renderer.get_stop_sequences()
    print(f"Stop sequences: {stop_sequences}")

    # For Qwen3, this is the <|im_end|> token
    for tok in stop_sequences:
        if isinstance(tok, int):
            print(f"  Token {tok} decodes to: {tokenizer.decode([tok])!r}")
    return (stop_sequences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `parse_response()` -- decoding tokens back to a message

    After sampling, you get raw token IDs. `parse_response()` converts them back into a structured message dict and a `ParseTermination` enum that tells you how the response ended:

    - `STOP_SEQUENCE` — the renderer's expected stop signal fired (e.g. `<|im_end|>` for chat templates, `\n\nUser:` for RoleColon).
    - `EOS` — the model emitted EOS instead. Some renderers (notably `RoleColonRenderer` for base models) accept this as a clean parse on single-turn prompts.
    - `MALFORMED` — no clean termination (truncated, or multiple/conflicting stop signals).

    Use `termination.is_clean` (any clean termination — what eval grading reads) or `termination.is_stop_sequence` (strict — what RL format-reward shaping reads).
    """)
    return


@app.cell
def _(renderer):
    # Simulate some sampled tokens (in practice these come from the model)
    fake_tokens = [45, 7741, 34651, 31410, 614, 4911, 76665, 13, 151645]

    parsed_message, termination = renderer.parse_response(fake_tokens)
    print(f"Parsed message: {parsed_message}")
    print(f"Termination: {termination} (is_clean={termination.is_clean})")
    return fake_tokens, parsed_message, termination


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together: sampling a response

    Here is the full pattern for generating a message from a model. This requires a running Tinker service (and `TINKER_API_KEY`).

    ```python
    import tinker
    from tinker.types import SamplingParams

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")

    prompt = renderer.build_generation_prompt(messages[:-1])
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = SamplingParams(max_tokens=100, temperature=0.5, stop=stop_sequences)

    output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
    sampled_message, success = renderer.parse_response(output.sequences[0].tokens)
    print(sampled_message)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `build_supervised_example()` -- for training

    For supervised fine-tuning, we need to distinguish **prompt tokens** (context the model reads) from **completion tokens** (what the model should learn to produce). `build_supervised_example()` returns both the tokens and per-token loss weights.

    - Weight `0` = prompt (no loss computed)
    - Weight `1` = completion (model trains on these)
    """)
    return


@app.cell
def _(messages, renderer, tokenizer):
    model_input, weights = renderer.build_supervised_example(messages)

    # Show which tokens are prompt vs completion
    token_ids = model_input.to_ints()
    for i, (tok_id, w) in enumerate(zip(token_ids, weights.tolist())):
        label = "COMPLETION" if w > 0 else "prompt"
        print(f"  [{i:3d}] {label:10s}  {tokenizer.decode([tok_id])!r}")
    return model_input, token_ids, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only the final assistant message has weight 1 (completion). Everything else -- system prompt, user messages, and even earlier assistant messages -- has weight 0. This way the loss only encourages the model to produce the correct response, without overfitting to the prompt content (system instructions, questions) which the model should not need to memorize.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `TrainOnWhat` -- controlling loss targets

    By default, `build_supervised_example` trains on the last assistant message. The `TrainOnWhat` enum gives you more control:

    | Value | Trains on |
    |---|---|
    | `LAST_ASSISTANT_MESSAGE` | Only the final assistant reply (default) |
    | `LAST_ASSISTANT_TURN` | Final assistant turn including tool calls/responses |
    | `ALL_ASSISTANT_MESSAGES` | Every assistant message in the conversation |
    | `ALL_MESSAGES` | All messages regardless of role |
    | `ALL_TOKENS` | Every token including special tokens |
    | `CUSTOMIZED` | Per-message `train` flags from the dataset |
    """)
    return


@app.cell
def _(messages, renderer, renderers):
    # Train on ALL assistant messages instead of just the last one
    _, weights_all = renderer.build_supervised_example(
        messages,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    print(f"Tokens with weight > 0: {(weights_all > 0).sum().item()}")

    # Compare with default (last assistant message only)
    _, weights_last = renderer.build_supervised_example(messages)
    print(f"Tokens with weight > 0 (default): {(weights_last > 0).sum().item()}")
    return weights_all, weights_last


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Available renderers

    Tinker ships renderers for several model families. Use `get_renderer()` with the appropriate name:

    | Name | Model family | Notes |
    |---|---|---|
    | `qwen3` | Qwen3 | Thinking enabled (default) |
    | `qwen3_disable_thinking` | Qwen3 | Thinking disabled |
    | `llama3` | Llama 3 | Omits the HF preamble |
    | `deepseekv3` | DeepSeek V3 | Non-thinking mode (default) |
    | `deepseekv3_thinking` | DeepSeek V3 | Thinking mode |
    | `nemotron3` | NVIDIA Nemotron 3 | Thinking enabled |
    | `kimi_k2` | Kimi K2 | Thinking format |

    Each renderer produces the correct special tokens for its model family. The default renderers match HuggingFace's `apply_chat_template` output, so models trained with Tinker work with the OpenAI-compatible endpoint.
    """)
    return


@app.cell
def _(renderers, tokenizer_utils):
    # Example: switching between renderers
    # Each model family needs its own tokenizer
    qwen_tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3-30B-A3B")
    qwen_renderer = renderers.get_renderer("qwen3", qwen_tokenizer)

    test_messages = [{"role": "user", "content": "Hello!"}]
    prompt_tokens = qwen_renderer.build_generation_prompt(test_messages)
    print("Qwen3 prompt:")
    print(qwen_tokenizer.decode(prompt_tokens.to_ints()))
    return prompt_tokens, qwen_renderer, qwen_tokenizer, test_messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vision inputs with `ImagePart`

    For vision-language models (like Qwen3-VL), message content can include images alongside text. Use `ImagePart` for images and `TextPart` for text within the same message.
    """)
    return


@app.cell
def _(renderers):
    from tinker_cookbook.renderers import ImagePart, Message, TextPart

    # A multimodal message with an image and text
    multimodal_message = Message(
        role="user",
        content=[
            ImagePart(type="image", image="https://example.com/photo.png"),
            TextPart(type="text", text="What is in this image?"),
        ],
    )
    print("Multimodal message:", multimodal_message)

    # Text-only messages still work as plain strings
    text_message = Message(role="user", content="Describe this in one word.")
    print("Text message:", text_message)
    return ImagePart, Message, TextPart, multimodal_message, text_message


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use vision renderers, you also need an image processor:

    ```python
    from tinker_cookbook.image_processing_utils import get_image_processor

    model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    image_processor = get_image_processor(model_name)

    renderer = renderers.get_renderer("qwen3_vl_instruct", tokenizer, image_processor=image_processor)
    ```

    The VL renderers handle vision special tokens (`<|vision_start|>`, `<|vision_end|>`) and image preprocessing automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom renderers with `register_renderer()`

    If you need a format not covered by the built-in renderers, you can register your own. This lets you use `get_renderer()` with a custom name throughout your codebase.
    """)
    return


@app.cell
def _(renderers):
    from tinker_cookbook.renderers.base import Renderer

    # Define a factory function that creates your renderer
    def my_renderer_factory(tokenizer, image_processor=None):
        # In practice, you would return a custom Renderer subclass here.
        # For demonstration, we just return the Qwen3 renderer.
        from tinker_cookbook.renderers.qwen3 import Qwen3Renderer

        return Qwen3Renderer(tokenizer)

    # Register it under a namespaced name
    renderers.register_renderer("MyOrg/custom_format", my_renderer_factory)

    # Now you can use it via get_renderer
    print(f"Registered renderers: {renderers.get_registered_renderer_names()}")

    # Clean up
    renderers.unregister_renderer("MyOrg/custom_format")
    return Renderer, my_renderer_factory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    The renderer is the bridge between conversations and tokens. Its four key methods cover the full lifecycle:

    | Method | Purpose | Used in |
    |---|---|---|
    | `build_generation_prompt()` | Messages to prompt tokens | RL, inference |
    | `get_stop_sequences()` | End-of-generation tokens | Sampling |
    | `parse_response()` | Tokens back to a message | RL, inference |
    | `build_supervised_example()` | Messages to tokens + loss weights | SFT, DPO |

    Use `get_renderer(name, tokenizer)` to get the right renderer for your model, and `TrainOnWhat` to control which parts of the conversation the model trains on.
    """)
    return


if __name__ == "__main__":
    app.run()
