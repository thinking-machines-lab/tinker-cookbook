import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from tinker_cookbook import model_info

    return (model_info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 201: Rendering

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

    We need a tokenizer (to map between text and token IDs) and a renderer (to apply the model's chat format). We use Inkling 2 Mini and select its renderer with `model_info.get_recommended_renderer_name()`. The cookbook uses `tml-renderers` for Inkling tokenization and message formatting.
    """)
    return


@app.cell
def _(model_info):
    from tinker_cookbook import renderers, tokenizer_utils

    MODEL_NAME = "thinkingmachines/Inkling-2-Mini"
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL_NAME), tokenizer
    )
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
    prompt = renderer.build_generation_prompt(messages[:-1], effort=0.9)
    print("ModelInput:", prompt)
    print()
    print("Decoded tokens:")
    print(tokenizer.decode(prompt.to_ints()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is a `ModelInput` containing Inkling's message boundaries and the start of a new assistant turn. Always let the renderer produce this structure rather than encoding a raw prompt or appending an assistant prefill.

    We explicitly use thinking effort `0.9`, also the default used by cookbook dataset and environment helpers. The renderer adds the effort directive. Use the same effort when rendering training data and sampling; larger efforts can need larger generation budgets.
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

    # Use the stop tokens supplied by the Inkling renderer.
    for tok in stop_sequences:
        if isinstance(tok, int):
            print(f"  Token {tok} decodes to: {tokenizer.decode([tok])!r}")
    return


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
    # Render a complete assistant turn, then remove the generation prompt.
    # This gives the parser a correctly framed response without an API call.
    _messages = [{"role": "user", "content": "How do naked mole rats live so long?"}]
    _prompt = renderer.build_generation_prompt(_messages, effort=0.9)
    _full, _weights = renderer.build_supervised_example(
        [
            *_messages,
            {
                "role": "assistant",
                "content": "They have efficient DNA repair and cancer-resistant cells.",
            },
        ],
        effort=0.9,
    )
    _full_ids = _full.to_ints()
    assert _full_ids[: _prompt.length] == _prompt.to_ints()
    fake_tokens = _full_ids[_prompt.length :]
    parsed_message, termination = renderer.parse_response(fake_tokens)

    print(f"Fake tokens: {fake_tokens}")
    print(f"Parsed message: {parsed_message}")
    print(f"Termination: {termination} (is_clean={termination.is_clean})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together: sampling a response

    Here is the full pattern for generating a message from a model. This requires a running Tinker service (and `TINKER_API_KEY`).

    ```python
    import tinker
    from tinker.types import SamplingParams

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="thinkingmachines/Inkling-2-Mini").result()

    prompt = renderer.build_generation_prompt(messages[:-1], effort=0.9)
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = SamplingParams(max_tokens=4096, temperature=1.0, stop=stop_sequences)

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
def _(messages, renderer, renderers, tokenizer):
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=0.9
    )

    # Show which tokens are prompt vs completion
    token_ids = model_input.to_ints()
    for i, (tok_id, w) in enumerate(zip(token_ids, weights.tolist())):
        label = "COMPLETION" if w > 0 else "prompt"
        print(f"  [{i:3d}] {label:10s}  {tokenizer.decode([tok_id])!r}")
    return


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

    Inkling defaults to `ALL_ASSISTANT_MESSAGES`. Above we explicitly selected `LAST_ASSISTANT_MESSAGE`. The `TrainOnWhat` enum gives you more control:

    | Value | Trains on |
    |---|---|
    | `LAST_ASSISTANT_MESSAGE` | Only the final assistant reply |
    | `LAST_ASSISTANT_TURN` | Final assistant turn including tool calls/responses |
    | `ALL_ASSISTANT_MESSAGES` | Every assistant message (Inkling default) |
    | `ALL_MESSAGES` | All messages regardless of role |
    | `ALL_TOKENS` | Every token including special tokens |
    | `CUSTOMIZED` | Per-message `train` flags from the dataset |
    """)
    return


@app.cell
def _(messages, renderer, renderers):
    # Whole-conversation rendering can produce more than one SFT example.
    _examples_all = renderer.build_supervised_examples(
        messages,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort=0.9,
    )
    _all_count = sum((_weights > 0).sum().item() for _, _weights in _examples_all)
    print(f"Tokens with weight > 0 (all assistant messages): {_all_count}")

    _, weights_last = renderer.build_supervised_example(
        messages, train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=0.9
    )
    print(f"Tokens with weight > 0 (last assistant message): {(weights_last > 0).sum().item()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Available renderers

    Tinker ships renderers for several model families. Use `get_renderer()` with the appropriate name:

    | Name | Model family | Notes |
    |---|---|---|
    | `tml_v0` | Inkling, including Inkling 2 Mini | Thinking effort via `effort` |
    | `qwen3_5` | Qwen3.5 / Qwen3.6 (incl. VL) | Thinking enabled (default) |
    | `qwen3_5_disable_thinking` | Qwen3.5 / Qwen3.6 (incl. VL) | Thinking disabled |
    | `qwen3_5_preserve_thinking` | Qwen3.5 / Qwen3.6 (incl. VL) | Thinking enabled, earlier reasoning kept in history (multi-turn RL) |
    | `qwen3_8_xhigh_reasoning` | Qwen3.8 (incl. VL) | Thinking enabled, reasoning effort xhigh (default) |
    | `qwen3_8_disable_thinking` | Qwen3.8 (incl. VL) | Thinking disabled |
    | `glm5_3_max_reasoning` | GLM-5.3 / GLM-5.3-Flash | Max reasoning effort (default) |
    | `glm5_3_low_reasoning` | GLM-5.3 / GLM-5.3-Flash | Low reasoning effort (thinking cannot be disabled) |
    | `deepseekv3` | DeepSeek V3 | Non-thinking mode (default) |
    | `deepseekv3_thinking` | DeepSeek V3 | Thinking mode |
    | `deepseekv4_1` | DeepSeek V4.1 | Thinking enabled, reasoning effort high (default) |
    | `deepseekv4_1_disable_thinking` | DeepSeek V4.1 | Thinking disabled |
    | `deepseekv4_1_preserve_thinking` | DeepSeek V4.1 | Thinking enabled, earlier reasoning kept in history (multi-turn RL) |
    | `nemotron3` | NVIDIA Nemotron 3 Nano/Super | Thinking enabled |
    | `nemotron3_ultra` | NVIDIA Nemotron 3 Ultra / 3.5 Lightning / 3.5 Super | Thinking enabled |
    | `kimi_k26` | Kimi K2.6 | Thinking enabled (default) |
    | `kimi_k26_disable_thinking` | Kimi K2.6 | Thinking disabled |
    | `kimi_k31_pro_max_reasoning` | Kimi K3.1-Pro | Thinking enabled, reasoning effort max (default) |
    | `kimi_k31_pro_disable_thinking` | Kimi K3.1-Pro | Thinking disabled |

    Each renderer produces the correct special tokens for its model family. Inkling uses `tml-renderers`; other families use their corresponding chat templates. Select the recommended renderer for the model rather than reusing another family's format.
    """)
    return


@app.cell
def _(model_info, renderers, tokenizer_utils):
    # Example: switching between renderers
    # Each model family needs its own tokenizer + matching renderer
    _test_messages = [{"role": "user", "content": "Hello!"}]

    for _model_name in ["thinkingmachines/Inkling-2-Mini", "thinkingmachines/Inkling-2-Small"]:
        _renderer_name = model_info.get_recommended_renderer_name(_model_name)
        _tokenizer = tokenizer_utils.get_tokenizer(_model_name)
        _renderer = renderers.get_renderer(_renderer_name, _tokenizer)
        _prompt_tokens = _renderer.build_generation_prompt(_test_messages, effort=0.9)
        print(f"--- {_model_name} ({_renderer_name}) ---")
        print(_tokenizer.decode(_prompt_tokens.to_ints()))
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vision inputs with `ImagePart`

    For vision-language models such as Inkling 2 Mini, message content can include images alongside text. Use `ImagePart` for images and `TextPart` for text within the same message.
    """)
    return


@app.cell
def _():
    from PIL import Image

    from tinker_cookbook.renderers import ImagePart, Message, TextPart

    # A multimodal message with an image and text
    multimodal_message = Message(
        role="user",
        content=[
            ImagePart(type="image", image=Image.new("RGB", (64, 64), color="blue")),
            TextPart(type="text", text="What is in this image?"),
        ],
    )
    print("Multimodal message:", multimodal_message)

    # Text-only messages still work as plain strings
    text_message = Message(role="user", content="Describe this in one word.")
    print("Text message:", text_message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same Inkling renderer handles images through `tml-renderers`; no HuggingFace image processor is needed. Use an in-memory image, local file, or data URI rather than a remote URL.

    ```python
    image_prompt = renderer.build_generation_prompt([multimodal_message], effort=0.9)
    ```

    The result can contain image chunks as well as text tokens, so do not call `to_ints()` on a multimodal prompt.
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
    # Define a factory function that creates your renderer
    def my_renderer_factory(tokenizer, image_processor=None):
        # In practice, you would return a custom Renderer subclass here.
        # For demonstration, we delegate to the Inkling renderer.
        from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

        return TmlV0Renderer(tokenizer)

    # Register it under a namespaced name
    renderers.register_renderer("MyOrg/custom_format", my_renderer_factory)

    # Now you can use it via get_renderer
    print(f"Registered renderers: {renderers.get_registered_renderer_names()}")

    # Clean up
    renderers.unregister_renderer("MyOrg/custom_format")
    return


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
