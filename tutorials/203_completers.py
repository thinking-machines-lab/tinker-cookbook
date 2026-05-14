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
    # Tutorial 203: Completers

    Completers are thin wrappers around `SamplingClient` that provide two levels of abstraction:

    - **TokenCompleter** -- operates on token IDs and `ModelInput`. Used by RL algorithms that work at the token level.
    - **MessageCompleter** -- operates on message dicts (role/content). Used by evaluators, LLM-as-judge patterns, and chat applications.

    In this tutorial you will:

    1. Build a `TinkerTokenCompleter` from a `SamplingClient`
    2. Use it to generate tokens with stop conditions
    3. Build a `TinkerMessageCompleter` with a renderer
    4. Use it to generate structured message responses
    5. Implement a simple LLM-as-judge pattern
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import asyncio

    import tinker

    from tinker_cookbook.completers import (
        MessageCompleter,
        TinkerMessageCompleter,
        TinkerTokenCompleter,
        TokenCompleter,
        TokensWithLogprobs,
    )
    from tinker_cookbook.renderers import get_renderer, get_text_content

    return (
        MessageCompleter,
        TinkerMessageCompleter,
        TinkerTokenCompleter,
        TokenCompleter,
        TokensWithLogprobs,
        asyncio,
        get_renderer,
        get_text_content,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TokenCompleter vs MessageCompleter

    ```
    TokenCompleter                          MessageCompleter
    +--------------------------+            +---------------------------+
    | Input:  ModelInput        |            | Input:  list[Message]     |
    |         (token IDs)       |            |         (role + content)  |
    | Output: TokensWithLogprobs|            | Output: Message           |
    |         (tokens + logps)  |            |         (role + content)  |
    +--------------------------+            +---------------------------+
          Used by RL loops                    Used by evals / judges
    ```

    `TokenCompleter` gives you raw tokens and log-probabilities -- essential for computing advantages and building RL datums. `MessageCompleter` hides the tokenization details and speaks the language of conversations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Create a sampling client and a renderer. We will use these throughout the tutorial.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key  # noqa: B018
    return (api_key,)


@app.cell
def _(api_key, get_renderer, mo, tinker):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    MODEL_NAME = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer)

    print(f"Sampling client ready for {MODEL_NAME}")
    return MODEL_NAME, renderer, sampling_client, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TinkerTokenCompleter

    `TinkerTokenCompleter` wraps a `SamplingClient` and exposes the `TokenCompleter` interface. You pass a `ModelInput` (tokenized prompt) and a stop condition (token IDs or strings).
    """)
    return


@app.cell
def _(TinkerTokenCompleter, sampling_client):
    token_completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=128,
        temperature=0.7,
    )
    print(
        f"TokenCompleter: max_tokens={token_completer.max_tokens}, temp={token_completer.temperature}"
    )
    return (token_completer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate tokens with stop conditions

    The `TokenCompleter` is an async callable. We pass a `ModelInput` and stop sequences. The result is a `TokensWithLogprobs` with the generated token IDs, their log-probabilities, and the stop reason.
    """)
    return


@app.cell
def _(asyncio, renderer, token_completer, tokenizer):
    # Build a prompt from messages
    messages_for_tokens = [
        {"role": "user", "content": "What is 7 * 8?"},
    ]
    model_input = renderer.build_generation_prompt(messages_for_tokens)
    stop_sequences = renderer.get_stop_sequences()

    # Generate tokens
    token_result = asyncio.run(token_completer(model_input, stop=stop_sequences))

    print(f"Generated {len(token_result.tokens)} tokens")
    print(f"Stop reason: {token_result.stop_reason}")
    print(f"Log-probs (first 5): {token_result.logprobs[:5]}")
    print(f"Decoded: {tokenizer.decode(token_result.tokens)}")
    return messages_for_tokens, model_input, stop_sequences, token_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The log-probabilities are always available on `TinkerTokenCompleter` results. In RL, these are used as the sampling logprobs for importance sampling correction:

    ```python
    sampling_logprobs = token_result.logprobs  # from the sampler
    # Later, forward_backward computes target_logprobs from the learner
    # The ratio exp(target - sampling) corrects for off-policy data
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TinkerMessageCompleter

    `TinkerMessageCompleter` wraps a `SamplingClient` and a `Renderer` to speak the message-level protocol. You pass a list of message dicts; it handles rendering, sampling, and parsing internally.
    """)
    return


@app.cell
def _(TinkerMessageCompleter, renderer, sampling_client):
    message_completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=256,
        temperature=0.7,
    )
    print("MessageCompleter ready")
    return (message_completer,)


@app.cell
def _(asyncio, get_text_content, message_completer):
    # Generate a message response
    conversation = [
        {"role": "user", "content": "Explain what a hash table is in one sentence."},
    ]

    response = asyncio.run(message_completer(conversation))
    print(f"Role: {response['role']}")
    print(f"Content: {get_text_content(response)}")
    return conversation, response


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-turn conversations

    `MessageCompleter` handles multi-turn conversations naturally -- just pass the full message history.
    """)
    return


@app.cell
def _(asyncio, get_text_content, message_completer):
    multi_turn = [
        {"role": "user", "content": "What is the largest planet in our solar system?"},
        {"role": "assistant", "content": "Jupiter."},
        {"role": "user", "content": "How many moons does it have?"},
    ]

    followup = asyncio.run(message_completer(multi_turn))
    print(f"Response: {get_text_content(followup)}")
    return followup, multi_turn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-judge pattern

    A common evaluation pattern uses one model as a "judge" to score outputs from another model (or the same model at a different checkpoint). The `MessageCompleter` makes this straightforward.

    The pattern:
    1. Generate a candidate answer using the model under evaluation
    2. Ask the judge to score it
    3. Parse the score from the judge's response
    """)
    return


@app.cell
def _(asyncio, get_text_content, message_completer):
    import re

    # Step 1: Generate a candidate answer
    question = "Why do leaves change color in autumn?"
    candidate = asyncio.run(message_completer([{"role": "user", "content": question}]))
    candidate_text = get_text_content(candidate)
    print(f"Candidate answer:\n{candidate_text}\n")

    # Step 2: Ask the judge to score it
    judge_prompt = f"""Rate the following answer on a scale of 1-5 for accuracy and clarity.

Question: {question}
Answer: {candidate_text}

Respond with ONLY a number from 1 to 5."""

    judge_response = asyncio.run(message_completer([{"role": "user", "content": judge_prompt}]))
    judge_text = get_text_content(judge_response)

    # Step 3: Parse the score
    match = re.search(r"[1-5]", judge_text)
    score = int(match.group()) if match else None
    print(f"Judge response: {judge_text}")
    print(f"Parsed score: {score}")
    return (
        candidate,
        candidate_text,
        judge_prompt,
        judge_response,
        judge_text,
        match,
        question,
        re,
        score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the judge as a reward function

    In RL training, you can wrap this judge pattern into a reward function:

    ```python
    async def judge_reward(message_completer, question, answer):
        judge_prompt = f"Rate this answer 1-5.\nQ: {question}\nA: {answer}\nScore:"
        response = await message_completer([{"role": "user", "content": judge_prompt}])
        text = get_text_content(response)
        match = re.search(r"[1-5]", text)
        return float(match.group()) / 5.0 if match else 0.0  # normalize to [0, 1]
    ```

    This is especially useful when you have a stronger model judging a weaker model's outputs, or when your reward function cannot be expressed as a simple string match.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Class | Input | Output | Use case |
    |---|---|---|---|
    | `TinkerTokenCompleter` | `ModelInput` + stop tokens | `TokensWithLogprobs` | RL rollouts, token-level control |
    | `TinkerMessageCompleter` | `list[Message]` | `Message` | Evals, judges, chat apps |

    Both are async callables that wrap a `SamplingClient`. `TokenCompleter` gives you log-probabilities for RL; `MessageCompleter` handles rendering and parsing for you.

    You can also implement the `TokenCompleter` or `MessageCompleter` interfaces with non-Tinker backends (e.g., a local vLLM server) for testing or hybrid setups.
    """)
    return


if __name__ == "__main__":
    app.run()
