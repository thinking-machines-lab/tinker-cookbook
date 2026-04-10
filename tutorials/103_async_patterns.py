import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 03: Efficient Sampling with Tinker

    Tinker runs on remote GPUs. Every API call involves network latency plus GPU compute time. If you send sampling requests one at a time -- send, wait, send, wait -- you spend most of your time idle while Tinker works.

    The solution: send requests **concurrently** as futures. Tinker can batch and pipeline concurrent requests on the GPU, so N requests take far less than N times the cost of one request. This matters most for sampling, where RL training may require hundreds of completions per step.
    """)
    return


@app.cell
def _():
    import time
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker

    from tinker_cookbook.renderers import get_renderer, get_text_content

    return get_renderer, get_text_content, time, tinker


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    We create a `SamplingClient` for the base Qwen3.5-4B model (no fine-tuning needed for this tutorial). We also set up a renderer to handle the chat template and a list of diverse prompts to sample from.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Paste your Tinker API key")
    api_key
    return (api_key,)


@app.cell
async def _(api_key, get_renderer, mo, tinker):
    import os

    mo.stop(
        "TINKER_API_KEY" not in os.environ and not api_key.value,
        "Paste your API key above",
    )

    if api_key.value:
        os.environ["TINKER_API_KEY"] = api_key.value

    BASE_MODEL = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=BASE_MODEL
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer("qwen3_5", tokenizer)

    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(
        max_tokens=150, temperature=0.7, stop=stop_sequences
    )

    # A diverse set of prompts to sample from
    prompts = [
        "What causes thunder?",
        "Write a haiku about the ocean.",
        "What is the capital of New Zealand?",
        "Explain what a hash table is in two sentences.",
        "Name three inventions from the 19th century.",
        "Why do leaves change color in autumn?",
        "Translate to Spanish: The library closes at nine.",
        "What is the smallest prime number greater than 50?",
    ]

    print(f"Model: {BASE_MODEL}")
    print(f"Prompts: {len(prompts)}")
    return params, prompts, renderer, sampling_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequential sampling (the slow way)

    The simplest approach: for each prompt, build the generation input, call `sample()`, immediately call `.result()` to block until it finishes, then move on to the next. Each request waits for the previous one to complete before starting.
    """)
    return


@app.cell
async def _(
    get_text_content,
    params,
    prompts,
    renderer,
    sampling_client,
    time,
):
    _start = time.time()
    sequential_results = []
    for _prompt_text in prompts:
        _messages = [{"role": "user", "content": _prompt_text}]
        _model_input = renderer.build_generation_prompt(_messages)
        _result = await sampling_client.sample_async(
            prompt=_model_input, num_samples=1, sampling_params=params
        )
        _response_msg, _ = renderer.parse_response(_result.sequences[0].tokens)
        sequential_results.append(
            get_text_content(_response_msg)
        )  # Block on each request before sending the next
    sequential_time = time.time() - _start
    for _prompt_text, _answer in zip(prompts, sequential_results):
        print(f"Q: {_prompt_text}")
        print(f"A: {_answer[:120]}...\n")
    print(
        f"Sequential: {sequential_time:.1f}s for {len(prompts)} prompts ({sequential_time / len(prompts):.1f}s each)"
    )
    return (sequential_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Concurrent sampling with futures

    `sample()` returns a future immediately -- the request is already in flight before you call `.result()`. The key insight: submit **all** requests first, then collect results. Tinker batches concurrent requests on the GPU for higher throughput.
    """)
    return


@app.cell
async def _(
    asyncio,
    get_text_content,
    params,
    prompts,
    renderer,
    sampling_client,
    sequential_time,
    time,
):
    _start = time.time()

    # Step 1: Submit ALL requests concurrently using asyncio.gather
    async def _sample_one(_prompt_text):
        _messages = [{"role": "user", "content": _prompt_text}]
        _model_input = renderer.build_generation_prompt(_messages)
        return await sampling_client.sample_async(
            prompt=_model_input, num_samples=1, sampling_params=params
        )

    _results = await asyncio.gather(*[_sample_one(p) for p in prompts])
    concurrent_results = []
    for _result in _results:
        # Step 2: Parse results (all requests were running in parallel)
        _response_msg, _ = renderer.parse_response(_result.sequences[0].tokens)
        concurrent_results.append(get_text_content(_response_msg))
    concurrent_time = time.time() - _start
    for _prompt_text, _answer in zip(prompts, concurrent_results):
        print(f"Q: {_prompt_text}")
        print(f"A: {_answer[:120]}...\n")
    print(f"Concurrent: {concurrent_time:.1f}s for {len(prompts)} prompts")
    print(f"Sequential: {sequential_time:.1f}s for {len(prompts)} prompts")
    print(f"Speedup: {sequential_time / concurrent_time:.1f}x")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple completions per prompt (num_samples)

    In GRPO-style RL, you need `group_size` independent completions for each problem so you can compare them and compute advantages. The `num_samples` parameter generates multiple completions in a single API call -- more efficient than sending separate requests for the same prompt.
    """)
    return


@app.cell
async def _(get_text_content, params, renderer, sampling_client, time):
    _GROUP_SIZE = 4
    test_prompt = "Name a famous scientist and explain their key contribution in one sentence."
    _messages = [{"role": "user", "content": test_prompt}]
    _model_input = renderer.build_generation_prompt(_messages)
    _start = time.time()
    _result = await sampling_client.sample_async(
        prompt=_model_input, num_samples=_GROUP_SIZE, sampling_params=params
    )
    # Single call with num_samples=4 -- generates 4 independent completions
    multi_time = time.time() - _start
    print(f"Prompt: {test_prompt}\n")
    for i, _seq in enumerate(_result.sequences):
        _response_msg, _ = renderer.parse_response(_seq.tokens)
        text = get_text_content(_response_msg)
        print(f"Completion {i + 1}: {text[:150]}\n")
    _start = time.time()
    for _ in range(_GROUP_SIZE):
        await sampling_client.sample_async(
            prompt=_model_input, num_samples=1, sampling_params=params
        )
    sequential_multi_time = time.time() - _start
    print(f"num_samples={_GROUP_SIZE} in one call: {multi_time:.1f}s")
    print(f"{_GROUP_SIZE} sequential calls:        {sequential_multi_time:.1f}s")
    # Compare: 4 sequential single calls
    print(f"Speedup: {sequential_multi_time / multi_time:.1f}x")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together: batch evaluation

    Combine both techniques -- concurrent futures across prompts and `num_samples` per prompt -- for maximum throughput. This is exactly the pattern used in RL training: submit many sampling requests in parallel, each generating a group of completions, then collect and grade them all.
    """)
    return


@app.cell
async def _(
    get_text_content,
    params,
    prompts,
    renderer,
    sampling_client,
    time,
):
    import asyncio

    _GROUP_SIZE = 4
    _start = time.time()

    # Submit all requests concurrently using asyncio.gather, each with num_samples=GROUP_SIZE
    async def _sample_group(_prompt_text):
        _messages = [{"role": "user", "content": _prompt_text}]
        _model_input = renderer.build_generation_prompt(_messages)
        _result = await sampling_client.sample_async(
            prompt=_model_input, num_samples=_GROUP_SIZE, sampling_params=params
        )
        return _prompt_text, _result

    _results = await asyncio.gather(*[_sample_group(p) for p in prompts])
    total_completions = 0
    for _prompt_text, _result in _results:
        completions = []
        for _seq in _result.sequences:
            # Collect all results
            _response_msg, _ = renderer.parse_response(_seq.tokens)
            completions.append(get_text_content(_response_msg))
        total_completions += len(completions)
        print(f"Q: {_prompt_text}")
        print(f"   ({len(completions)} completions, showing first): {completions[0][:100]}...\n")
    batch_time = time.time() - _start
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / batch_time:.1f} completions/second")
    return (asyncio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    This tutorial showed the two key techniques for efficient sampling: **concurrent futures** (submit all requests before collecting results) and **num_samples** (generate multiple completions per call). Together, they give you high throughput with minimal code changes.

    - **Tutorial 04** (`104_first_rl.py`): Uses this exact pattern -- sample many completions, grade them with a reward function, and train with GRPO.
    - **Async docs** (`docs/async.mdx`): Full reference for sync/async APIs, the double-await pattern, and overlapping training requests.
    """)
    return


if __name__ == "__main__":
    app.run()
