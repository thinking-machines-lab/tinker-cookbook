import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 08: Why RL? From Rejection Sampling to GRPO

    You have a model and a reward function that can verify whether an answer is correct.
    How should you train?

    The simplest approach is **Rejection Sampling Fine-Tuning (RFT)**: sample solutions,
    keep the correct ones, fine-tune with standard SFT loss. It's fast and stable.

    But on harder tasks, RFT hits a ceiling. **GRPO** (Group Relative Policy Optimization)
    breaks through that ceiling by learning from *both* correct and incorrect solutions.

    This tutorial runs both methods head-to-head on GSM8K math problems, then shows
    what happens on harder tasks (Hendrycks MATH).

    **What you'll learn:**

    1. How RFT and GRPO work -- with runnable code
    2. A live head-to-head comparison on GSM8K
    3. Why RFT plateaus and GRPO doesn't
    4. When to use which method
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The two approaches

    Both RFT and GRPO start the same way: sample K solutions per problem, then grade them.
    The difference is in **what they train on**.

    | | RFT | GRPO |
    |---|---|---|
    | **Correct solutions** | Train with SFT loss | Upweight (positive advantage) |
    | **Wrong solutions** | Throw away | Downweight (negative advantage) |
    | **Loss function** | Cross-entropy | Importance-weighted policy gradient |
    | **Intuition** | "Do more of this" | "Do more of this, less of that" |

    RFT is just SFT on the model's own correct outputs. GRPO is RL that learns from
    the full distribution of outputs -- correct and incorrect.
    """)
    return


# ==============================================================================
# Setup: model, data, grading
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    We use **Qwen3-8B** on **GSM8K** (grade school math). Each problem has a numeric
    answer that we can verify automatically -- a perfect setting for both RFT and GRPO.
    """)
    return


@app.cell
def _():
    import re
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker
    import torch
    from tinker import TensorData

    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.renderers.base import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    return (
        TensorData,
        conversation_to_datum,
        TrainOnWhat,
        get_renderer,
        get_text_content,
        re,
        tinker,
        torch,
    )


@app.cell
def _(re):
    import datasets

    _dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_data = _dataset["train"]
    test_data = _dataset["test"]

    def extract_gsm8k_answer(text: str) -> str:
        match = re.search(r"####\s*(.+)", text)
        if match:
            return match.group(1).replace(",", "").strip()
        raise ValueError("No #### answer found")

    def extract_boxed(text: str) -> str | None:
        match = re.findall(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match[-1].strip()
        return None

    def grade_answer(response: str, ground_truth: str) -> float:
        answer = extract_boxed(response)
        if answer is None:
            return 0.0
        answer = answer.replace(",", "").strip()
        ground_truth = ground_truth.replace(",", "").strip()
        return 1.0 if answer == ground_truth else 0.0

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    fewshot_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {
            "role": "assistant",
            "content": (
                "Let's count: s-t-r-a-w-b-e-r-r-y. "
                "Positions 3, 8, 9. \\boxed{3}"
            ),
        },
    ]

    print(f"Loaded {len(train_data)} train / {len(test_data)} test GSM8K problems")
    return (
        extract_boxed,
        extract_gsm8k_answer,
        fewshot_prefix,
        grade_answer,
        question_suffix,
        test_data,
        train_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training configuration

    We use the same hyperparameters as the full experiments in `recipes/math_rft/`
    so the results are directly comparable. Both methods use the same model, data,
    and group size.
    """)
    return


@app.cell
def _():
    # Shared hyperparameters -- matched to the full experiments
    N_STEPS = 30          # training steps per method
    BATCH_SIZE = 32       # problems per step (groups_per_batch)
    GROUP_SIZE = 16       # completions per problem
    MAX_TOKENS = 1024     # max generation tokens
    BASE_MODEL = "Qwen/Qwen3-8B"
    EVAL_EVERY = 5        # evaluate on test set every N steps
    N_EVAL_PROBLEMS = 200 # test problems for evaluation

    return BASE_MODEL, BATCH_SIZE, EVAL_EVERY, GROUP_SIZE, MAX_TOKENS, N_EVAL_PROBLEMS, N_STEPS


# ==============================================================================
# Part 1: RFT training loop
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: RFT (Rejection Sampling Fine-Tuning)

    The RFT loop:
    1. Sample K solutions per problem
    2. Grade them -- keep only correct ones
    3. Run standard SFT (cross-entropy) on the correct solutions
    4. Evaluate on the test set periodically

    No advantages, no importance weights, no negative signal.
    Just **SFT on whatever the model gets right**.
    """)
    return


@app.cell
async def _(
    BASE_MODEL,
    BATCH_SIZE,
    EVAL_EVERY,
    GROUP_SIZE,
    MAX_TOKENS,
    N_EVAL_PROBLEMS,
    N_STEPS,
    TrainOnWhat,
    conversation_to_datum,
    extract_gsm8k_answer,
    fewshot_prefix,
    get_renderer,
    get_text_content,
    grade_answer,
    question_suffix,
    test_data,
    tinker,
    train_data,
):
    import asyncio as _asyncio
    import time as _time

    async def _run_rft():
        # --- Setup ---
        service = tinker.ServiceClient()
        tc = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=32)
        tok = tc.get_tokenizer()
        rdr = get_renderer("qwen3", tok)
        adam = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95)
        samp_params = tinker.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=1.0, stop=rdr.get_stop_sequences()
        )

        async def _evaluate(training_client):
            sc = await training_client.save_weights_and_get_sampling_client_async()
            ep = tinker.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=0.0, stop=rdr.get_stop_sequences()
            )
            async def _one(row):
                convo = [*fewshot_prefix, {"role": "user", "content": row["question"] + question_suffix}]
                r = await sc.sample_async(
                    prompt=rdr.build_generation_prompt(convo), num_samples=1, sampling_params=ep
                )
                msg, _ = rdr.parse_response(r.sequences[0].tokens)
                return grade_answer(get_text_content(msg), extract_gsm8k_answer(row["answer"]))
            problems = test_data.select(range(min(N_EVAL_PROBLEMS, len(test_data))))
            scores = await _asyncio.gather(*[_one(row) for row in problems])
            return sum(scores) / len(scores)

        # --- Training loop ---
        metrics = []
        t0 = _time.time()

        for step in range(N_STEPS):
            t_step = _time.time()
            batch_start = step * BATCH_SIZE
            batch_rows = train_data.select(range(batch_start, batch_start + BATCH_SIZE))

            sampling_client = await tc.save_weights_and_get_sampling_client_async()

            async def _sample(question):
                convo = [*fewshot_prefix, {"role": "user", "content": question + question_suffix}]
                prompt = rdr.build_generation_prompt(convo)
                result = await sampling_client.sample_async(
                    prompt=prompt, num_samples=GROUP_SIZE, sampling_params=samp_params
                )
                return result, convo

            results = await _asyncio.gather(*[_sample(q) for q in batch_rows["question"]])

            correct_datums = []
            n_correct = 0
            n_total = 0
            n_solved = 0

            for (sample_result, convo), answer_text in zip(results, batch_rows["answer"]):
                ground_truth = extract_gsm8k_answer(answer_text)
                problem_correct = 0
                for seq in sample_result.sequences:
                    n_total += 1
                    parsed_msg, _ = rdr.parse_response(seq.tokens)
                    content = get_text_content(parsed_msg)
                    if grade_answer(content, ground_truth) == 1.0:
                        n_correct += 1
                        problem_correct += 1
                        full_convo = convo + [{"role": "assistant", "content": content}]
                        datum = conversation_to_datum(
                            full_convo, rdr, max_length=2048,
                            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                        )
                        correct_datums.append(datum)
                if problem_correct > 0:
                    n_solved += 1

            if correct_datums:
                fb = await tc.forward_backward_async(correct_datums, loss_fn="cross_entropy")
                op = await tc.optim_step_async(adam)
                await fb.result_async()
                await op.result_async()

            sample_acc = n_correct / n_total if n_total else 0
            elapsed = _time.time() - t0
            step_time = _time.time() - t_step
            remaining = step_time * (N_STEPS - step - 1)

            entry = {"step": step, "sample_accuracy": sample_acc, "solve_rate": n_solved / BATCH_SIZE}

            if step % EVAL_EVERY == 0:
                test_acc = await _evaluate(tc)
                entry["test_accuracy"] = test_acc
                print(
                    f"RFT step {step:2d} | sample_acc: {sample_acc:.0%} | "
                    f"test: {test_acc:.1%} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )
            else:
                print(
                    f"RFT step {step:2d} | sample_acc: {sample_acc:.0%} | "
                    f"solved: {n_solved}/{BATCH_SIZE} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )

            metrics.append(entry)

        # Final eval
        final_acc = await _evaluate(tc)
        metrics.append({"step": N_STEPS, "test_accuracy": final_acc})
        print(f"\nRFT done! Final test accuracy: {final_acc:.1%} ({_time.time()-t0:.0f}s total)")
        return metrics, tc, rdr

    rft_metrics, tc_rft, renderer_rft = await _run_rft()
    return renderer_rft, rft_metrics, tc_rft


# ==============================================================================
# Part 2: GRPO training loop
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: GRPO (Group Relative Policy Optimization)

    GRPO uses **all** solutions -- correct and incorrect -- weighted by
    group-relative advantages:

    ```
    advantage = reward - mean(rewards_in_group)
    ```

    Correct solutions get positive advantage, wrong ones get negative advantage.
    The model learns to do more of the good **and less of the bad**.
    """)
    return


@app.cell
async def _(
    BASE_MODEL,
    BATCH_SIZE,
    EVAL_EVERY,
    GROUP_SIZE,
    MAX_TOKENS,
    N_EVAL_PROBLEMS,
    N_STEPS,
    TensorData,
    extract_gsm8k_answer,
    fewshot_prefix,
    get_renderer,
    get_text_content,
    grade_answer,
    question_suffix,
    test_data,
    tinker,
    torch,
    train_data,
):
    import asyncio as _asyncio
    import time as _time

    async def _run_grpo():
        # --- Setup ---
        service = tinker.ServiceClient()
        tc = await service.create_lora_training_client_async(base_model=BASE_MODEL, rank=32)
        tok = tc.get_tokenizer()
        rdr = get_renderer("qwen3", tok)
        adam = tinker.AdamParams(learning_rate=2e-5, beta1=0.9, beta2=0.95)
        samp_params = tinker.SamplingParams(
            max_tokens=MAX_TOKENS, temperature=1.0, stop=rdr.get_stop_sequences()
        )

        async def _evaluate(training_client):
            sc = await training_client.save_weights_and_get_sampling_client_async()
            ep = tinker.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=0.0, stop=rdr.get_stop_sequences()
            )
            async def _one(row):
                convo = [*fewshot_prefix, {"role": "user", "content": row["question"] + question_suffix}]
                r = await sc.sample_async(
                    prompt=rdr.build_generation_prompt(convo), num_samples=1, sampling_params=ep
                )
                msg, _ = rdr.parse_response(r.sequences[0].tokens)
                return grade_answer(get_text_content(msg), extract_gsm8k_answer(row["answer"]))
            problems = test_data.select(range(min(N_EVAL_PROBLEMS, len(test_data))))
            scores = await _asyncio.gather(*[_one(row) for row in problems])
            return sum(scores) / len(scores)

        # --- Training loop ---
        metrics = []
        t0 = _time.time()

        for step in range(N_STEPS):
            t_step = _time.time()
            batch_start = step * BATCH_SIZE
            batch_rows = train_data.select(range(batch_start, batch_start + BATCH_SIZE))

            sampling_client = await tc.save_weights_and_get_sampling_client_async()

            prompts = []
            coros = []
            for question in batch_rows["question"]:
                convo = [*fewshot_prefix, {"role": "user", "content": question + question_suffix}]
                prompt = rdr.build_generation_prompt(convo)
                coros.append(
                    sampling_client.sample_async(
                        prompt=prompt, num_samples=GROUP_SIZE, sampling_params=samp_params
                    )
                )
                prompts.append(prompt)

            sample_results = await _asyncio.gather(*coros)

            datums = []
            rewards_all = []
            n_degenerate = 0

            for sample_result, prompt, answer_text in zip(
                sample_results, prompts, batch_rows["answer"]
            ):
                ground_truth = extract_gsm8k_answer(answer_text)
                rewards_G = []
                tokens_G = []
                logprobs_G = []

                for seq in sample_result.sequences:
                    tokens_G.append(seq.tokens)
                    logprobs_G.append(seq.logprobs)
                    parsed_msg, _ = rdr.parse_response(seq.tokens)
                    content = get_text_content(parsed_msg)
                    reward = grade_answer(content, ground_truth)
                    rewards_G.append(reward)

                mean_reward = sum(rewards_G) / len(rewards_G)
                advantages_G = [r - mean_reward for r in rewards_G]
                rewards_all.append(mean_reward)

                if all(a == 0.0 for a in advantages_G):
                    n_degenerate += 1
                    continue

                ob_len = prompt.length - 1
                for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                    model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                    target_tokens = [0] * ob_len + tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                    datum = tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    datums.append(datum)

            if datums:
                fb = await tc.forward_backward_async(datums, loss_fn="importance_sampling")
                op = await tc.optim_step_async(adam)
                await fb.result_async()
                await op.result_async()

            mean_reward = sum(rewards_all) / len(rewards_all) if rewards_all else 0
            elapsed = _time.time() - t0
            step_time = _time.time() - t_step
            remaining = step_time * (N_STEPS - step - 1)

            entry = {"step": step, "reward": mean_reward, "n_degenerate": n_degenerate}

            if step % EVAL_EVERY == 0:
                test_acc = await _evaluate(tc)
                entry["test_accuracy"] = test_acc
                print(
                    f"GRPO step {step:2d} | reward: {mean_reward:.3f} | "
                    f"test: {test_acc:.1%} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )
            else:
                print(
                    f"GRPO step {step:2d} | reward: {mean_reward:.3f} | "
                    f"degen: {n_degenerate}/{BATCH_SIZE} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )

            metrics.append(entry)

        # Final eval
        final_acc = await _evaluate(tc)
        metrics.append({"step": N_STEPS, "test_accuracy": final_acc})
        print(f"\nGRPO done! Final test accuracy: {final_acc:.1%} ({_time.time()-t0:.0f}s total)")
        return metrics, tc, rdr

    grpo_metrics, tc_grpo, renderer_grpo = await _run_grpo()
    return grpo_metrics, renderer_grpo, tc_grpo


# ==============================================================================
# Part 3: Visualize the comparison
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results: Learning curves

    Let's compare how each method's test accuracy evolves over training.
    Both models are evaluated with greedy decoding on 200 GSM8K test problems.
    """)
    return


@app.cell
def _(grpo_metrics, plt, rft_metrics):
    # Extract eval points (steps where test_accuracy was measured)
    _rft_eval = [(m["step"], m["test_accuracy"] * 100) for m in rft_metrics if "test_accuracy" in m]
    _grpo_eval = [(m["step"], m["test_accuracy"] * 100) for m in grpo_metrics if "test_accuracy" in m]

    _rft_x, _rft_y = zip(*_rft_eval) if _rft_eval else ([], [])
    _grpo_x, _grpo_y = zip(*_grpo_eval) if _grpo_eval else ([], [])

    fig_curves, ax_curves = plt.subplots(figsize=(9, 5))
    ax_curves.plot(_rft_x, _rft_y, "o-", color="#2196F3", linewidth=2.5,
                   markersize=7, label="RFT (lr=1e-4)")
    ax_curves.plot(_grpo_x, _grpo_y, "s-", color="#FF5722", linewidth=2.5,
                   markersize=7, label="GRPO (lr=2e-5)")

    ax_curves.set_xlabel("Training step", fontsize=12)
    ax_curves.set_ylabel("GSM8K test accuracy (%)", fontsize=12)
    ax_curves.set_title("RFT vs GRPO: test accuracy over training", fontsize=14, fontweight="bold")
    ax_curves.legend(fontsize=11, loc="lower right")
    ax_curves.grid(True, alpha=0.3)
    ax_curves.set_ylim(0, 100)
    plt.tight_layout()
    fig_curves
    return (fig_curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results: Training signal

    Compare how the training signal looks for each method:
    - **RFT**: fraction of sampled solutions that are correct (sample accuracy)
    - **GRPO**: mean reward per batch (fraction correct at T=1.0)
    """)
    return


@app.cell
def _(grpo_metrics, plt, rft_metrics):
    fig_train, (ax_ta, ax_tb) = plt.subplots(1, 2, figsize=(12, 4.5))

    _rft_sa = [(m["step"], m["sample_accuracy"] * 100) for m in rft_metrics if "sample_accuracy" in m]
    if _rft_sa:
        _xs, _ys = zip(*_rft_sa)
        ax_ta.plot(_xs, _ys, "o-", color="#2196F3", linewidth=1.5, markersize=3)
    ax_ta.set_xlabel("Training step")
    ax_ta.set_ylabel("Sample accuracy (%)")
    ax_ta.set_title("RFT: correct samples / total samples", fontweight="bold")
    ax_ta.grid(True, alpha=0.3)
    ax_ta.set_ylim(0, 105)

    _grpo_r = [(m["step"], m["reward"] * 100) for m in grpo_metrics if "reward" in m]
    if _grpo_r:
        _xs2, _ys2 = zip(*_grpo_r)
        ax_tb.plot(_xs2, _ys2, "s-", color="#FF5722", linewidth=1.5, markersize=3)
    ax_tb.set_xlabel("Training step")
    ax_tb.set_ylabel("Mean reward (%)")
    ax_tb.set_title("GRPO: mean batch reward", fontweight="bold")
    ax_tb.grid(True, alpha=0.3)
    ax_tb.set_ylim(0, 105)

    plt.tight_layout()
    fig_train
    return (fig_train,)


# ==============================================================================
# Part 4: The deeper story -- MATH results
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The deeper story: what happens on harder tasks?

    GSM8K is relatively easy -- Qwen3-8B solves ~60% at baseline. The real
    difference between RFT and GRPO shows up on **harder tasks** where the model
    starts with lower accuracy.

    We ran the same comparison on [Hendrycks MATH](https://arxiv.org/abs/2103.03874)
    (12K training problems, MATH-500 test set, 5 difficulty levels). Here are the
    results from 40 training steps each:
    """)
    return


@app.cell
def _():
    # Pre-computed results from full MATH experiments (Qwen3-8B, K=16, 40 steps)
    math_rft_steps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    math_rft_overall = [0.422, 0.788, 0.786, 0.780, 0.782, 0.796, 0.798, 0.796, 0.788]
    math_rft_by_level = {
        "L1": [0.814, 0.930, 0.977, 0.930, 0.930, 0.953, 0.907, 0.953, 0.930],
        "L2": [0.722, 0.911, 0.867, 0.867, 0.856, 0.867, 0.867, 0.889, 0.922],
        "L3": [0.476, 0.876, 0.857, 0.857, 0.857, 0.895, 0.895, 0.895, 0.886],
        "L4": [0.328, 0.766, 0.797, 0.805, 0.797, 0.812, 0.797, 0.805, 0.758],
        "L5": [0.142, 0.612, 0.604, 0.590, 0.612, 0.604, 0.642, 0.597, 0.604],
    }
    math_grpo_steps = [0, 5, 10, 15, 20, 25, 30, 35]
    math_grpo_overall = [0.359, 0.469, 0.673, 0.775, 0.823, 0.816, 0.841, 0.851]

    return math_grpo_overall, math_grpo_steps, math_rft_by_level, math_rft_overall, math_rft_steps


@app.cell
def _(math_grpo_overall, math_grpo_steps, math_rft_overall, math_rft_steps, plt):
    fig_math, ax_math = plt.subplots(figsize=(9, 5))

    ax_math.plot(math_rft_steps, [x * 100 for x in math_rft_overall], "o-", color="#2196F3",
                 linewidth=2.5, markersize=7, label="RFT (greedy eval)")
    ax_math.plot(math_grpo_steps, [x * 100 for x in math_grpo_overall], "s-", color="#FF5722",
                 linewidth=2.5, markersize=7, label="GRPO (T=1.0 eval)")

    ax_math.axvline(x=15, color="gray", linestyle="--", alpha=0.5)
    ax_math.annotate("crossover", xy=(15, 78), fontsize=9, color="gray", ha="center")
    ax_math.annotate("78.8%", xy=(5, 78.8), xytext=(7, 83), fontsize=9, color="#2196F3",
                     arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2))
    ax_math.annotate("85.1%", xy=(35, 85.1), xytext=(30, 89), fontsize=9, color="#FF5722",
                     arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.2))

    ax_math.set_xlabel("Training step", fontsize=12)
    ax_math.set_ylabel("MATH-500 accuracy (%)", fontsize=12)
    ax_math.set_title("RFT vs GRPO on Hendrycks MATH (Qwen3-8B)", fontsize=14, fontweight="bold")
    ax_math.legend(fontsize=11, loc="lower right")
    ax_math.grid(True, alpha=0.3)
    ax_math.set_ylim(30, 92)
    ax_math.set_xlim(-1, 42)
    plt.tight_layout()
    fig_math
    return (fig_math,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **On MATH, the pattern is dramatic:**

    - **RFT** jumps to 78.8% in 5 steps, then **flatlines for 35 more steps**
    - **GRPO** starts slower but keeps climbing to 85.1% at step 35 -- still improving
    - The **crossover** happens at step 15

    The per-difficulty breakdown reveals *where* RFT fails:
    """)
    return


@app.cell
def _(math_rft_by_level, math_rft_steps, plt):
    _level_colors = {
        "L1": "#4CAF50", "L2": "#8BC34A", "L3": "#FFC107",
        "L4": "#FF9800", "L5": "#F44336",
    }
    fig_levels, ax_levels = plt.subplots(figsize=(9, 5))
    for level, accs in math_rft_by_level.items():
        ax_levels.plot(math_rft_steps, [x * 100 for x in accs], "o-",
                       color=_level_colors[level], linewidth=2, markersize=5,
                       label=f"{level} ({accs[0]*100:.0f}% -> {accs[-1]*100:.0f}%)")

    ax_levels.set_xlabel("Training step", fontsize=12)
    ax_levels.set_ylabel("Accuracy (%)", fontsize=12)
    ax_levels.set_title("RFT on MATH: accuracy by difficulty level", fontsize=14, fontweight="bold")
    ax_levels.legend(fontsize=10, loc="right")
    ax_levels.grid(True, alpha=0.3)
    ax_levels.set_ylim(0, 102)
    plt.tight_layout()
    fig_levels
    return (fig_levels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **L5 (hardest problems) is stuck at ~60%** while L1-L3 saturate at 90%+.

    ---

    ## Why RFT plateaus

    The plateau is NOT because the model can't find correct solutions -- by step 4,
    it solves 85-100% of training problems. The real causes:

    1. **Redundant gradients:** Correct solutions become increasingly similar.
       SFT on near-identical outputs produces diminishing updates.

    2. **No negative signal:** RFT throws away wrong solutions. If the model
       makes systematic errors, RFT has no mechanism to correct them.

    3. **Easy problem bias:** Easy problems produce many more correct solutions,
       dominating the gradient even though the model already masters them.

    > Think of it this way: RFT is like a student who only reviews problems they
    > already solved. They get fast at those, but never learn from their mistakes.

    ## How GRPO breaks through

    | RFT limitation | How GRPO fixes it |
    |---|---|
    | Redundant gradients | **Advantage weighting** gives more credit to rare correct solutions |
    | No negative signal | **Negative advantages** penalize wrong solutions |
    | Easy problem bias | **Group-relative centering** normalizes per-problem |

    For a problem where the model gets 3/16 solutions right:
    - **RFT:** trains on 3 correct solutions with equal weight
    - **GRPO:** trains on all 16. The 3 correct ones get advantage **+0.81**,
      the 13 wrong ones get advantage **-0.19**. The model learns what worked AND what didn't.

    ## Summary

    | | RFT | GRPO |
    |---|---|---|
    | **Speed** | Fast (5 steps to plateau) | Slower (15+ steps) |
    | **Ceiling** | Limited by task difficulty | Keeps improving |
    | **Signal** | Correct solutions only | Correct + incorrect |
    | **Stability** | Very stable (pure SFT) | Needs LR tuning |
    | **Best for** | Easy tasks, quick wins | Hard tasks, pushing limits |

    **The key insight:** RL isn't just "fancier SFT." It provides a qualitatively
    different training signal. When correct solutions become redundant and mistakes
    go uncorrected, RL's ability to learn from the full distribution of outputs is
    what enables continued improvement.

    ## Next steps

    - **Full MATH experiments:** `python -m tinker_cookbook.recipes.math_rft.train env=math`
    - **GRPO recipe:** `python -m tinker_cookbook.recipes.math_rl.train env=math`
    - **RL hyperparameters:** `tutorials/402_rl_hyperparams.py`
    - **Research notes:** `tinker_cookbook/recipes/math_rft/NOTES.md`
    """)
    return


if __name__ == "__main__":
    app.run()
