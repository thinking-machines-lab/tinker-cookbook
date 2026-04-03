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
    # Tutorial 08: Why RL? From Rejection Sampling to GRPO

    You have a model and a reward function that can verify whether an answer is correct.
    How should you train?

    The simplest approach is **Rejection Sampling Fine-Tuning (RFT)**: sample solutions,
    keep the correct ones, fine-tune with standard SFT loss. It's fast and stable.

    But on harder tasks, RFT hits a ceiling. **GRPO** (Group Relative Policy Optimization)
    breaks through that ceiling by learning from *both* correct and incorrect solutions.

    This tutorial runs both methods head-to-head on GSM8K math problems and compares
    the results.

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
# Setup
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
    import asyncio
    import re
    import time
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import matplotlib.pyplot as plt
    import tinker
    import torch
    from tinker import TensorData

    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.renderers.base import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    return (
        TensorData,
        asyncio,
        conversation_to_datum,
        TrainOnWhat,
        get_renderer,
        get_text_content,
        plt,
        re,
        time,
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

    question_suffix = " Provide a step-by-step solution ending with \\boxed{answer}."

    print(f"Loaded {len(train_data)} train / {len(test_data)} test GSM8K problems")
    return (
        extract_boxed,
        extract_gsm8k_answer,
        grade_answer,
        question_suffix,
        test_data,
        train_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training configuration

    Both methods use the same model, data, and group size for a fair comparison.
    """)
    return


@app.cell
def _():
    N_STEPS = 30          # training steps per method
    BATCH_SIZE = 32       # problems per step (groups_per_batch)
    GROUP_SIZE = 16       # completions per problem
    MAX_TOKENS = 1024     # max generation tokens
    BASE_MODEL = "Qwen/Qwen3-8B"
    EVAL_EVERY = 5        # evaluate on test set every N steps
    N_EVAL_PROBLEMS = 200 # test problems for evaluation

    return BASE_MODEL, BATCH_SIZE, EVAL_EVERY, GROUP_SIZE, MAX_TOKENS, N_EVAL_PROBLEMS, N_STEPS


# ==============================================================================
# Part 1: RFT
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: RFT (Rejection Sampling Fine-Tuning)

    The RFT loop:
    1. Sample K solutions per problem from the current model
    2. Grade them -- keep only correct ones
    3. Run standard SFT (cross-entropy) on the correct solutions
    4. Evaluate on the test set periodically

    No advantages, no importance weights, no negative signal.
    Just **SFT on whatever the model gets right**.
    """)
    return


@app.cell
def _(
    TrainOnWhat,
    asyncio,
    conversation_to_datum,
    extract_gsm8k_answer,
    get_renderer,
    get_text_content,
    grade_answer,
    question_suffix,
    test_data,
    time,
    tinker,
    train_data,
):
    async def run_rft(
        base_model: str,
        n_steps: int,
        batch_size: int,
        group_size: int,
        max_tokens: int,
        eval_every: int,
        n_eval_problems: int,
    ) -> list[dict]:
        """Train with RFT: sample solutions, keep correct ones, SFT on them."""

        service = tinker.ServiceClient()
        tc = await service.create_lora_training_client_async(base_model=base_model, rank=32)
        tok = tc.get_tokenizer()
        rdr = get_renderer("qwen3", tok)
        adam = tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95)
        samp_params = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=1.0, stop=rdr.get_stop_sequences()
        )

        async def evaluate():
            sc = await tc.save_weights_and_get_sampling_client_async()
            ep = tinker.SamplingParams(
                max_tokens=max_tokens, temperature=0.0, stop=rdr.get_stop_sequences()
            )

            async def grade_one(row):
                convo = [{"role": "user", "content": row["question"] + question_suffix}]
                r = await sc.sample_async(
                    prompt=rdr.build_generation_prompt(convo), num_samples=1, sampling_params=ep
                )
                msg, _ = rdr.parse_response(r.sequences[0].tokens)
                return grade_answer(get_text_content(msg), extract_gsm8k_answer(row["answer"]))

            problems = test_data.select(range(min(n_eval_problems, len(test_data))))
            scores = await asyncio.gather(*[grade_one(row) for row in problems])
            return sum(scores) / len(scores)

        metrics = []
        t0 = time.time()

        for step in range(n_steps):
            t_step = time.time()
            batch_rows = train_data.select(
                range(step * batch_size, step * batch_size + batch_size)
            )

            # Sample K solutions per problem
            sc = await tc.save_weights_and_get_sampling_client_async()

            async def sample_one(question):
                convo = [{"role": "user", "content": question + question_suffix}]
                prompt = rdr.build_generation_prompt(convo)
                result = await sc.sample_async(
                    prompt=prompt, num_samples=group_size, sampling_params=samp_params
                )
                return result, convo

            results = await asyncio.gather(*[sample_one(q) for q in batch_rows["question"]])

            # Grade and keep only correct solutions
            correct_datums = []
            n_correct = 0
            n_total = 0
            n_solved = 0

            for (sample_result, convo), answer_text in zip(results, batch_rows["answer"]):
                gt = extract_gsm8k_answer(answer_text)
                problem_correct = 0
                for seq in sample_result.sequences:
                    n_total += 1
                    msg, _ = rdr.parse_response(seq.tokens)
                    content = get_text_content(msg)
                    if grade_answer(content, gt) == 1.0:
                        n_correct += 1
                        problem_correct += 1
                        full_convo = convo + [{"role": "assistant", "content": content}]
                        correct_datums.append(conversation_to_datum(
                            full_convo, rdr, max_length=2048,
                            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                        ))
                if problem_correct > 0:
                    n_solved += 1

            # SFT step on correct solutions
            if correct_datums:
                fb = await tc.forward_backward_async(correct_datums, loss_fn="cross_entropy")
                op = await tc.optim_step_async(adam)
                await fb.result_async()
                await op.result_async()

            sample_acc = n_correct / n_total if n_total else 0
            elapsed = time.time() - t0
            step_time = time.time() - t_step
            remaining = step_time * (n_steps - step - 1)

            entry = {"step": step, "sample_accuracy": sample_acc, "solve_rate": n_solved / batch_size}

            if step % eval_every == 0:
                test_acc = await evaluate()
                entry["test_accuracy"] = test_acc
                print(
                    f"RFT step {step:2d} | sample_acc: {sample_acc:.0%} | "
                    f"test: {test_acc:.1%} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )
            else:
                print(
                    f"RFT step {step:2d} | sample_acc: {sample_acc:.0%} | "
                    f"solved: {n_solved}/{batch_size} | "
                    f"{elapsed:.0f}s elapsed, ~{remaining/60:.0f}min remaining"
                )

            metrics.append(entry)

        # Final eval
        final_acc = await evaluate()
        metrics.append({"step": n_steps, "test_accuracy": final_acc})
        print(f"\nRFT done! Final test accuracy: {final_acc:.1%} ({time.time()-t0:.0f}s total)")
        return metrics

    return (run_rft,)


@app.cell
async def _(
    BASE_MODEL, BATCH_SIZE, EVAL_EVERY, GROUP_SIZE, MAX_TOKENS, N_EVAL_PROBLEMS, N_STEPS,
    run_rft,
):
    rft_metrics = await run_rft(
        base_model=BASE_MODEL,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        max_tokens=MAX_TOKENS,
        eval_every=EVAL_EVERY,
        n_eval_problems=N_EVAL_PROBLEMS,
    )
    return (rft_metrics,)


# ==============================================================================
# Part 2: GRPO
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

    For the GRPO implementation details, see [Tutorial 04](104_first_rl.py).
    Here we use the cookbook's production recipe directly:
    """)
    return


@app.cell
async def _(BASE_MODEL, BATCH_SIZE, EVAL_EVERY, GROUP_SIZE, MAX_TOKENS, N_STEPS):
    import json as _json

    from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main

    _grpo_log_path = "/tmp/tinker-examples/tutorial_408_grpo"

    await cli_main(CLIConfig(
        model_name=BASE_MODEL,
        env="gsm8k",
        group_size=GROUP_SIZE,
        groups_per_batch=BATCH_SIZE,
        learning_rate=2e-5,
        max_tokens=MAX_TOKENS,
        lora_rank=32,
        eval_every=EVAL_EVERY,
        max_steps=N_STEPS,
        log_path=_grpo_log_path,
        behavior_if_log_dir_exists="delete",
    ))

    # Read metrics from the log
    grpo_metrics = []
    with open(f"{_grpo_log_path}/metrics.jsonl") as _f:
        for _line in _f:
            _m = _json.loads(_line)
            _entry = {"step": _m.get("step", 0)}
            if "env/all/reward/total" in _m:
                _entry["reward"] = _m["env/all/reward/total"]
            if "test/env/all/reward/total" in _m:
                _entry["test_accuracy"] = _m["test/env/all/reward/total"]
            grpo_metrics.append(_entry)

    print(f"GRPO done! Loaded {len(grpo_metrics)} metric entries from {_grpo_log_path}")
    return (grpo_metrics,)


# ==============================================================================
# Results
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results: Learning curves

    Both models are evaluated with greedy decoding on 200 GSM8K test problems
    every 5 training steps.
    """)
    return


@app.cell
def _(grpo_metrics, plt, rft_metrics):
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
    fig_train, (_ax_a, _ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    _rft_sa = [(m["step"], m["sample_accuracy"] * 100) for m in rft_metrics if "sample_accuracy" in m]
    if _rft_sa:
        _xs, _ys = zip(*_rft_sa)
        _ax_a.plot(_xs, _ys, "o-", color="#2196F3", linewidth=1.5, markersize=3)
    _ax_a.set_xlabel("Training step")
    _ax_a.set_ylabel("Sample accuracy (%)")
    _ax_a.set_title("RFT: correct samples / total samples", fontweight="bold")
    _ax_a.grid(True, alpha=0.3)
    _ax_a.set_ylim(0, 105)

    _grpo_r = [(m["step"], m["reward"] * 100) for m in grpo_metrics if "reward" in m]
    if _grpo_r:
        _xs2, _ys2 = zip(*_grpo_r)
        _ax_b.plot(_xs2, _ys2, "s-", color="#FF5722", linewidth=1.5, markersize=3)
    _ax_b.set_xlabel("Training step")
    _ax_b.set_ylabel("Mean reward (%)")
    _ax_b.set_title("GRPO: mean batch reward", fontweight="bold")
    _ax_b.grid(True, alpha=0.3)
    _ax_b.set_ylim(0, 105)

    plt.tight_layout()
    fig_train
    return (fig_train,)


# ==============================================================================
# Analysis
# ==============================================================================


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why RFT plateaus

    RFT's training signal degrades in three ways:

    1. **Redundant gradients:** As the model improves, its correct solutions become
       increasingly similar. SFT on near-identical outputs produces diminishing updates.

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

    - **Harder tasks:** Run on Hendrycks MATH with `python -m tinker_cookbook.recipes.math_rft.train env=math` to see the RFT plateau more dramatically
    - **GRPO recipe:** `python -m tinker_cookbook.recipes.math_rl.train env=math`
    - **RL hyperparameters:** `tutorials/402_rl_hyperparams.py`
    - **Research notes:** `tinker_cookbook/recipes/math_rft/NOTES.md`
    """)
    return


if __name__ == "__main__":
    app.run()
