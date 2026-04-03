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

    This tutorial walks through:

    1. How RFT and GRPO work — with code
    2. A head-to-head comparison on math reasoning (MATH-500)
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
    the full distribution of outputs — correct and incorrect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RFT in code

    The RFT training loop is simple. Here is the core logic (simplified from
    `tinker_cookbook/recipes/math_rft/train.py`):

    ```python
    for batch in problems:
        # 1. Sample K solutions per problem
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        results = await asyncio.gather(*[
            sampling_client.sample_async(prompt, num_samples=K, sampling_params=params)
            for prompt in batch
        ])

        # 2. Grade and keep only correct solutions
        correct_datums = []
        for result, answer in zip(results, answers):
            for seq in result.sequences:
                response = renderer.parse_response(seq.tokens)
                if is_correct(response, answer):
                    datum = conversation_to_datum(prompt + response, renderer)
                    correct_datums.append(datum)

        # 3. Standard SFT step on correct solutions
        await training_client.forward_backward_async(correct_datums, loss_fn="cross_entropy")
        await training_client.optim_step_async(adam_params)
    ```

    That's it. No advantages, no importance weights, no negative signal. Just SFT on
    whatever the model gets right.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GRPO in code

    GRPO uses all solutions — correct and incorrect — weighted by group-relative advantages.
    Here is the core logic (simplified from `tutorials/104_first_rl.py`):

    ```python
    for batch in problems:
        # 1. Sample K solutions per problem (same as RFT)
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        results = await asyncio.gather(...)

        # 2. Grade ALL solutions and compute group-relative advantages
        datums = []
        for result, answer in zip(results, answers):
            rewards = [grade(seq, answer) for seq in result.sequences]
            mean_reward = sum(rewards) / len(rewards)
            advantages = [r - mean_reward for r in rewards]  # <-- the key difference

            if all(a == 0 for a in advantages):
                continue  # skip degenerate groups

            for seq, advantage, logprobs in zip(result.sequences, advantages, ...):
                datum = build_rl_datum(prompt, seq, logprobs, advantage)
                datums.append(datum)

        # 3. RL step with importance-weighted loss
        await training_client.forward_backward_async(datums, loss_fn="importance_sampling")
        await training_client.optim_step_async(adam_params)
    ```

    The key line is `advantages = [r - mean_reward for r in rewards]`. Correct solutions
    get positive advantages (reward 1.0 - mean), wrong solutions get negative advantages
    (reward 0.0 - mean). The model learns to do more of the good and less of the bad.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The experiment

    We trained Qwen3-8B on the [Hendrycks MATH](https://arxiv.org/abs/2103.03874) dataset
    using both methods. MATH has 12,000 training problems and a held-out test set
    ([MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)) with
    problems at 5 difficulty levels.

    **Setup:**
    - Model: `Qwen/Qwen3-8B` (base model, no instruction tuning)
    - Group size K = 16 (solutions sampled per problem)
    - 32 problems per batch, 40 training steps
    - RFT: learning rate 1e-4 (cross-entropy allows higher LR)
    - GRPO: learning rate 8e-5 (importance-weighted loss needs lower LR)
    - LoRA rank 32, max generation length 2048 tokens

    Both methods see the same number of problems and sample the same number of solutions.
    The only difference is the training algorithm.
    """)
    return


# ---- Pre-computed experimental data ----

@app.cell
def _():
    # RFT results on MATH-500 (greedy eval, Qwen3-8B, K=16, lr=1e-4)
    rft_steps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    rft_overall = [0.422, 0.788, 0.786, 0.780, 0.782, 0.796, 0.798, 0.796, 0.788]
    rft_by_level = {
        "L1": [0.814, 0.930, 0.977, 0.930, 0.930, 0.953, 0.907, 0.953, 0.930],
        "L2": [0.722, 0.911, 0.867, 0.867, 0.856, 0.867, 0.867, 0.889, 0.922],
        "L3": [0.476, 0.876, 0.857, 0.857, 0.857, 0.895, 0.895, 0.895, 0.886],
        "L4": [0.328, 0.766, 0.797, 0.805, 0.797, 0.812, 0.797, 0.805, 0.758],
        "L5": [0.142, 0.612, 0.604, 0.590, 0.612, 0.604, 0.642, 0.597, 0.604],
    }

    # GRPO results on MATH-500 (T=1.0 eval, Qwen3-8B, K=16, lr=8e-5)
    grpo_steps = [0, 5, 10, 15, 20, 25, 30, 35]
    grpo_overall = [0.359, 0.469, 0.673, 0.775, 0.823, 0.816, 0.841, 0.851]

    # GRPO training reward trajectory
    grpo_train_steps = list(range(40))
    grpo_train_reward = [
        0.415, 0.274, 0.425, 0.383, 0.433, 0.428, 0.620, 0.567,
        0.528, 0.719, 0.683, 0.753, 0.697, 0.771, 0.903, 0.845,
        0.806, 0.636, 0.830, 0.855, 0.741, 0.905, 0.895, 0.793,
        0.866, 0.825, 0.861, 0.812, 0.845, 0.853, 0.968, 0.750,
        0.876, 0.783, 0.784, 0.752, 0.809, 0.740, 0.914, 0.876,
    ]

    # RFT training metrics
    rft_train_steps = list(range(40))
    rft_sample_acc = [
        0.498, 0.340, 0.455, 0.494, 0.770, 0.801, 0.873, 0.768,
        0.789, 0.787, 0.770, 0.779, 0.787, 0.789, 0.863, 0.852,
        0.807, 0.648, 0.770, 0.795, 0.740, 0.859, 0.891, 0.736,
        0.840, 0.801, 0.840, 0.801, 0.770, 0.758, 0.928, 0.674,
        0.807, 0.758, 0.709, 0.723, 0.725, 0.680, 0.869, 0.795,
    ]
    rft_solve_rate = [
        0.656, 0.562, 0.688, 0.750, 0.938, 0.938, 1.000, 0.844,
        0.906, 0.875, 0.844, 0.875, 0.938, 0.906, 1.000, 1.000,
        1.000, 0.844, 0.938, 0.938, 0.938, 0.906, 1.000, 0.906,
        0.969, 0.938, 0.938, 0.875, 0.969, 0.969, 1.000, 0.906,
        0.969, 0.906, 0.906, 0.844, 0.875, 0.812, 0.938, 0.969,
    ]

    return (
        grpo_overall,
        grpo_steps,
        grpo_train_reward,
        grpo_train_steps,
        rft_by_level,
        rft_overall,
        rft_sample_acc,
        rft_solve_rate,
        rft_steps,
        rft_train_steps,
    )


# ---- Visualization: Learning curves ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result 1: Learning curves

    Let's compare how accuracy evolves during training.

    **Note on eval methods:** RFT is evaluated with greedy decoding (temperature=0),
    while GRPO is evaluated with temperature=1.0 sampling (standard for RL).
    Greedy decoding scores ~6-8pp higher, so keep this in mind when comparing the
    absolute numbers. The *trends* are the important part.
    """)
    return


@app.cell
def _(grpo_overall, grpo_steps, rft_overall, rft_steps):
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(rft_steps, [x * 100 for x in rft_overall], "o-", color="#2196F3",
             linewidth=2.5, markersize=7, label="RFT (greedy eval)", zorder=3)
    ax1.plot(grpo_steps, [x * 100 for x in grpo_overall], "s-", color="#FF5722",
             linewidth=2.5, markersize=7, label="GRPO (T=1.0 eval)", zorder=3)

    # Annotate the crossover
    ax1.axvline(x=15, color="gray", linestyle="--", alpha=0.5)
    ax1.annotate("crossover", xy=(15, 78), fontsize=9, color="gray", ha="center")

    # Annotate key points
    ax1.annotate("78.8%", xy=(5, 78.8), xytext=(7, 83),
                 fontsize=9, color="#2196F3",
                 arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2))
    ax1.annotate("85.1%", xy=(35, 85.1), xytext=(30, 89),
                 fontsize=9, color="#FF5722",
                 arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.2))

    ax1.set_xlabel("Training step", fontsize=12)
    ax1.set_ylabel("MATH-500 accuracy (%)", fontsize=12)
    ax1.set_title("RFT vs GRPO on MATH-500 (Qwen3-8B)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 92)
    ax1.set_xlim(-1, 42)
    plt.tight_layout()
    fig1
    return (fig1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this shows:**

    - **RFT converges in ~5 steps** to 78.8%, then flatlines for 35 more steps.
      No amount of additional training helps.
    - **GRPO starts slower** but keeps improving. By step 15 it catches RFT,
      and by step 35 it reaches 85.1% — and it's still climbing.
    - The **crossover at step 15** is the key moment: before it, RFT is more efficient.
      After it, GRPO is the only method still making progress.

    RFT gives you a fast ceiling. GRPO gives you a higher ceiling.
    """)
    return


# ---- Visualization: Per-difficulty breakdown ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result 2: Where does RFT fail?

    MATH problems have 5 difficulty levels. Let's see how RFT performs on each.
    """)
    return


@app.cell
def _(rft_by_level, rft_steps):
    import matplotlib.pyplot as plt

    colors = {"L1": "#4CAF50", "L2": "#8BC34A", "L3": "#FFC107", "L4": "#FF9800", "L5": "#F44336"}

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for level, accs in rft_by_level.items():
        ax2.plot(rft_steps, [x * 100 for x in accs], "o-", color=colors[level],
                 linewidth=2, markersize=5, label=f"{level} ({accs[0]*100:.0f}% -> {accs[-1]*100:.0f}%)")

    ax2.set_xlabel("Training step", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("RFT accuracy by difficulty level", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 102)
    plt.tight_layout()
    fig2
    return (fig2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The difficulty-dependent ceiling:**

    | Level | Baseline | After RFT | Improvement |
    |-------|----------|-----------|-------------|
    | L1 (easiest) | 81% | 93% | +12pp |
    | L2 | 72% | 92% | +20pp |
    | L3 | 48% | 89% | +41pp |
    | L4 | 33% | 76% | +43pp |
    | L5 (hardest) | **14%** | **60%** | +46pp, then **stuck** |

    RFT improves every level dramatically in the first 5 steps. But then:
    - L1-L3 saturate at 90%+ (good enough)
    - **L5 is stuck at ~60%** and never improves further, no matter how long you train

    This 60% ceiling on L5 is the core limitation of RFT.
    """)
    return


# ---- Visualization: Training signal analysis ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Result 3: The training signal paradox

    Here's the puzzle: if RFT plateaus because it can't find correct solutions,
    we'd expect the *training solve rate* (fraction of problems where at least one
    of K=16 samples is correct) to be low. But look at what actually happens:
    """)
    return


@app.cell
def _(rft_sample_acc, rft_solve_rate, rft_train_steps):
    import matplotlib.pyplot as plt

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax3a.plot(rft_train_steps, [x * 100 for x in rft_solve_rate], "o-",
              color="#9C27B0", linewidth=1.5, markersize=3, alpha=0.8)
    ax3a.axhline(y=85, color="gray", linestyle="--", alpha=0.5)
    ax3a.set_xlabel("Training step")
    ax3a.set_ylabel("Solve rate (%)")
    ax3a.set_title("Training solve rate\n(% of problems with >= 1 correct solution)")
    ax3a.set_ylim(50, 105)
    ax3a.grid(True, alpha=0.3)

    ax3b.plot(rft_train_steps, [x * 100 for x in rft_sample_acc], "o-",
              color="#00BCD4", linewidth=1.5, markersize=3, alpha=0.8)
    ax3b.set_xlabel("Training step")
    ax3b.set_ylabel("Sample accuracy (%)")
    ax3b.set_title("Sample accuracy\n(% of all K=16 samples that are correct)")
    ax3b.set_ylim(30, 100)
    ax3b.grid(True, alpha=0.3)

    plt.suptitle("RFT training signal over time", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3
    return (fig3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The paradox:** By step 4, the model solves 85-100% of training problems
    (even hard ones!). Sample accuracy stabilizes around 75-85%. RFT has *plenty*
    of correct solutions to train on.

    **So why does test accuracy plateau?**

    The answer reveals the fundamental limitation of RFT:
    """)
    return


# ---- Why RFT plateaus ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why RFT plateaus

    RFT's training signal degrades in three ways:

    ### 1. Redundant gradients
    As the model improves, its correct solutions become increasingly *similar*.
    Training on near-identical outputs produces diminishing gradient updates —
    the model is just reinforcing what it already knows.

    ### 2. No negative signal
    RFT throws away wrong solutions. It can never say "stop doing this."
    If the model makes a systematic error on certain problem types (e.g., always
    mishandling combinatorics), RFT has no mechanism to correct it — it can only
    wait for a lucky correct sample, and reinforce that.

    ### 3. Easy problem bias
    In each batch, easy problems (L1-L3) generate many more correct solutions than
    hard problems (L5). The gradient is dominated by easy-problem signal, even though
    the model already masters them.

    > **Think of it this way:** RFT is like a student who only reviews problems they
    > already solved. They get very fast at those problems, but never learn from
    > their mistakes on harder ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How GRPO breaks through

    GRPO addresses each of RFT's limitations:

    | RFT limitation | How GRPO fixes it |
    |---|---|
    | Redundant gradients on easy problems | **Advantage weighting**: rare correct solutions on hard problems get higher advantage |
    | No negative signal | **Negative advantages**: wrong solutions are actively penalized, pushing the model away from failure modes |
    | Easy problem bias | **Group-relative centering**: advantages are computed *within* each problem, so hard and easy problems contribute equally |

    The key insight: **GRPO's advantage function is richer than binary correct/incorrect.**

    For a problem where the model gets 3/16 solutions right:
    - RFT: trains on 3 correct solutions with uniform weight
    - GRPO: trains on all 16 solutions. The 3 correct ones get advantage +0.81,
      the 13 wrong ones get advantage -0.19. The model learns what worked AND what didn't.
    """)
    return


# ---- Visualization: GRPO training reward ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GRPO's training reward trajectory

    Unlike RFT (which shows no progress after step 5), GRPO's training reward
    keeps climbing throughout training:
    """)
    return


@app.cell
def _(grpo_overall, grpo_steps, grpo_train_reward, grpo_train_steps):
    import matplotlib.pyplot as plt

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training reward
    ax4a.plot(grpo_train_steps, [x * 100 for x in grpo_train_reward], "o-",
              color="#FF5722", linewidth=1, markersize=3, alpha=0.7)
    # Add trend line
    import numpy as np
    z = np.polyfit(grpo_train_steps, grpo_train_reward, 2)
    p = np.poly1d(z)
    smooth_x = np.linspace(0, 39, 100)
    ax4a.plot(smooth_x, [x * 100 for x in p(smooth_x)], "--",
              color="#FF5722", linewidth=2, alpha=0.5, label="Trend")
    ax4a.set_xlabel("Training step")
    ax4a.set_ylabel("Training reward (%)")
    ax4a.set_title("Training batch reward")
    ax4a.grid(True, alpha=0.3)
    ax4a.set_ylim(20, 105)
    ax4a.legend()

    # Test reward
    ax4b.plot(grpo_steps, [x * 100 for x in grpo_overall], "s-",
              color="#FF5722", linewidth=2.5, markersize=8)
    ax4b.set_xlabel("Training step")
    ax4b.set_ylabel("MATH-500 accuracy (%)")
    ax4b.set_title("Test accuracy (T=1.0)")
    ax4b.grid(True, alpha=0.3)
    ax4b.set_ylim(30, 92)

    # Annotate slope
    ax4b.annotate("Still climbing\nat step 35",
                  xy=(35, 85.1), xytext=(25, 72),
                  fontsize=10, color="#FF5722",
                  arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1.5))

    plt.suptitle("GRPO training progression", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4
    return (fig4,)


# ---- The surprise: warm-start hurts ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Surprise: warm-starting GRPO from RFT hurts

    A natural idea: use RFT for 5 steps to quickly reach ~79%, then switch to GRPO
    to break through the ceiling. Best of both worlds, right?

    **Wrong.** We ran this experiment and the result was surprising:
    """)
    return


@app.cell
def _():
    # Hybrid experiment data (5 steps RFT, then 35 steps GRPO)
    hybrid_grpo_steps = [0, 5, 10, 15, 20, 25, 30]
    hybrid_test = [0.771, 0.758, 0.783, 0.785, 0.786, 0.787, 0.793]

    # Pure GRPO for comparison
    pure_grpo_steps = [0, 5, 10, 15, 20, 25, 30, 35]
    pure_grpo_test = [0.359, 0.469, 0.673, 0.775, 0.823, 0.816, 0.841, 0.851]

    return hybrid_grpo_steps, hybrid_test, pure_grpo_steps, pure_grpo_test


@app.cell
def _(hybrid_grpo_steps, hybrid_test, pure_grpo_steps, pure_grpo_test):
    import matplotlib.pyplot as plt

    fig5, ax5 = plt.subplots(figsize=(9, 5))

    ax5.plot(pure_grpo_steps, [x * 100 for x in pure_grpo_test], "s-",
             color="#FF5722", linewidth=2.5, markersize=7, label="Pure GRPO (from scratch)")
    ax5.plot(hybrid_grpo_steps, [x * 100 for x in hybrid_test], "D-",
             color="#9C27B0", linewidth=2.5, markersize=7, label="RFT(5 steps) -> GRPO")

    # RFT plateau line for reference
    ax5.axhline(y=78.8, color="#2196F3", linestyle=":", alpha=0.5, label="RFT plateau (~79%)")

    ax5.set_xlabel("GRPO training step (after warm-start)", fontsize=12)
    ax5.set_ylabel("MATH-500 accuracy (%)", fontsize=12)
    ax5.set_title("Warm-starting GRPO from RFT checkpoint", fontsize=14, fontweight="bold")
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(30, 92)
    plt.tight_layout()
    fig5
    return (fig5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The hybrid (purple) never catches pure GRPO (red).**

    Despite starting at 77.1% (vs GRPO's 35.9%), the warm-started GRPO peaks at only
    79.3% — barely above RFT's plateau and far below pure GRPO's 85.1%.

    **Why?** RFT collapses the model's output distribution. After 5 steps of SFT
    on correct solutions, the model becomes very *confident* — it generates similar
    outputs with high probability. This is fine for SFT, but it kills GRPO's ability
    to explore. GRPO needs diverse outputs to compute meaningful advantages, and the
    RFT-trained model doesn't produce them.

    > **Lesson:** RFT and GRPO aren't complementary stages — they're fundamentally
    > different optimization strategies. Switching from one to the other mid-training
    > requires careful handling (e.g., KL regularization, entropy bonuses) to preserve
    > the model's exploration ability.
    """)
    return


# ---- Practical recommendations ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to use which method

    The choice depends on your task's difficulty relative to the model's capability:

    | Regime | Baseline accuracy | Recommendation | Why |
    |--------|------------------|----------------|-----|
    | Easy | > 60% | **RFT** | Fast convergence, ~5 steps to plateau. GRPO adds complexity without benefit. |
    | Hard | 20-60% | **GRPO** | RFT will plateau quickly. GRPO breaks through by learning from failures. |
    | Very hard | < 20% | **GRPO** (with larger K) | Both methods struggle, but GRPO's negative signal still provides gradient. |

    **The practical test:** run RFT for 5-10 steps. If accuracy is still improving,
    keep going. If it flatlines, switch to GRPO from scratch (not from the RFT checkpoint).
    """)
    return


# ---- Summary ----

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | | RFT | GRPO |
    |---|---|---|
    | **Speed** | Fast (5 steps to plateau) | Slow (15+ steps) |
    | **Ceiling** | Limited by task difficulty | Continues improving |
    | **Signal** | Correct solutions only | Correct + incorrect |
    | **Stability** | Very stable (pure SFT) | Needs LR tuning |
    | **Best for** | Easy tasks, quick wins | Hard tasks, pushing limits |

    **The key insight from this tutorial:** RL isn't just "fancier SFT." It provides
    a qualitatively different training signal. When SFT runs out of steam (because
    correct solutions become redundant and mistakes go uncorrected), RL's ability to
    learn from the full distribution of outputs — both good and bad — is what enables
    continued improvement.

    ## Try it yourself

    - **RFT recipe**: `python -m tinker_cookbook.recipes.math_rft.train env=math model_name=Qwen/Qwen3-8B`
    - **GRPO recipe**: `python -m tinker_cookbook.recipes.math_rl.train env=math model_name=Qwen/Qwen3-8B`
    - **Tutorial 04**: `tutorials/104_first_rl.py` for a hands-on introduction to GRPO
    """)
    return


if __name__ == "__main__":
    app.run()
