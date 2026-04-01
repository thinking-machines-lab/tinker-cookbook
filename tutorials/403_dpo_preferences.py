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
    # Tutorial 11: DPO and Preference Learning

    Build preference data, train with DPO, and evaluate with a PreferenceModel.

    **Direct Preference Optimization (DPO)** trains a model to prefer "chosen" over "rejected" responses without an explicit reward model. The key idea: the optimal policy under a KL-constrained reward maximization objective has a closed-form relationship to a preference model.
    """)
    return


@app.cell
def _():
    from tinker_cookbook.preference.types import (
        Comparison,
        ComparisonRenderer,
        ComparisonRendererFromChatRenderer,
        LabeledComparison,
        PreferenceModel,
        PreferenceModelFromChatRenderer,
    )

    return (
        Comparison,
        ComparisonRenderer,
        ComparisonRendererFromChatRenderer,
        LabeledComparison,
        PreferenceModel,
        PreferenceModelFromChatRenderer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- Create preference data

    A `Comparison` pairs two completions for the same prompt. A `LabeledComparison` adds a human preference label (A, B, or Tie).
    """)
    return


@app.cell
def _(Comparison, LabeledComparison):
    # Create a labeled comparison: the human prefers completion A
    comparison = Comparison(
        prompt_conversation=[
            {"role": "user", "content": "Explain gravity in one sentence."},
        ],
        completion_A=[
            {
                "role": "assistant",
                "content": "Gravity is the force that attracts objects with mass toward each other.",
            },
        ],
        completion_B=[
            {"role": "assistant", "content": "Gravity is like magnets but for everything."},
        ],
    )

    labeled = LabeledComparison(comparison=comparison, label="A")
    print(f"Prompt:       {comparison.prompt_conversation[0]['content']}")
    print(f"Completion A: {comparison.completion_A[0]['content']}")
    print(f"Completion B: {comparison.completion_B[0]['content']}")
    print(f"Preferred:    {labeled.label}")

    # Swapping reverses the label
    swapped = labeled.swap()
    print("\nAfter swap:")
    print(f"Completion A: {swapped.comparison.completion_A[0]['content']}")
    print(f"Preferred:    {swapped.label}")
    return (comparison, labeled, swapped)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- ComparisonRenderer

    The `ComparisonRendererFromChatRenderer` converts a `Comparison` into tokenized model input by formatting both completions with section markers:

    ```
    [prompt] ==== Completion A ==== [text A] ==== Completion B ==== [text B] ==== Preference ====
    ```

    For DPO training, each labeled comparison produces two datums (chosen + rejected) with per-token loss weights on the completion tokens.
    """)
    return


@app.cell
def _(ComparisonRendererFromChatRenderer, comparison):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer("qwen3", tokenizer)

    comparison_renderer = ComparisonRendererFromChatRenderer(renderer)

    # Build a generation prompt for preference prediction
    model_input = comparison_renderer.build_generation_prompt(comparison)
    print(f"Prompt tokens: {model_input.length}")
    print(f"Decoded (last 100 chars): ...{tokenizer.decode(list(model_input.to_ints())[-50:])}")
    return (MODEL_NAME, comparison_renderer, model_input, renderer, tokenizer)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Configure DPO training

    `preference.train_dpo.Config` is similar to the SFT config but adds:
    - **`dpo_beta`** -- KL penalty coefficient (higher = more conservative updates)
    - **`reference_model_name`** -- optional explicit reference model (default: initial weights)

    The dataset builder must produce interleaved chosen/rejected datum pairs.
    """)
    return


@app.cell
def _(MODEL_NAME):
    from tinker_cookbook.preference.train_dpo import Config as DPOConfig
    from tinker_cookbook.preference.train_dpo import compute_dpo_loss

    # Example config (not running training here)
    print("DPO Config fields:")
    print(f"  model_name:      {MODEL_NAME}")
    print("  dpo_beta:        0.1  (default)")
    print("  learning_rate:   1e-5 (default, lower than SFT)")
    print("  lr_schedule:     linear")
    print("  lora_rank:       32")
    return DPOConfig, compute_dpo_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 -- Understanding the DPO loss

    The DPO loss is:
    ```
    L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    ```
    where `log_ratio = log p_policy(y|x) - log p_ref(y|x)`.

    Intuitively:
    - The model should assign **higher probability** to chosen over rejected
    - The `beta` parameter controls how much the model can deviate from the reference
    - Higher `beta` = more conservative (stays closer to reference)
    """)
    return


@app.cell
def _(compute_dpo_loss):
    import torch

    # Simulate DPO loss computation
    # Positive log-ratio means policy prefers this over reference
    chosen_logprobs = [torch.tensor(-2.0), torch.tensor(-1.5)]
    rejected_logprobs = [torch.tensor(-3.0), torch.tensor(-4.0)]
    chosen_ref_logprobs = [torch.tensor(-2.5), torch.tensor(-2.0)]
    rejected_ref_logprobs = [torch.tensor(-2.5), torch.tensor(-3.0)]

    for beta in [0.05, 0.1, 0.5]:
        loss, metrics = compute_dpo_loss(
            chosen_logprobs,
            rejected_logprobs,
            chosen_ref_logprobs,
            rejected_ref_logprobs,
            dpo_beta=beta,
        )
        print(
            f"beta={beta:.2f}: loss={metrics['dpo_loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.2f}, margin={metrics['margin']:.4f}"
        )
    return (chosen_logprobs, chosen_ref_logprobs, rejected_logprobs, rejected_ref_logprobs)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    DPO workflow:
    1. Collect **preference data** as `LabeledComparison` objects (from humans or an AI judge)
    2. Render them into **chosen/rejected datum pairs** using `DPODatasetBuilderFromComparisons`
    3. Configure training with `train_dpo.Config` (set `dpo_beta`, `learning_rate`)
    4. Run `train_dpo.main(config)` -- handles reference logprob computation, custom loss, and checkpointing
    5. Evaluate with a `PreferenceModel` to measure win rate against a baseline

    Key hyperparameters:
    - **`dpo_beta`**: 0.05-0.5 (start with 0.1)
    - **`learning_rate`**: 1e-6 to 5e-5 (lower than SFT)
    - **`num_epochs`**: 1-3 (DPO is prone to overfitting)
    """)
    return


if __name__ == "__main__":
    app.run()
