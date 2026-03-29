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
    # Tutorial 09: SL Hyperparameters

    Use `get_lora_lr_multiplier` to pick learning rate; compare rank and batch size effects.

    Choosing the right learning rate for LoRA fine-tuning depends on the model architecture. The cookbook provides utilities that encode empirical scaling laws so you can transfer good hyperparameters across models.
    """)
    return


@app.cell
def _():
    from tinker_cookbook.hyperparam_utils import (
        get_lora_lr_multiplier,
        get_lora_lr_over_full_finetune_lr,
        get_lora_param_count,
        get_lr,
    )

    return (
        get_lora_lr_multiplier,
        get_lora_lr_over_full_finetune_lr,
        get_lora_param_count,
        get_lr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The LoRA LR multiplier

    `get_lora_lr_multiplier(model_name)` returns a model-specific scalar. Given a known-good LR for model A, you can estimate the LR for model B:

    ```
    LR_B = LR_A * get_lora_lr_multiplier(B) / get_lora_lr_multiplier(A)
    ```

    Under the hood, it combines two factors:
    1. **Full fine-tune scaling**: `1 / sqrt(param_count)` -- larger models need smaller LRs
    2. **LoRA factor**: a fixed 10x multiplier (LoRA adapters converge faster than full fine-tuning)
    """)
    return


@app.cell
def _(get_lora_lr_over_full_finetune_lr, get_lr):
    # The LoRA multiplier over full fine-tuning is a constant 10x
    lora_factor = get_lora_lr_over_full_finetune_lr("Qwen/Qwen3-8B")
    print(f"LoRA over full-FT multiplier: {lora_factor}x")

    # get_lr() gives a recommended starting LR for calibrated model families
    for model in ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"]:
        lr = get_lr(model, is_lora=True)
        print(f"  {model}: LR = {lr:.6f}")
    return (lora_factor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing LoRA ranks

    Higher LoRA rank = more trainable parameters = more expressive adapter, but also higher memory and compute cost. Use `get_lora_param_count` to see the trade-off.
    """)
    return


@app.cell
def _(get_lora_param_count):
    model = "Qwen/Qwen3-4B-Instruct-2507"
    ranks = [8, 32, 128]

    print(f"Model: {model}")
    print(f"{'Rank':<8} {'Params':<15} {'Params (M)':<12}")
    print("-" * 35)
    for rank in ranks:
        count = get_lora_param_count(model, lora_rank=rank)
        print(f"{rank:<8} {count:<15,} {count / 1e6:<12.1f}")
    return (model, ranks)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LR scheduling

    The `train.Config` supports three LR schedules via the `lr_schedule` field:
    - **`"linear"`** -- linear decay from peak to 0 over all steps
    - **`"cosine"`** -- cosine annealing to 0
    - **`"constant"`** -- no decay

    The schedule is applied as a multiplier: `effective_lr = learning_rate * schedule_multiplier(step, total_steps)`.
    """)
    return


@app.cell
def _():
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    total_steps = 100
    schedules = ["linear", "cosine", "constant"]

    for schedule in schedules:
        mults = [
            compute_schedule_lr_multiplier(schedule, step, total_steps)
            for step in range(total_steps)
        ]
        print(f"{schedule:>8}: start={mults[0]:.2f}, mid={mults[50]:.2f}, end={mults[-1]:.2f}")
    return (schedules, total_steps)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweep pattern

    A typical hyperparameter sweep for SFT:

    | Parameter | Values to try | Notes |
    |-----------|---------------|-------|
    | `learning_rate` | `get_lr(model)`, 0.5x, 2x | Start from the recommended LR |
    | `lora_rank` | 8, 32, 128 | Higher rank = more capacity |
    | `lr_schedule` | linear, cosine | Linear is the default |
    | `batch_size` | 4, 8, 16 | Larger = smoother gradients |
    | `num_epochs` | 1, 2, 3 | Watch for overfitting |

    Start with the recommended LR and rank 32, then sweep outward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transferring LR across models

    If you have a good LR for one model, use `get_lora_lr_multiplier` to estimate the LR for another:
    """)
    return


@app.cell
def _(get_lora_lr_multiplier):
    # Suppose LR=5e-4 works well for Qwen3-4B
    known_lr = 5e-4
    source_model = "Qwen/Qwen3-4B-Instruct-2507"
    target_model = "Qwen/Qwen3-8B"

    source_mult = get_lora_lr_multiplier(source_model)
    target_mult = get_lora_lr_multiplier(target_model)
    estimated_lr = known_lr * target_mult / source_mult

    print(f"Source:    {source_model}, LR = {known_lr}")
    print(f"Target:    {target_model}, estimated LR = {estimated_lr:.6f}")
    print(f"Ratio:     {target_mult / source_mult:.4f}")
    return (estimated_lr, known_lr, source_model, source_mult, target_model, target_mult)


if __name__ == "__main__":
    app.run()
