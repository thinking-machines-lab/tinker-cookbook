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

    Finding the right learning rate and LoRA rank for SFT. The cookbook provides a `sweep` module that automates grid search over these parameters.

    The two most important hyperparameters for LoRA SFT are:
    - **Learning rate** -- too high diverges, too low underfits
    - **LoRA rank** -- higher rank = more expressive adapter, but higher compute cost

    The right values depend on the model, dataset, and task. Rather than guessing, run a sweep.
    """)
    return


@app.cell
def _():
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    return (compute_schedule_lr_multiplier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running a sweep

    The `sweep.run()` function takes a recipe's `main` function and config, then runs all combinations of the specified hyperparameters. Each run gets its own log directory, and results are collected into a DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CLI

    The quickest way to run a sweep:

    ```bash
    python -m tinker_cookbook.recipes.chat_sl.sweep \
        recipe=sft \
        base.model_name=Qwen/Qwen3.5-4B \
        base.dataset=tulu3 \
        'learning_rates=[1e-4, 3e-4, 1e-3]' \
        'lora_ranks=[32, 128]'
    ```

    This runs 6 configurations (3 LRs x 2 ranks) sequentially and prints a results table.

    ### Python API

    For more control, use the Python API directly:

    ```python
    from tinker_cookbook.recipes.chat_sl import sweep
    from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main

    results = sweep.run(
        cli_main,
        CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
        learning_rate=[1e-4, 3e-4, 1e-3],
        lora_rank=[32, 128],
    )

    # Results is a pandas DataFrame with one row per run
    best = results.loc[results["train_mean_nll"].idxmin()]
    print(f"Best LR: {best['learning_rate']:.2e}, rank: {best['lora_rank']}")
    ```

    For parallel execution, set `max_parallel`:

    ```python
    results = sweep.run(
        cli_main,
        CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
        max_parallel=4,
        learning_rate=[1e-4, 3e-4, 1e-3],
        lora_rank=[32, 128],
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How it works

    `sweep.run()` builds a cartesian product grid over the parameters you specify, then runs each configuration:
    """)
    return


@app.cell
def _():
    from tinker_cookbook.recipes.chat_sl.sweep.grid import grid

    # Generate all combinations
    configs = grid(learning_rate=[1e-4, 3e-4, 1e-3], lora_rank=[32, 128])
    print(f"Grid: {len(configs)} configurations\n")
    for _cfg in configs:
        print(f"  lr={_cfg['learning_rate']:.0e}, rank={_cfg['lora_rank']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each run writes `metrics.jsonl` and `config.json` to its own subdirectory. After the sweep, `sweep.collect()` reads all results into a DataFrame:

    ```
    /tmp/tinker-sweeps/20260330_143000/
    ├── learning_rate=0.0001_lora_rank=32/
    │   ├── config.json
    │   └── metrics.jsonl
    ├── learning_rate=0.0001_lora_rank=128/
    │   ├── config.json
    │   └── metrics.jsonl
    └── ...
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference results

    Here are pre-computed sweep results across several models and configurations. Use these as starting points for your own experiments.

    ### DeepSeek V3.1 Base

    | LR | LoRA Rank | Test NLL | Train NLL |
    |---:|----------:|---------:|----------:|
    | 1e-04 | 1 | 0.4853 | 0.5137 |
    | 4e-04 | 2 | 0.4842 | 0.5132 |
    | **4e-04** | **4** | **0.4826** | **0.5128** |
    | 1e-03 | 4 | 0.4904 | 0.5221 |

    Best: **rank=4, lr=4e-04** (test NLL = 0.4826)

    ### Nemotron Nano 30B (3B active)

    | LR | LoRA Rank | Test NLL | Train NLL |
    |---:|----------:|---------:|----------:|
    | 4e-04 | 4 | 0.5482 | 0.6228 |
    | 4e-04 | 16 | 0.5455 | 0.6190 |
    | 2e-04 | 64 | 0.5501 | 0.6219 |
    | **4e-04** | **64** | **0.5449** | **0.6181** |

    Best: **rank=64, lr=4e-04** (test NLL = 0.5449)

    ### Nemotron Super 120B (12B active)

    | LR | LoRA Rank | Test NLL | Train NLL |
    |---:|----------:|---------:|----------:|
    | 4e-04 | 16 | 0.4783 | 0.5334 |
    | 4e-04 | 64 | 0.4776 | 0.5330 |
    | **1e-03** | **64** | **0.4767** | **0.5348** |

    Best: **rank=64, lr=1e-03** (test NLL = 0.4767)

    All results use tulu3 dataset, batch size 128, 780 training steps. Full results with NLL curves: [sft_sweep.md](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/chat_sl/results/sft_sweep.md).
    """)
    return


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
def _(compute_schedule_lr_multiplier):
    _total_steps = 100
    _schedules = ["linear", "cosine", "constant"]

    for _schedule in _schedules:
        _mults = [
            compute_schedule_lr_multiplier(_schedule, _step, _total_steps)
            for _step in range(_total_steps)
        ]
        print(f"{_schedule:>8}: start={_mults[0]:.2f}, mid={_mults[50]:.2f}, end={_mults[-1]:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing LoRA ranks

    Higher LoRA rank means more trainable parameters. Use `get_lora_param_count` to see the trade-off:
    """)
    return


@app.cell
def _():
    from tinker_cookbook.hyperparam_utils import get_lora_param_count

    _model = "Qwen/Qwen3.5-4B"
    _ranks = [8, 32, 128]

    print(f"Model: {_model}")
    print(f"{'Rank':<8} {'Params':<15} {'Params (M)':<12}")
    print("-" * 35)
    for _rank in _ranks:
        _count = get_lora_param_count(_model, lora_rank=_rank)
        print(f"{_rank:<8} {_count:<15,} {_count / 1e6:<12.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical recommendations

    1. **Start with a sweep** -- even a small one (3 LRs x 2 ranks) gives much better results than guessing
    2. **Use `training_budget_examples`** -- set a smaller budget for quick iteration, then scale up the best config
    3. **LR matters more than rank** -- for most tasks, rank 32 is sufficient; bad LR is harder to recover from
    4. **Watch for divergence** -- if NLL increases after initial decrease, the LR is too high
    5. **Use test NLL** -- if you have a held-out set, use `metric=test/nll` to avoid overfitting
    """)
    return


if __name__ == "__main__":
    app.run()
