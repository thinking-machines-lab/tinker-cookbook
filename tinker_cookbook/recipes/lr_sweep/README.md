# LR Sweep

Find the optimal learning rate for any model across any training recipe. Built on the [`tinker_cookbook.sweep`](../../sweep/) module.

Works with any recipe that follows the `CLIConfig` + `cli_main` convention — use a short alias or a full module path.

## Quick start

### SFT sweep

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    base.dataset=tulu3
```

### RL sweep (math)

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=math_rl \
    base.model_name=Qwen/Qwen3-8B \
    base.env=gsm8k \
    base.group_size=16 \
    base.groups_per_batch=64
```

### DPO sweep

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=dpo \
    base.model_name=Qwen/Qwen3-8B
```

### Any recipe via module path

For recipes without a short alias, pass the full module path:

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=tinker_cookbook.recipes.harbor_rl.train \
    base.model_name=Qwen/Qwen3-8B
```

Available aliases: `sft`, `math_rl`, `code_rl`, `harbor_rl`, `rubric`, `search_tool`, `dpo`, `shorter`, `distillation`.

## Configuring the sweep

All training parameters are inherited from the selected recipe's `CLIConfig` via the `base` field. Override any training parameter with `base.<param>=<value>`.

### Custom LR range and ranks

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    'learning_rates=[1e-4, 3e-4, 5e-4, 1e-3]' \
    'lora_ranks=[32, 64, 128]'
```

### Smaller budget for quick iteration

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    training_budget_examples=640 \
    'learning_rates=[1e-4, 3e-4, 1e-3]' \
    'lora_ranks=[32]'
```

With `batch_size=128` and `budget=640`, each run trains for 5 steps — useful for testing the pipeline.

### Custom metric

By default, the sweep optimizes `train_mean_nll`. For RL sweeps, you may want a different metric:

```bash
python -m tinker_cookbook.recipes.lr_sweep.sweep \
    recipe=math_rl \
    metric=env/all/reward/total
```

## Using the sweep module directly

The LR sweep recipe is built on `tinker_cookbook.sweep`, which can wrap any recipe config with sweep axes. You can use it directly in Python for more control.

### Sequential (default)

```python
from tinker_cookbook import sweep
from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main

results = sweep.run(
    cli_main,
    CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
    learning_rate=[1e-4, 3e-4, 1e-3],
    lora_rank=[32, 128],
)
best = results.loc[results["train_mean_nll"].idxmin()]
print(f"Best LR: {best['learning_rate']:.2e}")
```

### Parallel with ProcessPoolExecutor

Run multiple training jobs concurrently on the same machine:

```python
results = sweep.run(
    cli_main,
    CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
    max_parallel=4,
    learning_rate=[1e-4, 3e-4, 1e-3],
    lora_rank=[32, 128],
)
```

### Parallel with Ray

For distributed execution across a Ray cluster:

```python
from ray.util.multiprocessing.pool import Pool

results = sweep.run(
    cli_main,
    CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
    executor=Pool(),
    learning_rate=[1e-4, 3e-4, 1e-3],
    lora_rank=[32, 128],
)
```

### With xmux (tmux-based)

For long-running sweeps on a remote machine with interactive monitoring:

```python
from tinker_cookbook import sweep
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm
from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main

import chz

base = CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3")
grid = sweep.grid(learning_rate=[1e-4, 3e-4, 1e-3], lora_rank=[32, 128])

job_specs = [
    JobSpec(
        main_fn=cli_main,
        log_relpath=f"lr_sweep/{sweep.default_run_name(point)}",
        entrypoint_config=chz.replace(base, **point),
    )
    for point in grid
]
launch_swarm(job_specs, SwarmConfig(sweep_name="lr_sweep"))

# After all jobs complete:
results = sweep.collect("~/experiments/lr_sweep")
```

### With wandb logging

Set `wandb_project` on the base config — metrics are logged to both `metrics.jsonl` and wandb automatically:

```python
results = sweep.run(
    cli_main,
    CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3",
              wandb_project="lr-sweep"),
    learning_rate=[1e-4, 3e-4, 1e-3],
)
```

### Collecting results from external runs

If runs were launched externally (shell scripts, hydra, wandb agent, etc.), collect results from any directory containing run subdirectories with `metrics.jsonl` + `config.json`:

```python
results = sweep.collect("/path/to/sweep/dir")
```
