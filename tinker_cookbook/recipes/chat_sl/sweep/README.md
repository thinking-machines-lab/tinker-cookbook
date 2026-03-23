# SFT Hyperparameter Sweep

Sweep infrastructure for finding optimal learning rate and LoRA rank for SFT recipes.

See [results/sft_sweep.md](../results/sft_sweep.md) for pre-computed sweep results across multiple models.

## CLI

```bash
python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    base.dataset=tulu3
```

### Custom LR range and ranks

```bash
python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    'learning_rates=[1e-4, 3e-4, 5e-4, 1e-3]' \
    'lora_ranks=[32, 64, 128]'
```

### Smaller budget for quick iteration

```bash
python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    training_budget_examples=640 \
    'learning_rates=[1e-4, 3e-4, 1e-3]' \
    'lora_ranks=[32]'
```

With `batch_size=128` and `budget=640`, each run trains for 5 steps — useful for testing the pipeline.

### Custom metric

By default, the sweep optimizes `train_mean_nll`. To use a different metric:

```bash
python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    metric=test/nll
```

## Python API

```python
from tinker_cookbook.recipes.chat_sl import sweep
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

### Parallel execution

```python
results = sweep.run(
    cli_main,
    CLIConfig(model_name="Qwen/Qwen3.5-4B", dataset="tulu3"),
    max_parallel=4,
    learning_rate=[1e-4, 3e-4, 1e-3],
    lora_rank=[32, 128],
)
```

### Collecting results from external runs

```python
results = sweep.collect("/path/to/sweep/dir")
```

## Regenerating results

To regenerate the results tables and NLL curve plots from W&B:

```bash
uv run --with wandb --with matplotlib python -m tinker_cookbook.recipes.chat_sl.sweep.analyze
```
