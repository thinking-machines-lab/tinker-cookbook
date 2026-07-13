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

**Comparing across tokenizers:** `train_mean_nll` / `test/nll` are *per-token*
cross-entropy, which is not comparable across models with different tokenizers —
a coarser tokenizer packs more text into each token (higher nats/token) while a
finer one spreads it over more (lower nats/token). Training also logs
`train_mean_bpb` / `test/bpb` (bits per byte), which divide the total log-loss by
the number of UTF-8 bytes of the *semantic content* of the trained messages
(message text, rendered thinking, tool-call names/arguments) instead of by the
token count. Chat-template scaffolding — think tags, role markers, end-of-turn
tokens — is not counted as bytes, so a verbose template gains no artificial BPB
advantage: the denominator depends on *what* content a renderer trains on, not
on how its template formats it. That makes BPB comparable across tokenizers and
chat templates, with two caveats. Renderers that legitimately train on
different content count different bytes — e.g. gpt-oss preserves (and counts)
thinking in historical assistant turns that Qwen3/Kimi renderers strip — so
cross-model comparisons are cleanest on data without thinking in historical
turns. And datums without renderer-reported content bytes (custom renderers
that override only `build_supervised_example`, or examples whose trained span
was cut by `max_length`) fall back to the previous token-byte computation.
Sweeping over models? Select on bits per byte:

```bash
python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    metric=test/bpb
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
