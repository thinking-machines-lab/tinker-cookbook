# Tinker Tutorials

A guided introduction to Tinker, from your first API call to building custom RL training pipelines.

These tutorials are [marimo](https://marimo.io/) notebooks — reactive Python notebooks stored as `.py` files.

## Prerequisites

- Python 3.10+
- `tinker-cookbook` installed (`pip install tinker-cookbook`)
- `marimo` installed (`pip install marimo`)
- A Tinker API key set as `TINKER_API_KEY` in your environment ([get one here](https://tinker-console.thinkingmachines.ai))

## Running a tutorial

```bash
marimo edit tutorials/01_hello_tinker.py
```

This opens the notebook in your browser with an interactive editor. Rendered versions are also available on the [Tinker docs site](https://tinker-docs.thinkingmachines.ai/tutorials).

## Tutorials

| # | Notebook | What you'll learn | Time |
|---|----------|-------------------|------|
| 01 | [Hello Tinker](01_hello_tinker.py) | Architecture overview, client hierarchy, sampling from a model | ~2 min |
| 02 | [Your First SFT](02_first_sft.py) | Renderers, datum construction, training loop, Kimi K2.5 scaling demo | ~5 min |
| 03 | [Efficient Sampling](03_async_patterns.py) | Concurrent futures, `num_samples`, batch evaluation throughput | ~5 min |
| 04 | [First RL](04_first_rl.py) | GRPO on GSM8K: reward functions, group-relative advantages, degenerate groups | ~10 min |
| 05 | [Cookbook RL Abstractions](05_custom_task.py) | `Env`, `EnvGroupBuilder`, `RLDataset`, `ProblemEnv` — how the raw loop maps to reusable types | ~10 min |
| 06 | [Custom RL Environment](06_custom_env.py) | Build your own `ProblemEnv` subclass and `RLDataset` for a new task | ~10 min |

Work through them in order -- each builds on concepts from the previous one.

## After the tutorials

- **Production recipes** with logging, checkpointing, and evaluation: see [`tinker_cookbook/recipes/`](../tinker_cookbook/recipes/)
- **Full documentation**: see [`docs/`](../docs/)
- **API reference**: see [`docs/api-reference/`](../docs/api-reference/)
