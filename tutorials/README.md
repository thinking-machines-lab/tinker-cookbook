# Tinker Tutorials

A guided introduction to Tinker, from your first API call to building custom RL training pipelines.

## Prerequisites

- Python 3.10+
- `tinker-cookbook` installed (`pip install tinker-cookbook`)
- A Tinker API key set as `TINKER_API_KEY` in your environment ([get one here](https://tinker-console.thinkingmachines.ai))

## Tutorials

| # | Notebook | What you'll learn | Time |
|---|----------|-------------------|------|
| 01 | [Hello Tinker](01_hello_tinker.ipynb) | Architecture overview, client hierarchy, sampling from a model | ~2 min |
| 02 | [Your First SFT](02_first_sft.ipynb) | Renderers, datum construction, `forward_backward` + `optim_step`, verifying the model learned | ~5 min |
| 03 | [Async Patterns](03_async_patterns.ipynb) | Double-await pattern, pipelining across steps, timing comparisons | ~5 min |
| 04 | [First RL](04_first_rl.ipynb) | GRPO on GSM8K: reward functions, group-relative advantages, degenerate groups | ~10 min |
| 05 | [Custom Task](05_custom_task.ipynb) | Adapt the RL pattern to your own dataset and reward function | ~10 min |

Work through them in order -- each builds on concepts from the previous one.

## After the tutorials

- **Production recipes** with logging, checkpointing, and evaluation: see [`tinker_cookbook/recipes/`](../tinker_cookbook/recipes/)
- **Full documentation**: see [`docs/`](../docs/)
- **API reference**: see [`docs/api-reference/`](../docs/api-reference/)
