# Tinker Tutorials

A guided introduction to Tinker, from your first API call to building custom RL training pipelines.

These tutorials are [marimo](https://marimo.io/) notebooks — reactive Python notebooks stored as `.py` files.

## Prerequisites

- Python 3.10+
- A Tinker API key ([get one here](https://tinker-console.thinkingmachines.ai))

## Setup

```bash
uv pip install tinker tinker-cookbook marimo
export TINKER_API_KEY="your-api-key-here"
```

## Running a tutorial

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
marimo edit tutorials/101_hello_tinker.py
```

This opens the notebook in your browser with an interactive editor. Rendered versions are also available on the [Tinker docs site](https://tinker-docs.thinkingmachines.ai/tutorials).

## Tutorials

### Basics (1xx)

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 101 | [Hello Tinker](101_hello_tinker.py) | Architecture overview, client hierarchy, sampling from a model |
| 102 | [Your First SFT](102_first_sft.py) | Renderers, datum construction, training loop |
| 103 | [Async Patterns](103_async_patterns.py) | Concurrent futures, `num_samples`, batch evaluation throughput |
| 104 | [First RL](104_first_rl.py) | GRPO on GSM8K: reward functions, group-relative advantages |

### Core Concepts (2xx)

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 201 | [Rendering](201_rendering.py) | Renderers, tokenization, vision inputs, TrainOnWhat |
| 202 | [Loss Functions](202_loss_functions.py) | cross_entropy, IS, PPO, CISPO, custom loss |
| 203 | [Completers](203_completers.py) | TokenCompleter vs MessageCompleter, LLM-as-judge |
| 204 | [Weights](204_weights.py) | Checkpoint lifecycle, save/load/download/TTL |
| 205 | [Evaluations](205_evaluations.py) | Custom evaluators, NLL, Inspect AI |

### Cookbook Abstractions (3xx)

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 301 | [Cookbook Abstractions](301_cookbook_abstractions.py) | `Env`, `EnvGroupBuilder`, `RLDataset`, `ProblemEnv` |
| 302 | [Custom Environment](302_custom_environment.py) | Build your own `ProblemEnv` subclass and `RLDataset` |
| 303 | [SFT with Config](303_sft_with_config.py) | `train.Config`, `ChatDatasetBuilder`, `train.main()` |
| 304 | [RL with Config](304_rl_with_config.py) | `RLDatasetBuilder`, RL training pipeline |

### Advanced (4xx)

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 401 | [SL Hyperparameters](401_sl_hyperparams.py) | LR scaling, rank selection, sweeps |
| 402 | [RL Hyperparameters](402_rl_hyperparams.py) | KL penalty, group size, advantages |
| 403 | [DPO & Preferences](403_dpo_preferences.py) | Comparison, DPO loss, PreferenceModel |
| 404 | [Sequence Extension](404_sequence_extension.py) | Multi-turn RL, conversation masks |
| 405 | [Multi-Agent RL](405_multi_agent.py) | MessageEnv, self-play, group rewards |
| 406 | [Prompt Distillation](406_prompt_distillation.py) | Teacher/student, context distillation |
| 407 | [RLHF Pipeline](407_rlhf_pipeline.py) | 3-stage SFT, preference model, RL |

### Deployment (5xx)

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 501 | [Export to HF](501_export_hf.py) | Merge LoRA into full model |
| 502 | [Build LoRA Adapter](502_lora_adapter.py) | PEFT format for vLLM/SGLang |
| 503 | [Publish to Hub](503_publish_hub.py) | Upload to HuggingFace with model card |

Work through them in order — each builds on concepts from the previous one.

## After the tutorials

- **Production recipes** with logging, checkpointing, and evaluation: see [`tinker_cookbook/recipes/`](../tinker_cookbook/recipes/)
- **Full documentation**: see the [Tinker docs site](https://tinker-docs.thinkingmachines.ai)
- **API reference**: see the [Tinker API reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)
