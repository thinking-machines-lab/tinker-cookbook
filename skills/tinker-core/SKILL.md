---
name: tinker-core
description: Core guide for using the Tinker API — installation, model selection, SDK basics, types, CLI, and hyperparameters. Use this skill whenever the user asks about getting started with Tinker, choosing a model, using the SDK, API types, CLI commands, or tuning hyperparameters. This is the foundational skill — trigger it for any general Tinker question.
---

# Tinker Core

Everything you need to get started with Tinker and the tinker-cookbook.

## Quick start

### 1. Get an API key

Sign up at [https://auth.thinkingmachines.ai/sign-up](https://auth.thinkingmachines.ai/sign-up), create a key from the [console](https://tinker-console.thinkingmachines.ai), and export it:

```bash
export TINKER_API_KEY=<your-key>
```

### 2. Install

```bash
pip install tinker                                    # SDK + CLI
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook && pip install -e .                 # Cookbook
```

For dev setup: `uv sync --extra dev && pre-commit install`

### 3. Verify

```python
import tinker
svc = tinker.ServiceClient()
tc = svc.create_lora_training_client(base_model="meta-llama/Llama-3.2-1B", rank=32)
print(tc.get_info())
```

### 4. Run a minimal example

```bash
python -m tinker_cookbook.recipes.sl_basic    # Supervised learning
python -m tinker_cookbook.recipes.rl_basic    # Reinforcement learning
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `TINKER_API_KEY` | Required — authenticates with Tinker service |
| `HF_TOKEN` | Optional — access gated HuggingFace models (Llama, etc.) |
| `WANDB_API_KEY` | Optional — log to Weights & Biases |

## Model selection

Always use the automatic renderer lookup:
```python
from tinker_cookbook import model_info
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

### By task type

- **Instruction tuning / chat SFT**: Instruction models (e.g., `Llama-3.1-8B-Instruct`, `Qwen3-30B-A3B-Instruct-2507`)
- **RL with verifiable rewards**: Instruction or Hybrid models
- **Reasoning / chain-of-thought**: Reasoning or Hybrid models (`Kimi-K2-Thinking`, `Qwen3-8B`)
- **Full post-training pipeline**: Base models (e.g., `Qwen3-8B-Base`, `Llama-3.1-8B`)
- **Vision tasks**: Vision or Hybrid+Vision models (`Qwen3.5-35B-A3B`, `Qwen3-VL-*`)
- **Quick prototyping**: Compact models (`Llama-3.2-1B`, `Qwen3.5-4B`)

### Cost tip

Prefer MoE models — cost scales with active parameters. `Qwen3-30B-A3B` (3B active) is cheaper than `Qwen3-32B` (32B active) at similar quality.

For the full model lineup with families, sizes, and architecture details, read `references/models.md`.

## SDK overview

`ServiceClient` is the entry point. All other clients are created from it:

```python
from tinker import ServiceClient, SamplingParams, AdamParams

svc = ServiceClient()

# Training
tc = svc.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)
result = tc.forward_backward(data=[datum], loss_fn="cross_entropy")
tc.optim_step(adam_params=AdamParams(learning_rate=2e-4))

# Sampling (IMPORTANT: create a NEW client after each weight save)
sc = tc.save_weights_and_get_sampling_client()
response = sc.sample(prompt=model_input, num_samples=4,
                     sampling_params=SamplingParams(max_tokens=256))
```

### Async pattern (critical for throughput)

Tinker runs GPU work on remote servers. The single most important performance pattern is **submitting async calls back-to-back before awaiting** — this overlaps GPU computation with CPU data preparation, maximizing throughput.

```python
# CORRECT: submit both, then await both — GPU works while CPU prepares next batch
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=adam_params)
# ... prepare next batch here while GPU is busy ...
fb_result = fb_future.result()
optim_result = optim_future.result()

# WRONG: awaiting between calls serializes everything — wastes GPU cycles
result = tc.forward_backward_async(data=batch, loss_fn="cross_entropy").result()  # GPU idle during this wait!
tc.optim_step_async(adam_params=adam_params).result()  # Sequential = slow
```

The same principle applies to **evaluation and sampling** — run them concurrently, not in a sequential loop:

```python
# CORRECT: evaluate all test problems concurrently
import asyncio

async def evaluate_one(sc, problem):
    response = await sc.sample_async(prompt=problem, num_samples=1,
                                     sampling_params=SamplingParams(max_tokens=256))
    return check_answer(response)

results = await asyncio.gather(*[evaluate_one(sc, p) for p in test_problems])

# WRONG: sequential loop — each problem waits for the previous one
for problem in test_problems:
    response = sc.sample(prompt=problem, ...)  # One at a time = very slow
```

This applies everywhere: training steps, sampling, evaluation, checkpoint saves. All SDK methods have `_async` variants. Never write sequential `.result()` chains or sequential `for` loops over API calls.

For the complete SDK API (all client methods, loss functions, retry behavior), read `references/sdk.md`.

## Core types

```
Datum
├── model_input: ModelInput (list of token/image chunks)
└── loss_fn_inputs: dict[str, TensorData]
```

```python
from tinker import Datum, ModelInput, TensorData
import numpy as np

mi = ModelInput.from_ints([1, 2, 3, 4, 5])
td = TensorData.from_numpy(np.array([1.0, 0.0, 1.0]))
datum = Datum(model_input=mi, loss_fn_inputs={"weights": td})
```

Use cookbook helpers instead of manual construction:
- `conversation_to_datum(messages, renderer, max_length, train_on_what)` — full pipeline
- `renderer.build_supervised_example(messages)` — returns (ModelInput, weights)
- `datum_from_model_input_weights(model_input, weights, max_length)` — from components

For the complete type reference (SamplingParams, AdamParams, response types, errors), read `references/types.md`.

## CLI

The `tinker` CLI manages runs and checkpoints from the terminal:

```bash
tinker run list                                    # List training runs
tinker checkpoint list --run-id <RUN_ID>           # List checkpoints
tinker checkpoint download <TINKER_PATH> -o ./adapter   # Download weights
tinker checkpoint push-hf <PATH> --repo user/model      # Push to HuggingFace
```

For the full CLI reference, read `references/cli.md`.

## Hyperparameters

### Learning rate

```python
from tinker_cookbook.hyperparam_utils import get_lr
lr = get_lr("meta-llama/Llama-3.1-8B", is_lora=True)
```

| Training type | Typical LR range |
|---------------|------------------|
| SL (LoRA) | 1e-4 to 5e-4 |
| RL | 1e-5 to 4e-5 |
| DPO | ~1e-5 |
| Distillation | ~1e-4 |

### Quick-start defaults

- **LoRA rank**: 32 (most tasks)
- **Batch size**: 128 tokens (SL), 128x16 (RL)
- **LR schedule**: `"linear"` decay

For the full hyperparameter guide (formulas, LoRA rank selection, per-scenario recommendations), read `references/hyperparams.md`.

## Common pitfalls

- **Sequential API calls**: The #1 performance mistake. Always use `_async` variants and submit calls back-to-back before awaiting. Sequential `.result()` chains waste GPU cycles. This applies to training, sampling, evaluation, and checkpointing.
- **Sampler desync**: After saving weights, the old SamplingClient still points at the previous weights. Always create a **new** SamplingClient after each `save_weights_for_sampler()` or `save_state()`. Reusing a stale client silently samples from outdated weights.
- **Renderer mismatch**: Always use `model_info.get_recommended_renderer_name()`, never hardcode
- **LoRA LR**: LoRA needs ~10x higher LR than full fine-tuning — use `get_lr()`
- **Llama tokenizer**: Requires `HF_TOKEN` (gated on HuggingFace)
- **forward() vs forward_backward()**: `forward()` computes loss without gradients — use it for evaluation only, never in the training loop

## Further reading

For deeper detail, read the reference files bundled with this skill:
- `references/sdk.md` — Complete SDK API
- `references/types.md` — All SDK types
- `references/models.md` — Full model lineup
- `references/cli.md` — CLI command reference
- `references/hyperparams.md` — Hyperparameter formulas and recommendations
