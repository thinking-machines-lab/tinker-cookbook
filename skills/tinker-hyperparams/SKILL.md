---
name: tinker-hyperparams
description: Guide for hyperparameter selection — learning rate formulas, LoRA rank, batch size, group size, schedules, and model-specific tuning. Use when the user asks about learning rate, batch size, hyperparameter tuning, or how to configure training parameters.
---

# Hyperparameter Selection

Guide for choosing training hyperparameters across SL, RL, DPO, and distillation.

## Reference

- `docs/supervised-learning/sl-hyperparams.mdx` — SL hyperparameter guide with LR formula
- `docs/rl/rl-hyperparams.mdx` — RL hyperparameters (batch_size, group_size, num_substeps, async)
- `tinker_cookbook/hyperparam_utils.py` — LR formulas and model-specific calculations

## Learning rate

### The formula

The recommended LR for a model `m` with LoRA:

```
LR(m) = lr_base × M_LoRA × (2000 / H_m) ^ P_m
```

Where:
- `lr_base = 5e-5`
- `M_LoRA = 10` (1 for full fine-tuning)
- `H_m` = hidden size of the model
- `P_m` = model-specific exponent (0.0775 for Qwen, 0.781 for Llama)

### Use the helper function

```python
from tinker_cookbook.hyperparam_utils import get_lr

lr = get_lr("meta-llama/Llama-3.1-8B", is_lora=True)
# Returns model-specific recommended LR
```

This formula gives <0.5% regret vs exhaustive sweeps across diverse SFT experiments.

### Rules of thumb

| Training type | Typical LR range | Notes |
|---------------|------------------|-------|
| SL (LoRA) | 1e-4 to 5e-4 | Use `get_lr()` |
| SL (full FT) | 1e-5 to 5e-5 | LoRA LR / 10 |
| RL | 1e-5 to 4e-5 | Lower than SL |
| DPO | ~1e-5 | Much lower than SL |
| RLHF (RL stage) | ~1e-5 | Same as RL |
| Distillation | ~1e-4 | Similar to SL |

## LoRA rank

- **Default**: 32 for most tasks
- **Higher rank** (64–128): More capacity, needed for complex tasks or large models
- **Lower rank** (8–16): Faster, sufficient for simple adaptations
- LR is **independent** of LoRA rank (validated empirically)

```python
from tinker_cookbook.hyperparam_utils import get_lora_param_count

# Check parameter count for a given rank
params = get_lora_param_count("meta-llama/Llama-3.1-8B", lora_rank=32)
```

## Batch size

### SL batch size
- Measured in **tokens**, not examples
- **Recommended**: Start with 128
- Smaller batch sizes often give better final performance at cost of longer training
- Scale LR proportionally: `LR ∝ √batch_size`
- Aim for at least 100 training steps (best results with 1000+)

### RL batch size and group size
Two parameters control RL batch composition:

- **`batch_size`** (or `groups_per_batch`): Number of unique problems/environments per batch
- **`group_size`**: Number of rollouts per problem (advantages centered within group)

```
total_rollouts = batch_size × group_size
```

Guidelines:
- If limited problems: increase `group_size` for more training signal
- Scale LR with batch_size: `LR ∝ √batch_size`
- Start small for debugging (`groups_per_batch=4, group_size=2`)

## Learning rate schedule

Available schedules:
- `"linear"` — Linear decay to 0 (most common)
- `"cosine"` — Cosine annealing
- `"constant"` — No decay

Set via `lr_schedule` parameter in config.

## `num_substeps` (RL)

Controls how many optimizer updates per sampling iteration:

- `num_substeps=1` (default): One update per batch — simplest, usually sufficient
- `num_substeps>1`: Splits batch into mini-batches, one update each. Requires PPO objective.
- Start with 2–4 if experimenting; decrease LR with higher values

## DPO-specific

- **`dpo_beta=0.1`** — Well-tested default. Controls deviation from reference model.
- Lower beta = more aggressive optimization
- Higher beta = stays closer to reference

## Distillation-specific

- **`kl_penalty_coef=1.0`** — Weight of KL penalty from teacher
- **`kl_discount_factor=0.0`** — No discounting (increase for long sequences)

## Quick-start recommendations

| Scenario | Model | LR | Batch | LoRA Rank |
|----------|-------|-----|-------|-----------|
| SFT on chat data | Llama-3.1-8B | `get_lr(model)` | 128 | 32 |
| Math GRPO | Llama-3.1-8B-Instruct | 4e-5 | 128×16 | 32 |
| DPO | Llama-3.2-1B | 1e-5 | 256 | 32 |
| Distillation | Qwen3-8B-Base | 1e-4 | 1024×4 | 128 |
| Multi-turn RL | Kimi-K2-Thinking | 1e-5 | 8×4 | 32 |

## Common pitfalls
- LoRA needs ~10x higher LR than full fine-tuning — use `get_lr()` to get it right
- `get_lr()` currently only supports Llama and Qwen families — other models need manual tuning
- DPO LR should be much lower than SFT (1e-5 vs 2e-4)
- RL LR should be lower than SFT — too aggressive updates destabilize the policy
- Batch size too small = noisy gradients; too large = diminishing returns
- Monitor KL divergence in RL — training is stable when KL < 0.01
