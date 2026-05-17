# Hyperparameter Selection

Detailed guide for choosing training hyperparameters across SL, RL, DPO, and distillation.

## Reference

- `tinker_cookbook/hyperparam_utils.py` — LR formulas and calculations

## Learning rate formula

The recommended LR for LoRA:

```
LR(m) = lr_base * M_LoRA * (2000 / H_m) ^ P_m
```

Where:
- `lr_base = 5e-5`
- `M_LoRA = 10` (1 for full fine-tuning)
- `H_m` = hidden size of the model
- `P_m` = model-specific exponent (0.0775 for Qwen, 0.781 for Llama)

This formula gives <0.5% regret vs exhaustive sweeps across diverse SFT experiments.

```python
from tinker_cookbook.hyperparam_utils import get_lr
lr = get_lr("Qwen/Qwen3.5-9B-Base", is_lora=True)
```

## LoRA rank

- **Default**: 32 for most tasks
- **Higher rank** (64-128): More capacity for complex tasks or large models
- **Lower rank** (8-16): Faster, sufficient for simple adaptations
- LR is independent of LoRA rank (validated empirically)

```python
from tinker_cookbook.hyperparam_utils import get_lora_param_count
params = get_lora_param_count("Qwen/Qwen3.5-9B-Base", lora_rank=32)
```

## Batch size

### SL batch size
- Measured in **tokens**, not examples
- Start with 128
- Scale LR proportionally: `LR ~ sqrt(batch_size)`
- Aim for at least 100 training steps (best results with 1000+)

### RL batch size and group size
- `batch_size` / `groups_per_batch`: Number of unique problems per batch
- `group_size`: Rollouts per problem (advantages centered within group)
- `total_rollouts = batch_size * group_size`
- Start small for debugging: `groups_per_batch=4, group_size=2`

## Learning rate schedule

- `"linear"` — Linear decay to 0 (most common)
- `"cosine"` — Cosine annealing
- `"constant"` — No decay

## `num_substeps` (RL)

- `num_substeps=1` (default): One update per batch
- `num_substeps>1`: Splits batch into mini-batches. Requires PPO objective.
- Start with 2-4 if experimenting; decrease LR with higher values

## DPO-specific

- `dpo_beta=0.1` — Well-tested default
- Lower beta = more aggressive; higher beta = closer to reference

## Distillation-specific

- `kl_penalty_coef=1.0` — Weight of KL penalty from teacher
- `kl_discount_factor=0.0` — No discounting (increase for long sequences)

## Quick-start recommendations

| Scenario | Model | LR | Batch | LoRA Rank |
|----------|-------|-----|-------|-----------|
| SFT on chat data | Qwen3.5-9B-Base | `get_lr(model)` | 128 | 32 |
| Math GRPO | Qwen3.5-9B (non-thinking) | 4e-5 | 128x16 | 32 |
| DPO | Qwen3.5-9B-Base | 1e-5 | 256 | 32 |
| Distillation | Qwen3.5-9B-Base | 1e-4 | 1024x4 | 128 |
| Multi-turn RL | Kimi-K2.6 | 1e-5 | 8x4 | 32 |

## Pitfalls

- `get_lr()` currently only supports Llama and Qwen families
- DPO LR should be much lower than SFT (1e-5 vs 2e-4)
- RL LR should be lower than SFT — too aggressive updates destabilize the policy
- Monitor KL divergence in RL — training is stable when KL < 0.01
