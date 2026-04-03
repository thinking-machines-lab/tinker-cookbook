# Comparing RL Loss Functions

Tinker provides four built-in RL loss functions: `importance_sampling`, `ppo`, `cispo`, and `dro`. This recipe trains the **same task** with each loss function so you can see how they differ in practice.

## Background

All four losses optimize the same objective — increase the probability of high-reward completions — but differ in how they handle the importance sampling ratio between the sampling and training policies.

### importance_sampling (REINFORCE with IS correction)

The default. Applies the raw importance weight to correct for the distribution shift between sampler and learner:

```
L = -(p_θ(x) / q(x)) * A(x)
```

**When to use:** On-policy or near-on-policy training with a single gradient step per rollout batch. Simple and effective for most tasks.

### ppo (Proximal Policy Optimization)

Clips the importance ratio to prevent destructively large policy updates:

```
L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

Default clip range: `[0.8, 1.2]`. Configurable via `loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}`.

**When to use:** When training is unstable with `importance_sampling`, or when taking multiple gradient steps per rollout batch (`num_substeps > 1`).

### cispo (Clipped Importance Sampling Policy Optimization)

From the MiniMax-M1 paper. Clips the IS ratio but applies it as a **detached weight** on `log p_θ` rather than clipping the product:

```
L = -sg(clip(ratio, 1-ε_low, 1+ε_high)) * A * log p_θ(x)
```

where `sg()` is stop-gradient. Unlike PPO, **all tokens contribute gradients** — rare but important tokens (e.g., "Wait", "Actually", correction tokens in chain-of-thought) are not dropped.

Default clip range: `[0.8, 1.2]`. Configurable via `loss_fn_config`.

**When to use:** Long chain-of-thought reasoning tasks where rare correction/reflection tokens carry important learning signal. Can converge faster than PPO on reasoning-heavy tasks.

### dro (Distributionally Robust Optimization)

Adds a quadratic KL penalty that makes the policy conservative about deviating too far from the sampling policy:

```
L = -log p_θ(x) * A(x) + 0.5 * β * (log(p_θ/q))²
```

Configurable via `loss_fn_config={"beta": 0.05}`.

**When to use:** Off-policy or offline RL settings, or when you want robust updates that are less sensitive to distribution shift. Particularly relevant when training on stale rollout data or when mixing data from multiple sources.

## Quick start

### Run a single loss function

The existing `math_rl` recipe already supports `loss_fn` as a parameter:

```bash
# Importance sampling (default)
python -m tinker_cookbook.recipes.math_rl.train loss_fn=importance_sampling

# PPO with default clipping
python -m tinker_cookbook.recipes.math_rl.train loss_fn=ppo

# CISPO
python -m tinker_cookbook.recipes.math_rl.train loss_fn=cispo

# DRO
python -m tinker_cookbook.recipes.math_rl.train loss_fn=dro

# PPO with custom clipping thresholds
python -m tinker_cookbook.recipes.math_rl.train loss_fn=ppo 'loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}'
```

### Run all four in parallel (xmux sweep)

```bash
# Quick comparison on arithmetic (~5 min)
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep

# On GSM8K with a larger model
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep \
    env=gsm8k \
    model_name="Qwen/Qwen3-8B" \
    max_tokens=512 \
    groups_per_batch=64 \
    group_size=16 \
    learning_rate=2e-5 \
    max_steps=100

# Dry run (see commands without executing)
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep --dry-run
```

### Analyze results

```bash
# Compare metrics from all runs
python -m tinker_cookbook.recipes.loss_fn_comparison.analyze --log-dir /tmp/tinker-examples/math_rl
```

## Expected results

On **arithmetic** (Llama-3.2-1B, 50 steps):

All four loss functions solve this toy task quickly (reward approaches 1.0 within ~10 steps). The differences are subtle at this scale — you may observe:

- `importance_sampling` and `cispo` converge slightly faster due to unbounded/wider gradient signal
- `ppo` is the most conservative, with the smoothest reward curve
- `dro` behaves similarly to `importance_sampling` with light regularization

On **GSM8K** or longer reasoning tasks, the differences become more pronounced — particularly `cispo` vs `ppo` on chain-of-thought stability, and `dro` vs `importance_sampling` on robustness to off-policy data.

## Choosing a loss function

| Scenario | Recommended | Why |
|----------|------------|-----|
| Default / getting started | `importance_sampling` | Simple, effective, the default |
| Training instability (loss spikes) | `ppo` | Clipping prevents large updates |
| Long chain-of-thought reasoning | `cispo` | Preserves gradients for rare correction tokens |
| Off-policy / stale rollout data | `dro` | Quadratic penalty handles distribution shift |
| Multiple gradient steps per batch | `ppo` or `cispo` | Both handle multi-step updates well |

## Configuration reference

All RL loss functions accept the same `loss_fn_inputs` in each Datum:
- `target_tokens`: The token sequence to compute logprobs on
- `logprobs`: Sampling logprobs from the policy that generated the completion
- `advantages`: Per-token advantages (typically group-relative: `(R - mean) / std`)

Optional `loss_fn_config` parameters:

| Loss | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `ppo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS ratio |
| `ppo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS ratio |
| `cispo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS weight |
| `cispo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS weight |
| `dro` | `beta` | 0.05 | Strength of quadratic KL penalty |
