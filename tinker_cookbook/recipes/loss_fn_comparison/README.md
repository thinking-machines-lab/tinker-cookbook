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
L = -(log p_θ(x) * A(x) - 0.5 * β * (log(p_θ(x)/q(x)))²)
```

The quadratic term penalizes large divergence between the training and sampling policies. Configurable via `loss_fn_config={"beta": 0.05}`.

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

### Run all four sequentially

```bash
# Quick comparison on arithmetic
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

# Run only a subset
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep loss_fns=ppo,cispo
```

### Analyze results

```bash
# Compare metrics from all runs
python -m tinker_cookbook.recipes.loss_fn_comparison.analyze --log-dir /tmp/tinker-examples/math_rl
```

## Results

### Arithmetic (Llama-3.2-1B, 50 steps)

All four loss functions solve this toy task quickly. IS, PPO, and CISPO reach 95%+ reward in 3-5 steps. DRO converges 3-4x slower (12 steps to 95%) due to its quadratic penalty constraining update size. All reach 100% by step 50.

### GSM8K (Qwen3-8B, 100 steps, lr=2e-5, group_size=16)

This is where the loss functions diverge meaningfully:

| Step | IS | PPO | CISPO | DRO |
|------|------|------|------|------|
| 0 | 8.9% | 6.6% | 7.7% | 8.1% |
| 10 | 11.3% | 12.1% | 11.2% | 6.6% |
| 20 | 42.9% | **51.3%** | 46.1% | 7.4% |
| 30 | **88.7%** | 85.7% | 88.5% | 8.7% |
| 40 | 93.3% | 93.0% | **93.7%** | 10.1% |
| 50 | 92.9% | **93.8%** | 93.4% | 12.5% |
| 80 | **94.2%** | 94.0% | 93.3% | 18.2% |

**Best test accuracy:** IS=94.2%, PPO=94.0%, CISPO=93.7%, DRO=19.9%

Key observations:

1. **IS, PPO, and CISPO all reach ~94% test accuracy.** The three IS-based losses converge to essentially the same final performance.
2. **PPO leads early** (51% at step 20 vs 43-46% for IS/CISPO). Its clipping stabilizes the rapid early policy changes.
3. **CISPO maintains the highest entropy** (0.085 at step 99 vs PPO 0.035, IS 0.059) while matching accuracy — it keeps the policy more exploratory, consistent with preserving gradient signal for diverse tokens.
4. **DRO fails on this on-policy task.** The quadratic KL penalty is too conservative for the large policy changes needed to learn math reasoning from scratch. Even with reduced `beta=0.01`, DRO only reaches 29% after 100 steps.

### When DRO makes sense

DRO is designed for **off-policy/offline** RL where rollout data is stale and conservative updates prevent distributional collapse. On standard on-policy training (fresh rollouts every step), the conservatism slows learning without benefit. Use DRO when:
- Training on a fixed dataset of pre-collected rollouts
- Mixing data from multiple policies or sources
- Using `async_config` with high `max_steps_off_policy`

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
- `advantages`: Per-token advantages (group-relative, mean-centered: `R - mean(R)`, no std normalization)

Optional `loss_fn_config` parameters:

| Loss | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `ppo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS ratio |
| `ppo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS ratio |
| `cispo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS weight |
| `cispo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS weight |
| `dro` | `beta` | 0.05 | Strength of quadratic KL penalty |
