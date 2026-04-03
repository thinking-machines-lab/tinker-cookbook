# Rejection Sampling Fine-Tuning (RFT) for Math Reasoning

This recipe implements **iterative Rejection Sampling Fine-Tuning (RFT)**, also known as the core loop of STaR (Self-Taught Reasoner). It provides a clean comparison with the GRPO approach in `math_rl/`.

## How It Works

```
For each batch of math problems:
  1. Sample K solutions per problem from the current model
  2. Grade each solution using verifiable math rewards
  3. Keep only correct solutions
  4. Fine-tune on correct solutions using standard SFT loss
  5. Update model and repeat
```

## RFT vs GRPO

Both methods sample K solutions per problem. The difference is in how they use those samples:

| Aspect | GRPO | RFT |
|--------|------|-----|
| Training data | All solutions | Only correct solutions |
| Loss function | Importance-weighted policy gradient | Standard cross-entropy (SFT) |
| Advantage signal | reward - group_mean | Binary (correct = include, wrong = exclude) |
| Learning from failures | Yes (negative advantages) | No |
| Stability | Can be unstable (importance weights) | Very stable (pure SFT) |

## Quick Start

```bash
# MATH dataset with Qwen3-8B
python -m tinker_cookbook.recipes.math_rft.train \
    model_name="Qwen/Qwen3-8B" \
    env=math \
    group_size=16 \
    groups_per_batch=32 \
    learning_rate=1e-4 \
    max_tokens=2048 \
    max_length=3072

# GSM8K (easier)
python -m tinker_cookbook.recipes.math_rft.train \
    model_name="Qwen/Qwen3-8B" \
    env=gsm8k \
    group_size=16 \
    groups_per_batch=32 \
    learning_rate=1e-4 \
    max_tokens=1024
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-8B` | HuggingFace model identifier |
| `env` | `gsm8k` | Dataset: `gsm8k` or `math` |
| `group_size` | 16 | K: solutions sampled per problem |
| `groups_per_batch` | 64 | Problems per training batch |
| `learning_rate` | 2e-5 | Adam learning rate |
| `max_tokens` | 1024 | Max generation length |
| `max_length` | 2048 | Max sequence length for SFT datums |
| `temperature` | 1.0 | Sampling temperature |
| `eval_every` | 5 | Evaluate every N batches |
| `lora_rank` | 32 | LoRA adapter rank |
| `max_datums_per_problem` | None | Limit SFT datums per problem (None = all correct) |

## Results

### GSM8K with Qwen3-8B

```
Step  pass@1   sample_acc  NLL
  0   62.4%    67.0%       0.377
  5   91.4%    98.2%       0.201
 10   94.0%    91.2%       0.252   <- peak
 15   93.6%    93.0%       0.244
 30   93.4%    --          --
```

Config: `group_size=16, groups_per_batch=32, lr=1e-4, max_tokens=1024, lora_rank=32`

### MATH-500 with Qwen3-8B (per-difficulty breakdown)

```
Step  Overall  L1     L2     L3     L4     L5     Format
  0   42.2%   81.4%  72.2%  47.6%  32.8%  14.2%  44.0%
  5   78.8%   93.0%  91.1%  87.6%  76.6%  61.2%  90.4%
 10   78.6%   97.7%  86.7%  85.7%  79.7%  60.4%  92.6%
 20   78.2%   93.0%  85.6%  85.7%  79.7%  61.2%  90.4%
 40   78.8%   93.0%  92.2%  88.6%  75.8%  60.4%  --
```

Config: `group_size=16, groups_per_batch=32, lr=1e-4, max_tokens=2048, lora_rank=32`

Key observations:
- **+36.6pp in 5 steps** (42.2% -> 78.8%), then **complete plateau** for 35 more steps
- L1-L2 saturate at 90%+, but **L5 is stuck at ~60%** -- RFT cannot break through
- Format compliance jumps 44% -> 90% in 5 steps (easy for SFT to learn)
- Training solve rate reaches 85-100% even for L5, but test L5 doesn't improve --
  the model solves seen problems but doesn't generalize

### GRPO comparison on MATH-500 (same model, same group_size)

```
Step  RFT (greedy)  GRPO (T=1.0)
  0   42.2%         35.9%
  5   78.8%         46.9%        <- RFT dominates early
 10   78.6%         67.3%
 15   78.0%         77.5%        <- crossover
 20   78.2%         82.3%        <- GRPO breaks through
 25   79.6%         81.6%
 30   79.8%         84.1%
 35   79.6%         85.1%        <- GRPO still climbing
```

GRPO config: `lr=8e-5, loss_fn=importance_sampling`

The crossover at step 15 reveals when GRPO's richer gradient signal (learning from
both correct and incorrect solutions) overcomes RFT's faster initial convergence.
GRPO at step 35 (85.1% at T=1.0) clearly surpasses RFT's ceiling (78.8% at greedy).

## Why RFT Plateaus on Hard Tasks

RFT's plateau is NOT caused by inability to find correct solutions -- the model
achieves 85-100% training solve rate by step 4. The real bottleneck:

1. **Redundant gradients**: Correct solutions become increasingly similar; SFT loss
   on near-identical outputs provides diminishing signal
2. **No negative signal**: RFT cannot push the model away from systematic errors
3. **Easy problem bias**: Easy problems generate more correct solutions, dominating
   the gradient even though the model already masters them

GRPO addresses all three: advantage weighting upweights rare successes, negative
advantages penalize failure patterns, and importance weighting provides richer gradients.

### RFT->GRPO hybrid (warm-start experiment)

5 steps RFT (lr=1e-4), then 35 steps GRPO (lr=8e-5) from RFT checkpoint:

```
GRPO step  Pure GRPO  Hybrid (RFT+GRPO)
  0        35.9%      77.1%
  5        46.9%      75.8%
 10        67.3%      78.3%
 15        77.5%      78.5%
 20        82.3%      78.6%
 25        81.6%      78.7%
 30        84.1%      79.3%
```

**Surprising negative result**: The naive warm-start *hurts* GRPO. The hybrid (79.3%)
underperforms pure GRPO (85.1%). RFT appears to push the model into a low-entropy local
optimum that GRPO's small-step updates can't escape. See NOTES.md for detailed analysis.

## Practical Recommendations

1. **Easy tasks (GSM8K-level):** RFT alone is sufficient. 5-10 steps to ~94%.
2. **Hard tasks (MATH-level):** Use pure GRPO. It's slower but breaks through ceilings.
3. **Don't naively warm-start GRPO from RFT.** The entropy collapse hurts more than
   the better initialization helps. If a two-stage approach is needed, consider
   entropy-preserving alternatives (KL regularization during RFT, LR warmup on transition).

## References

- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - Zelikman et al., 2022
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) - Snell et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek AI, 2025 (uses RFT as part of training pipeline)
