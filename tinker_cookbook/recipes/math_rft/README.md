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
| `env` | `math` | Dataset: `gsm8k` or `math` |
| `data_path` | None | Local data directory (e.g., `~/data`). Downloads from HF if None |
| `group_size` | 16 | K: solutions sampled per problem |
| `groups_per_batch` | 32 | Problems per training batch |
| `learning_rate` | 1e-4 | Adam learning rate |
| `max_tokens` | 2048 | Max generation length |
| `max_length` | 3072 | Max sequence length for SFT datums |
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

Train: DigitalLearningGmbH/MATH-lighteval (7500), Eval: HuggingFaceH4/MATH-500

```
Step  Overall  L1     L2     L3     L4     L5
  0   42.8%   88%    68%    49%    36%    13%
  5   80.4%   93%    92%    89%    80%    62%
 10   78.4%   93%    88%    87%    78%    61%
 15   79.0%   93%    88%    88%    76%    65%
 20   79.2%   91%    92%    87%    78%    62%
 30   80.2%   95%    90%    90%    79%    63%
 40   80.2%   95%    87%    88%    80%    65%
```

Config: `group_size=16, groups_per_batch=32, lr=1e-4, max_tokens=2048, lora_rank=32`

Key observations:
- **+37.6pp in 5 steps** (42.8% -> 80.4%), then **complete plateau** for 35 more steps
- L1-L2 saturate at 87-95%, but **L5 is stuck at ~63%** -- RFT cannot break through
- Training solve rate reaches 90-100% even for L5, but test L5 doesn't improve --
  the model solves seen problems but doesn't generalize

### GRPO comparison on MATH-500 (same model, same group_size)

```
Step  RFT      GRPO
  0   42.8%    43.4%
  5   80.4%    55.2%          <- RFT dominates early
 10   78.4%    74.6%
 15   79.0%    81.4%          <- crossover
 20   79.2%    84.6%          <- GRPO breaks through
 25   79.6%    84.2%
 30   80.2%    83.4%
 35   80.4%    84.4%
```

RFT eval: greedy (T=0). GRPO eval: on-policy (T=1.0).
GRPO config: `lr=8e-5, loss_fn=importance_sampling`

The crossover at step 15 reveals when GRPO's richer gradient signal (learning from
both correct and incorrect solutions) overcomes RFT's faster initial convergence.
GRPO at step 20 (84.6%) clearly surpasses RFT's ceiling (~80%).

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
underperforms pure GRPO (~85%). RFT appears to push the model into a low-entropy local
optimum that GRPO's small-step updates can't escape. See NOTES.md for detailed analysis.

## Practical Recommendations

1. **Easy tasks (GSM8K-level):** RFT alone is sufficient. 5-10 steps to ~94%.
2. **Hard tasks (MATH-level):** Use pure GRPO. It's slower but breaks through ceilings.
3. **Don't naively warm-start GRPO from RFT.** The entropy collapse hurts more than
   the better initialization helps. If a two-stage approach is needed, consider
   entropy-preserving alternatives (KL regularization during RFT, LR warmup on transition).

## References

- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - Zelikman et al., 2022
- [Yuan et al.: Scaling Mathematical Reasoning with LLMs](https://arxiv.org/abs/2308.01825) - coined RFT, showed distinct reasoning paths matter
- [ReST-EM: Beyond Human Data](https://arxiv.org/abs/2312.06585) - EM formulation of self-training
- [DART-Math: Difficulty-Aware Rejection Tuning](https://arxiv.org/abs/2407.13690) - NeurIPS 2024, addresses easy-problem bias
- [A Minimalist Approach to LLM Reasoning](https://arxiv.org/abs/2504.11343) - GRPO vs RAFT++ comparison, entropy collapse analysis
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek AI, 2025 (uses RFT as part of training pipeline)
