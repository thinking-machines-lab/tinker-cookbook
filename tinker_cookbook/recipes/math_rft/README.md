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
# GSM8K with Qwen3-8B
python -m tinker_cookbook.recipes.math_rft.train \
    model_name="Qwen/Qwen3-8B" \
    env=gsm8k \
    group_size=16 \
    groups_per_batch=32 \
    learning_rate=1e-4 \
    max_tokens=1024

# MATH dataset
python -m tinker_cookbook.recipes.math_rft.train \
    env=math \
    model_name="Qwen/Qwen3-8B" \
    group_size=16 \
    groups_per_batch=64 \
    learning_rate=5e-5 \
    max_tokens=512
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

## Key Metrics

- `test/correct`: Pass@1 accuracy on the held-out test set (greedy decoding)
- `train/sample_accuracy`: Fraction of sampled solutions that are correct
- `train/solve_rate`: Fraction of problems with at least one correct solution
- `train/n_sft_datums`: Number of correct solutions used for SFT per batch
- `train/mean_nll`: Average negative log-likelihood on correct solutions

## Research Context

RFT is the simplest method that uses verifiable rewards for post-training. It sits between pure SFT (which requires pre-existing correct solutions) and RL/GRPO (which learns from both correct and incorrect solutions).

Key research questions:
1. **When does GRPO's advantage over RFT justify its complexity?**
   - On easy datasets (GSM8K), RFT may match GRPO since most problems are solvable
   - On hard datasets (MATH), GRPO may win since it learns from failures too
2. **Does RFT plateau earlier than GRPO?**
   - RFT can only learn from problems the model already solves
   - As the model improves, it solves more problems → more training data → further improvement
3. **Is RFT more stable than GRPO?**
   - No importance weights, no negative advantages
   - Pure SFT loss is well-understood and stable

## References

- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - Zelikman et al., 2022
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) - Snell et al., 2024
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - DeepSeek AI, 2025 (uses RFT as part of training pipeline)
