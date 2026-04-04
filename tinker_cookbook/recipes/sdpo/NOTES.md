# SDPO Research Notes

## Paper Reference
- **Title**: Reinforcement Learning via Self-Distillation (arXiv:2601.20802v2)
- **Authors**: Hubotter et al., ETH Zurich + MPI + MIT + Stanford
- **Code**: https://github.com/lasgroup/SDPO

## Algorithm Summary

SDPO replaces standard RL's scalar per-sequence reward with dense, token-level
credit assignment via self-distillation. Per training step:

1. **Rollout**: Sample G responses per question from current policy
2. **Evaluate**: Get rewards + environment feedback
3. **Build teacher prompt**: Condition the teacher on a successful solution from
   another rollout in the same group (+ optional env feedback like compiler errors)
4. **Teacher forward pass**: Teacher-force the original response tokens through the
   teacher model under the conditioned prompt → get full-vocab logprobs at each position
5. **Student forward pass**: Normal forward pass on original prompt + response
6. **Compute JSD loss**: Jensen-Shannon divergence between student and teacher
   distributions at each token position, using top-K approximation
7. **Gradient step**: With IS correction for off-policy drift

## Loss Function

The paper uses **Generalized JSD (α=0.5)** for the generalization setting:

```
JSD(student, teacher) = 0.5 * KL(student || M) + 0.5 * KL(teacher || M)
where M = 0.5 * (student + teacher)
```

### Our Approximation: 0.5 * CE + 0.5 * IS

We decompose the bidirectional KL as two separate losses:

**Term 1: Forward KL** via `cross_entropy` with top-K=20 teacher soft targets
- KL(teacher || student) = CE(teacher, student) - H(teacher)
- H(teacher) is constant w.r.t. student params → CE = forward KL for optimization
- Uses Tinker's `topk_prompt_logprobs` API to get teacher's top-K distribution
- Weights = teacher's renormalized top-K probabilities

**Term 2: Reverse KL** via `importance_sampling` as a REINFORCE proxy
- KL(student || teacher) = CE(student, teacher) - H(student)
- H(student) depends on student params → cannot use CE directly
- REINFORCE/score function trick handles the entropy term correctly:
  ∇_θ KL(p_θ || q) = E_{y~p_θ}[ (log p_θ(y) - log q(y)) · ∇_θ log p_θ(y) ]
- advantage = teacher_lp - student_lp (per sampled token)

**Combined loss**: `L = 0.5 * L_CE + 0.5 * L_IS`

This is NOT exactly JSD (which uses mixture M), but captures both KL directions.
The forward KL (CE) gives dense multi-token signal; reverse KL (IS) adds mode-seeking pressure.

### Differences from Paper

| Aspect | Paper | Our Implementation |
|---|---|---|
| Loss | JSD (α=0.5) with mixture M | 0.5*CE + 0.5*IS (both KL directions) |
| Top-K | K=100 (generalization), K=20 (code) | K=20 |
| Tail bucket | Yes (residual mass term) | No |
| Teacher | EMA (rate=0.05) or trust-region | Frozen reference |
| IS clipping | 2.0 | None (on-policy, ratio ≈ 1) |
| IS rollout correction | Per-token, threshold 2.0 | None |

### Justification for Differences

- **K=20 vs K=100**: SDFT experiments show K=20 captures most of the signal.
  Paper uses K=20 for rich-feedback setting anyway.
- **Frozen teacher vs EMA**: Table 4 shows frozen=48.8%, EMA=49.3% — minimal gap.
  Much simpler implementation.
- **No IS clipping**: We're on-policy (one gradient step per rollout batch), so
  the importance ratio p_θ/q ≈ 1. Clipping is unnecessary.
- **No tail bucket**: Simplification. The top-K captures most probability mass.

## Data: SciKnowEval

The paper trains and evaluates on SciKnowEval (hicai-zju/SciKnowEval).

**Our implementation matches the paper:**
- HF dataset: `hicai-zju/SciKnowEval`, split `test`
- Filter: L3 level, mcq-4-choices + mcq-2-choices
- Train/test split: test_ratio=0.1, seed=42
- Prompt: `{question}\n\n{choices}\nPlease reason step by step.`
- System prompt: XML answer format (`<answer>A</answer>`)
- Reward: Binary 0/1, exact letter match
- Zero-shot (no few-shot examples)

Minor difference: our system prompt says "answer options" vs paper's "four options".

**The paper does NOT train on MATH.** MATH-500 is used only as an eval dataset.
Our PR's MATH results (43%→63%) were our own experiment, not a paper reproduction.

## Paper Results (Table 3 — SciKnowEval, Qwen3-8B, avg@16)

| Domain | Base | GRPO (5h) | SDPO (5h) |
|---|---|---|---|
| Chemistry | 41.2 | 74.5 | **80.9** |
| Physics | 59.2 | 72.7 | **75.6** |
| Biology | 30.8 | **59.9** | 56.8 |
| Materials | 58.9 | 77.1 | **78.4** |

## Paper Hyperparameters (Generalization Setting — Table 12)

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Batch size (questions) | 32 |
| Group size (G) | 8 |
| Mini batch size | 32 (on-policy) |
| LR | 1e-5 |
| LR warmup | 10 steps |
| Top-K | 100 |
| Divergence | JSD (α=0.5) |
| Teacher | EMA (rate=0.05) |
| IS clip | 2.0 |
| Temperature | 1.0 |
| Max prompt length | 2048 |
| Max response length | 8192 |
| Total epochs | 30 |
| Eval frequency | Every 5 steps |
| Eval rollouts | avg@16 |

## Experiment Plan

### Phase 1: SciKnowEval Chemistry (paper's most reported domain)

| Run | Loss | Top-K | Notes |
|---|---|---|---|
| 1a | IS-only (topk=0) | — | Reproduce existing baseline |
| 1b | GRPO | — | Standard RL comparison |
| 1c | 0.5 CE + 0.5 IS | K=20 | New combined loss |

Settings: model=Qwen/Qwen3-8B, group_size=8, groups_per_batch=32, lr=1e-5,
max_tokens=8192, temperature=1.0. Eval every 5 steps on Chemistry test split.

### Phase 2: Other domains (if Phase 1 looks good)
Run Phase 1's best config on Physics, Biology, Materials.

### Phase 3 (stretch): LiveCodeBench with feedback
env=code_rl, include_environment_feedback=True, topk=20, lr=1e-6.
