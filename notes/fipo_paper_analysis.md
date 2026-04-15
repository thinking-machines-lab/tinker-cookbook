# FIPO Paper Deep Analysis

## Core Problem and Motivation

GRPO computes a single scalar advantage per trajectory (correct/incorrect, normalized within the group) and broadcasts it uniformly to every token. This means a pivotal reasoning step and a trivial filler token receive identical gradient signal. The paper argues this coarse credit assignment creates a performance ceiling — DAPO/GRPO plateaus at ~4,000 token response lengths and ~50% on AIME 2024.

FIPO's solution: use the **signed change in log-probability between the old and current policy** at each future token as a proxy for how much a given token's context influences the model's subsequent behavior.

## Mathematical Formulation

### GRPO Background
- Advantage: `A_hat_i = (R_i - mu) / sigma` — binary correctness normalized within group
- Same scalar applied to every token in the sequence

### DAPO Extensions (inherited by FIPO)
1. **Clip-Higher**: Asymmetric [1-0.2, 1+0.28] instead of symmetric ±0.2. Higher upper bound promotes exploration.
2. **Dynamic Sampling**: Filter groups where all correct or all incorrect (zero gradient). Oversample until mixed.
3. **Token-level loss normalization**: Normalize by total token count across samples, not per-sample averages.
4. **Overlong reward shaping**: Graduated penalty for responses exceeding 16,384 tokens (with 4,096 buffer before 20,480 hard max).

### FIPO: Key Components

**Step 1: Probability Shift (signed)**
```
Δlog p_t = log π_θ(o_t|q,o_{<t}) - log π_old(o_t|q,o_{<t})
```
**Why signed?** Preserves directionality: positive = model reinforcing this token, negative = suppressing. Absolute KL loses this information.

**Step 2: Future-KL Accumulation**
```
FutureKL_t = Σ_{k=t}^{T} M_k · γ^{k-t} · Δlog p_k
```
where γ = 2^{-1/τ} (exponential decay) and M_k is a participation mask.

The exponential decay ensures tokens closest in context have the most influence. The half-life τ controls the "attention span" of the credit assignment.

**Step 3: Participation Mask**
```
M_k = 1 if (Â < 0 and ratio > c) is false, else 0
```
Only filters ratio > c (one direction) — tokens where the model has become much MORE likely (potential collapse). Threshold c = 10.0 (dual-clip threshold from DAPO).

**Step 4: Influence Weights**
```
f_t = clip(exp(FutureKL_t), 1-ε_low, 1+ε_high)
```
For 32B: [1.0, 1.2], for 7B: [0.8, 1.2]

Safety: For negative-advantage tokens where IS ratio > 4.0, clamp influence to [0.8, 1.0].

**Step 5: Final Loss**
Standard PPO-clipped loss with dual-clip, using `Ã_t = Â_t · f_t` as reweighted advantages.
Sequence-level filtering: reject entire sequences with >1 lower-clipped token.

## Key Hyperparameters (32B)

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-32B-Base |
| Training data | DAPO-17K |
| Group size G | 16 |
| Learning rate | 1×10⁻⁶ |
| Weight decay | 0.1 |
| PPO clip | [0.2, 0.28] (asymmetric) |
| Max response | 20,480 tokens |
| Overlong buffer | 4,096 tokens |
| Temperature / top-p | 1.0 / 0.7 |
| Dual-clip c | 10.0 |
| KL penalty | 0.0 |
| Decay τ | 32 (paper) / 128 (code default) |
| Influence clip | [1.0, 1.2] (32B) / [0.8, 1.2] (7B) |
| Safety threshold | 4.0 |
| Mini-batch size | 64 prompts (1024 samples) |

## DAPO-17K Dataset
- 17,000 math competition problems from diverse sources
- Publicly released alongside the DAPO paper
- Binary verification reward (correct/incorrect)

## Results
- Qwen2.5-32B-Base with FIPO: AIME 2024 Pass@1 = 58.0% (peak), 56.0% (converged)
- vs DAPO baseline: ~50.0%
- Response length: extended from ~4,000 to >10,000 tokens
- Surpasses o1-mini (~56.0%) and DeepSeek-R1-Zero-Math-32B (~47.0%)

## Related Work Comparison

### GRPO (DeepSeek-Math)
- Eliminates value model; uses group-centered advantages
- ε=0.2 symmetric, β=0.04 KL, G=64

### DAPO
- 4 modifications to GRPO for long-CoT RL
- Key insight: symmetric clipping causes entropy collapse

### Dr.GRPO
- Token-level credit via per-token KL penalty
- Different approach: adjusts KL penalty per token rather than reweighting advantages

### PRIME
- Uses process reward models for token-level credit
- Requires a separate PRM; FIPO is "value-free" (no additional model)

## Key Insights for Our Implementation
1. The signed shift is crucial — absolute KL loses directional information
2. Participation mask only filters ONE direction (ratio > c) — tokens becoming too likely
3. Sequence-level filtering prevents training on corrupted sequences
4. DAPO's dynamic sampling is important but not implemented in our version (our RL loop already removes constant-reward groups)
5. The decay half-life τ controls the future "attention span" — 32 tokens means each token is influenced mainly by the next ~32 tokens of context
