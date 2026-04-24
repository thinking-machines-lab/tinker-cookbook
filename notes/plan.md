# FIPO: Future-KL Influenced Policy Optimization

## Research Question
Can we reproduce the core FIPO result — token-level advantage reweighting via future-KL divergence improves RL training for math reasoning — using the Tinker API?

## Paper Summary
FIPO (arXiv:2603.19835) modifies GRPO/DAPO by replacing uniform trajectory-level advantages with token-level reweighted advantages. For each token t, it computes:

1. **Log-prob shift**: Δlog p_t = log π_θ(o_t) - log π_old(o_t)
2. **Participation mask**: M_t = 1 if (Â < 0 and ratio > c) is false, else 0 (dual-clip filter)
3. **Future-KL**: FutureKL_t = Σ_{k≥t} M_k · γ^(k-t) · Δlog p_k, where γ = 2^(-1/τ)
4. **Influence weight**: f_t = clip(exp(FutureKL_t), 1-ε_low, 1+ε_high)
5. **Weighted advantage**: Ã_t = Â_t · f_t
6. **Loss**: Standard clipped PPO/GRPO loss with Ã_t instead of Â_t

Key hyperparameters from the paper (32B):
- τ (half-life): 32 tokens (paper), 128 (code default)
- ε_low, ε_high: [1.0, 1.2] (32B), [0.8, 1.2] (7B)
- Safety threshold: 4.0 (cap influence weights for negative high-IS-ratio samples)
- Dual-clip c: 10.0 (filter threshold for participation mask)
- chunk_size: 128 (for efficient computation)

## Hypothesis
Adding future-KL reweighting to token-level advantages will improve math reasoning accuracy over vanilla GRPO, particularly by enabling longer reasoning chains.

## Experiment Design

### Experiment 1: Sanity Check
- **Model**: Small model (Llama-3.2-1B or Qwen-3-0.6B)
- **Dataset**: arithmetic or gsm8k (small)
- **Steps**: 5-10
- **Purpose**: Verify the pipeline runs end-to-end, custom loss computes correctly

### Experiment 2: FIPO vs GRPO Baseline
- **Model**: Qwen-3-4B or Llama-3.1-8B-Instruct
- **Dataset**: math (competition math)
- **Steps**: 50-100
- **Group size**: 8-16
- **Compare**: vanilla GRPO (importance_sampling) vs FIPO (custom loss)
- **Metrics**: reward/mean, reward accuracy, response length, KL divergence

## Implementation Strategy
FIPO requires access to per-token training logprobs during loss computation, which is only available inside `forward_backward_custom`. We'll:
1. Create `tinker_cookbook/rl/fipo.py` with the future-KL computation
2. Create a custom loss function compatible with `forward_backward_custom`
3. Modify the RL training loop to use the custom loss when FIPO is enabled
4. Create a recipe at `tinker_cookbook/recipes/fipo/`

## Success Criteria
- Pipeline runs without errors
- FIPO shows improvement over vanilla GRPO on math accuracy
- Longer average response lengths with FIPO (indicating deeper reasoning)
