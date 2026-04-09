# Related Work: GRPO, DAPO, Dr.GRPO, PRIME

## GRPO (DeepSeek-Math, arXiv 2402.03300)

Eliminates the value model from PPO. Advantages computed by normalizing rewards within a group of G outputs per prompt:
```
A_hat_i = (R_i - mean(R)) / std(R)
```
Same advantage broadcast to all tokens. Uses symmetric clip ε=0.2, β=0.04 KL, G=64.

## DAPO (arXiv 2503.14476) — FIPO's direct baseline

Four modifications to GRPO:

1. **Clip-Higher**: [0.2, 0.28] asymmetric clip — higher upper bound prevents entropy collapse
2. **Dynamic Sampling**: Filter groups where all correct or all incorrect
3. **Token-level loss**: Normalize by total token count, not per-sample
4. **Overlong reward shaping**: Graduated penalty for responses >16,384 tokens

Results: DAPO 50% vs GRPO 30% on AIME 2024 (Qwen2.5-32B)

## Dr.GRPO (arXiv 2503.20783)

Corrects two biases in GRPO:
1. Remove per-sequence length normalization (creates length bias)
2. Remove std normalization in advantages (creates difficulty bias)

Does NOT assign different advantages per token — still uniform. Contribution is correcting loss aggregation.

## PRIME (arXiv 2502.01456)

Uses implicit PRM (process reward model) trained from outcome labels to extract token-level rewards. Token rewards = temporal differences of the implicit value function:
```
r_phi(y_t) = v_phi(y_{<t+1}) - v_phi(y_{<t})
```

Requires training a separate reward model. FIPO achieves similar token-level credit without auxiliary models.

## DAPO-17K Dataset

- Source: AoPS website + math competition homepages
- 17K problems, all reformulated to integer answers for reliable verification
- Binary reward: +1 correct, -1 incorrect
- Available at: BytedTsinghua-SIA/DAPO-Math-17k on HuggingFace

## Key Ablation Results from FIPO Paper (7B)

| Setting | AIME 2024 |
|---------|-----------|
| GRPO | 22% |
| DAPO | 36% |
| FIPO [1.0, 1.2] with filtering | 36% |
| FIPO [0.8, 1.2] without filtering | 38% |
| FIPO [0.8, 1.2] with filtering | 40% |

τ (decay half-life): 8→40%, 32→40%, 128→39%, 256→42% — robust to choice.
