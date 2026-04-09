# Experiment: FIPO vs GRPO on Qwen3-30B-A3B-Base + DAPO-17K

## Setup
- **Model**: Qwen/Qwen3-30B-A3B-Base (MoE, 3B active params)
- **Training data**: DAPO-17K (17K competition math from AoPS)
- **Eval**: AIME 2025 (30 problems) — standalone baseline only; in-training eval broken due to parse failures on base model output
- **Hyperparameters**: G=16, 32 groups/batch, 8 substeps, lr=1e-6, max_tokens=20480
- **FIPO config**: τ=32, influence_clip=[0.8, 1.2], PPO clip=[0.2, 0.28], dual_clip_c=10.0
- **Steps**: 50 each
- **Commit**: ca8442c (research/fipo branch)

## Results

| | FIPO | GRPO |
|---|---|---|
| Train correct (overall) | **18.2%** | 13.5% |
| Train correct (1st half) | 16.1% | 11.5% |
| Train correct (2nd half) | **20.2%** | 15.5% |
| Best single step | **28.3%** | 21.1% |
| Tokens/ep | 1089 | 1196 |
| KL | 0.00111 | 0.00106 |
| AIME 2025 (baseline, pre-training) | 3.3% (1/30) | 3.3% (1/30) |

## Observations

1. **FIPO shows higher train accuracy** throughout (18.2% vs 13.5% overall), with the gap maintained across the run. However, train accuracy is on different random batches (same seed but filtering diverges), so this is suggestive but not conclusive.

2. **Both methods show learning** — second half accuracy is higher than first half for both (FIPO: +4.1pp, GRPO: +4.0pp).

3. **No response length growth observed** — tokens/ep stayed flat at ~1000-1200 for both. The paper's signature finding (4K→10K+ tokens) was not reproduced. Likely because 50 steps is insufficient and the 30B-A3B-Base (3B active) is much smaller than the paper's 32B dense model.

4. **FIPO influence weights active** — std=0.215, indicating meaningful token-level reweighting. Future-KL abs mean=0.162.

5. **FIPO runs ~2x slower** than GRPO due to `forward_backward_custom` requiring 2 forward passes per substep.

6. **AIME eval broken for base models** — the existing benchmark uses `EnvFromMessageEnv` which fails to parse base model output (no `<|im_end|>` markers). Standalone eval with custom `grade_fn` confirmed 3.3% baseline.

## Limitations

- No shared held-out eval that works during training
- 50 steps is likely insufficient (paper ran much longer on 32B)
- 30B-A3B-Base has only 3B active params vs paper's 32B dense
- Train accuracy comparison is confounded by different random batches
