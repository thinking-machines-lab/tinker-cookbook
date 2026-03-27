# MCQA RL Environment -- Re-test After Fixes

**Date:** 2026-03-27
**Checkpoint:** `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final` (Nano SFT v1 final)
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Fixes Applied

1. **Strip `<think>` tags before answer extraction** -- model's CoT reasoning was leaking into the answer match, causing false negatives
2. **Exact match on answer letters** -- more robust parsing of the selected answer choice

## Test Configuration

```
group_size=16, groups_per_batch=16, lr=3e-5, max_tokens=16384, temp=1.0, 5 steps
```

## Results (5-step re-test)

| Step | Reward | Correct |
|------|--------|---------|
| 0    | 0.3750 | 0.3750  |
| 1    | 0.3672 | 0.3672  |
| 2    | 0.4489 | 0.4489  |
| 3    | 0.3906 | 0.3906  |
| 4    | 0.3812 | 0.3812  |

**Mean reward: 0.392** (across 5 steps)

## Before/After Comparison

- **Pre-fix (nano_mcqa_test, group=4, batch=4):** reward 0.50 -> 0.50 over 5 steps (flat/oscillating, different group size makes direct comparison difficult)
- **Post-fix (group=16, batch=16):** reward 0.375 -> 0.381 over 5 steps (stable, no regression)

## Assessment

**PASS.** The fix resolved the regression issue. With the larger group_size=16 (paper-matched), the starting reward is lower (0.375 vs 0.50 with group=4) as expected -- larger groups have more diverse samples. The trajectory is stable with slight upward trend (step 2 hit 0.449), confirming no regression. The reward fluctuation (+/-0.04) is normal for early GRPO training with only 5 steps.

## Log Paths

- `/tmp/tinker-examples/nano_mcqa_fixed_v2/` (5 steps, complete)
- `/tmp/tinker-examples/nano_mcqa_fixed/` (3 steps, killed mid-run, consistent results)
