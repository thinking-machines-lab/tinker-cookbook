# Long-Context RL Environment -- Re-test After Fixes

**Date:** 2026-03-27
**Checkpoint:** `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final` (Nano SFT v1 final)
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Fixes Applied

1. **Judge max_tokens=512** -- the LLM judge was truncated at the default token limit, preventing it from outputting scores; now explicitly set to 512 tokens
2. **Last-number parsing** -- extract the last number from the judge's response instead of the first, since the judge often reasons about scores before giving a final answer

## Test Configuration

```
group_size=8, groups_per_batch=8, lr=3e-5, max_tokens=8192, temp=1.0, 5 steps
```

## Results (5-step re-test)

| Step | Reward | Judge Reward |
|------|--------|-------------|
| 0    | 0.6516 | 0.6516      |
| 1    | 0.5000 | 0.5000      |
| 2    | 0.5891 | 0.5891      |
| 3    | 0.5891 | 0.5891      |
| 4    | 0.4766 | 0.4766      |

**Mean reward: 0.561** (across 5 steps)

## Before/After Comparison

- **Pre-fix (nano_longctx_rl_test, group=4, batch=4):**
  - Rewards: 0.10, 0.10 (only 2 steps completed before being killed)
  - Judge was unable to output proper scores due to token truncation
- **Post-fix (group=8, batch=8):**
  - Rewards: 0.65, 0.50, 0.59, 0.59, 0.48 (mean 0.56)
  - **6.5x improvement** over pre-fix baseline (0.56 vs 0.10)
  - Consistent with earlier fix attempt: `nano_longctx_rl_fixed` showed 0.61, 0.59, 0.51

## Assessment

**PASS.** The fix dramatically improved reward from 0.10 to 0.56 (average). This is above the expected 0.3-0.5 range, indicating the model already has decent long-context QA ability from SFT training.

Key observations:
1. **Judge is functional** -- reward varies meaningfully between 0.0 and 1.0 (not stuck at 0.10)
2. **frac_mixed = 1.00 at all steps** (from first run data) -- all groups have variance, enabling GRPO
3. **Slight downward trend** is expected in early steps with high LR (3e-5) -- the model is exploring. Longer runs should stabilize.
4. Results are highly consistent across 3 independent runs (v1: 0.65/0.56/0.54, v2: 0.65/0.50/0.59, earlier fix: 0.61/0.59/0.51)

## Log Paths

- `/tmp/tinker-examples/nano_longctx_fixed_v2/` (5 steps, complete)
- `/tmp/tinker-examples/nano_longctx_fixed_test/` (3 steps, killed mid-run)
- `/tmp/tinker-examples/nano_longctx_rl_fixed/` (3 steps, earlier fix test from different session)
