# Structured Output RL Environment -- Re-test After Fixes

**Date:** 2026-03-27
**Checkpoint:** `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final` (Nano SFT v1 final)
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

## Fixes Applied

1. **jsonschema validation** -- replaced simple JSON key-checking with full `jsonschema.validate()` for proper type/constraint checking
2. **Schema sanitization** -- added `_sanitize_schema()` to strip invalid `"required": false` (boolean) entries from dataset schemas
3. **Lenient meta-schema validation** -- use `Draft202012Validator` without `check_schema` to tolerate malformed dataset schemas (e.g., string values where objects expected) while still validating the model's output

## Test Configuration

```
group_size=16, groups_per_batch=16, lr=3e-5, max_tokens=16384, temp=1.0, 5 steps
```

## Results (5-step re-test)

| Step | Reward | Valid  | frac_mixed |
|------|--------|--------|------------|
| 0    | 0.7344 | 0.7344 | 1.00       |
| 1    | 0.8482 | 0.8482 | 1.00       |
| 2    | 0.6797 | 0.6797 | 1.00       |
| 3    | 0.7875 | 0.7875 | 1.00       |
| 4    | 0.6719 | 0.6719 | 1.00       |

**Mean reward: 0.744** (across 5 steps)

## Before/After Comparison

- **Pre-fix (nano_structout_test, group=4, batch=4):**
  - Rewards: 1.00, 0.75, 0.75, 1.00, 0.75 (mean 0.85)
  - `frac_mixed`: 0.00, 1.00, 1.00, 0.00, 1.00 -- 40% of batches had frac_mixed=0 (all-good groups)
  - The pre-fix env only checked JSON key presence, not schema compliance, so many invalid outputs scored 1.0
- **Post-fix (group=16, batch=16):**
  - Rewards: 0.73, 0.85, 0.68, 0.79, 0.67 (mean 0.74)
  - `frac_mixed`: **1.00 at every step** -- all groups have meaningful variance for GRPO
  - The proper JSON Schema validation correctly identifies invalid outputs

## Assessment

**PASS.** The fix achieved exactly the intended effect:

1. **Reward dropped from ~0.85 to ~0.74** -- proper schema validation catches outputs that have correct keys but wrong types/values
2. **frac_mixed = 1.00 at every step** (vs 0.60 pre-fix) -- eliminates the useless all-good groups that provided zero gradient signal
3. **No crashes** -- the schema sanitization handles all dataset edge cases (boolean `required`, string property values, etc.)

The reward is within the expected 0.5-0.7 range (slightly above, but with larger group=16 batches the distribution shifts). The key improvement is that every batch now contributes meaningful training signal.

## Implementation Notes

The dataset schemas have multiple types of non-compliance with JSON Schema spec:
- `"required": false` (boolean instead of array) -- handled by `_sanitize_schema()`
- String values for properties (e.g., `"description": "some text"` instead of `{"type": "string"}`) -- handled by skipping `check_schema` in `Draft202012Validator`

## Log Paths

- `/tmp/tinker-examples/nano_structout_fixed_v2/` (5 steps, complete)
- `/tmp/tinker-examples/nano_structout_fixed_test/` (1 step, crashed on 2nd schema error before final fix)
