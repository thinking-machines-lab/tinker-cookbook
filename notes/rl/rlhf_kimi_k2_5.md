# RLHF GenRM: Switch from Qwen3.5-397B to Kimi K2.5

## Summary

Replaced Qwen3.5-397B-A17B with Kimi K2.5 (`moonshotai/Kimi-K2.5`) as the default GenRM for the RLHF environment. This fixes the major issue where Qwen3.5's thinking mode consumed all GenRM tokens before producing a VERDICT line.

## Kimi K2.5 Model Details

- **Model name in Tinker**: `moonshotai/Kimi-K2.5`
- **Type**: Reasoning + Vision (MoE, Large)
- **Renderer**: `kimi_k25_disable_thinking` (uses `<think></think>` prefill to skip chain-of-thought)
- **Thinking behavior**: Kimi K2.5 uses `<think>` tags, same pattern as Qwen3.5. The `kimi_k25_disable_thinking` renderer suppresses thinking by prefilling `<think></think>`, which causes the model to output its response directly.
- **Why disable thinking for GenRM**: The GenRM prompt already asks for reasoning before a VERDICT line. Enabling thinking would add an extra layer of chain-of-thought before the actual response, wasting tokens and sometimes preventing the VERDICT from appearing (the exact problem with Qwen3.5).

## Test Results (2-step run, 2026-03-27)

Checkpoint: `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final`
Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
Config: `groups_per_batch=2, group_size=4, max_tokens=512, genrm_max_tokens=4096`

### Step 0
- `env/all/reward/total = 0.0006`
- `env/all/win_rate = 0.500`
- `env/all/response_length = 1592.1`
- `env/all/format_valid = 0.0` (expected: max_tokens=512 truncates responses)
- `time/compute_group_rewards:mean = 147s`
- No "unparseable verdict" warnings in logs

### Step 1
- `env/all/reward/total = 0.073`
- `env/all/win_rate = 0.500`
- `env/all/response_length = 2068.5`
- `env/all/format_valid = 0.125` (1/8 responses had valid format)
- `time/compute_group_rewards:mean = 125s`
- No "unparseable verdict" warnings in logs

### Comparison with Qwen3.5-397B Results

| Metric | Qwen3.5 (old) | Kimi K2.5 (new) |
|--------|---------------|-----------------|
| Verdict parse rate | Mostly unparseable at step 1 | 100% (no warnings) |
| compute_group_rewards time | ~357s | ~125-147s |
| Reward variance | Near-zero (all ties from parse failures) | Non-trivial (wins/losses differentiated) |

Key improvement: Kimi K2.5 with thinking disabled reliably outputs VERDICT lines within the 4096-token GenRM budget. The old Qwen3.5 setup had `genrm_max_tokens=512` in the initial test (later raised to 4096 in constants) but the thinking block consumed tokens before producing a verdict.

## Changes Made

- `rlhf_env.py`: Changed `GENRM_MODEL_NAME` from `Qwen/Qwen3.5-397B-A17B` to `moonshotai/Kimi-K2.5`
- `rlhf_env.py`: Changed `GENRM_RENDERER_NAME` from `qwen3_5` to `kimi_k25_disable_thinking`
- `GENRM_MAX_TOKENS` kept at 4096 (sufficient with thinking disabled)
- GenRM prompt unchanged (works well with Kimi K2.5)
- All GenRM configuration fields already existed as configurable parameters on `RLHFGroupBuilder` and `RLHFDatasetBuilder`

## Remaining Notes

- `format_valid` is low because test uses `max_tokens=512` for the policy model, truncating responses before proper stop tokens. At production lengths (16K+), this should improve.
- The GenRM creates a new `ServiceClient` and `SamplingClient` per group in `_create_genrm()`. This is by design (not stored on the frozen dataclass to maintain pickleability).
- Win rates center around 0.5 as expected for early training steps with random-ish policy outputs.
