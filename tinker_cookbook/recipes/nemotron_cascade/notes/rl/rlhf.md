# RLHF Environment Analysis

## Status: FIXED -- Rewards now logged successfully

Two bugs prevented rewards from being logged. Both fixed 2026-03-27.

## Root Causes

### Bug 1: Wrong API parameter for GenRM sampling client (fatal)

`_create_genrm()` in `RLHFGroupBuilder` used `model_path=` instead of `base_model=` when creating the GenRM sampling client:

```python
# BEFORE (broken): model_path expects a tinker:// checkpoint URL
service_client.create_sampling_client(model_path=self.genrm_model_name)

# AFTER (fixed): base_model is for model names like "Qwen/Qwen3.5-397B-A17B"
service_client.create_sampling_client(base_model=self.genrm_model_name)
```

This raised `ValueError: model_path must start with 'tinker://'` inside `compute_group_rewards`, causing every group rollout to fail. The FailFast strategy re-raised the error, but `asyncio.gather` in `gather_with_progress` propagated it... which resulted in "all groups failed or filtered, skipping batch" and no training or reward logging.

### Bug 2: Wrong dataset field name (fatal)

`_make_env_group_builder()` looked for a `"prompt"` field in HelpSteer3 rows, but HelpSteer3 stores conversations in a `"context"` field (a list of message dicts). Every row returned `None`, so `get_batch()` always produced an empty list of builders.

```python
# BEFORE (broken): HelpSteer3 has no "prompt" field
prompt_text = row.get("prompt", "")

# AFTER (fixed): use "context" field, strip trailing assistant turns
context = row.get("context")
```

The fix parses the conversation context, strips trailing assistant messages so the prompt ends on the last user turn, and extracts plain text for the GenRM.

## Reward Architecture (Unique Among All 9 Envs)

RLHF is the **only** env that uses `compute_group_rewards` exclusively for reward computation. All other envs compute reward in `Env.step()`.

- `RLHFEnv.step()` always returns `reward=0.0`
- `RLHFGroupBuilder.compute_group_rewards()` runs pairwise GenRM comparisons
- Total reward = sum of per-step rewards (0) + group reward from `compute_group_rewards`

## Verified Test Results (2-step run, 2026-03-27)

Checkpoint: `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final`

**Step 0:**
- `env/all/reward/total = 0.0076`
- `env/all/win_rate = 0.500`
- `env/all/response_length = 393.75`
- `time/compute_group_rewards = 357s` (GenRM is slow -- 6 pairwise comparisons)

**Step 1:**
- `env/all/reward/total = 0.0`
- `env/all/win_rate = 0.500`
- `env/all/response_length = 1346.5`
- Most GenRM verdicts unparseable (thinking too long for 512-token limit)

## Known Issues (non-blocking)

### GenRM verdicts often unparseable
Qwen3.5-397B-A17B is a thinking model. With `genrm_max_tokens=512`, the `<think>` block often consumes all tokens before the model outputs the `VERDICT:` line. Mitigation options:
- Increase `genrm_max_tokens` to 2048+ (slower but more reliable)
- Use `qwen3_5_disable_thinking` renderer for the GenRM (bypasses thinking)
- Add a system prompt instruction like "Be concise in your reasoning"

### format_valid = 0.0
All responses showed `format_valid=0.0` because the test used `max_tokens=256`, which truncates responses before proper ending tokens. At production lengths (16K+), this should improve.

### Metric aggregation path (confirmed working)
`compute_group_rewards` returns `list[tuple[float, Metrics]]` where Metrics includes `win_rate`, `response_length`, `format_valid`. These flow through:
`TrajectoryGroup.metrics_G` -> `_compute_trajectory_metrics` (via `dict_mean`) -> `compute_trajectory_metrics` (prefixed with `env/all/`) -> `metrics.jsonl`
