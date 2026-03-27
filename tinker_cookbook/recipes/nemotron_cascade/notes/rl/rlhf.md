# RLHF Environment Analysis

## Status: RUNS BUT NO REWARD LOGGED — Debugging needed

## Reward Architecture (Unique Among All 9 Envs)

RLHF is the **only** env that uses `compute_group_rewards` exclusively for reward computation. All other envs compute reward in `Env.step()`. Here's how it works:

### Step-level: `RLHFEnv.step()` returns reward=0.0 always
```python
async def step(self, action, *, extra=None):
    return StepResult(reward=0.0, episode_done=True, ...)
```

### Group-level: `RLHFGroupBuilder.compute_group_rewards()` does the real work
1. Parses responses from all trajectories
2. Runs pairwise GenRM comparisons (Qwen3.5-397B-A17B)
3. Aggregates win/loss into per-trajectory win rates
4. Applies length-normalized reward with conciseness bonus
5. Returns `list[tuple[float, Metrics]]`

### How total reward is computed (in `TrajectoryGroup.get_total_rewards`):
```python
total = sum(transition.reward for transition in trajectory.transitions) + final_reward
```
Since per-step reward is always 0.0, total reward = the `compute_group_rewards` value.

## Why Reward Might Not Be Logged

### Hypothesis 1: GenRM call fails silently
The `compute_group_rewards` method creates a new `TinkerMessageCompleter` for the GenRM. If:
- The Qwen3.5-397B-A17B model is not available on the Tinker service
- The sampling client creation fails
- The GenRM response doesn't parse ("unparseable verdict" → returns 0.0)

Then all rewards would be 0.0 or close to it. The `logger.warning` for unparseable verdicts goes to stderr, not to the training metrics.

### Hypothesis 2: Reward IS computed but not visible in expected location
The `compute_group_rewards` rewards flow through `TrajectoryGroup.final_rewards_G` → `get_total_rewards()` → advantage computation → training. Metrics from `compute_group_rewards` (like `win_rate`, `response_length`, `format_valid`) are stored in `TrajectoryGroup.metrics_G`.

Check: Are these metrics being picked up by `compute_trajectory_metrics` in `metric_util.py`? The function aggregates both per-step metrics and group-level metrics. Look for `win_rate` in the logged metrics.

### Hypothesis 3: matchup_group_size=4 limits comparisons
With group_size=16 and matchup_group_size=4, comparisons are chunked: only within chunks of 4. This means 6 comparisons per chunk, 4 chunks = 24 total comparisons (vs 120 for full pairwise). Win rates are more noisy but should still be non-zero.

## Actionable Improvements

### P0: Verify GenRM is accessible
Before training, test that:
```python
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(model_path="Qwen/Qwen3.5-397B-A17B")
```
works. If this model isn't available, use a different judge model.

### P1: Add reward to per-step metrics for visibility
Even though reward is computed at group level, log a diagnostic in the step function:
```python
metrics={"rlhf_step": 1.0}  # just to confirm steps are happening
```

### P2: Check metric aggregation
Verify that `compute_trajectory_metrics` includes the `Metrics` dict returned by `compute_group_rewards` (the `win_rate`, `response_length`, `format_valid` keys). These should appear in `metrics.jsonl`.

### P3: Add fallback for GenRM failures
If the GenRM call fails, fall back to a simpler reward (e.g., response length penalty or format check) rather than silent 0.0.

### P4: Lower matchup_group_size or increase it
matchup_group_size=4 means small tournament brackets. For 16 rollouts, try matchup_group_size=8 or 16 for less noisy win rates.

## Expected Impact
P0 is the most likely fix — if GenRM is not accessible, no reward can be computed. Once GenRM works, RLHF should produce rewards in the 0.3-0.7 range (reflecting preference ranking).
