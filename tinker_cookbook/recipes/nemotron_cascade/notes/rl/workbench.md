# Workbench Tool-Calling RL Environment Analysis

## Status: PARTIALLY FIXED — Filtered to single-call, tool schemas injected, model makes tool calls but reward still 0 in 3-step test

## Changes Made

### 1. Single-call filter (P0 from original analysis)
Added `ds.filter(lambda x: len(x.get("ground_truth", [])) == 1)` in `WorkbenchRLDataset.__init__`.
Result: 3,248 of 4,686 workbench tasks (69%) are single-call. These are tasks where the
correct answer is a single tool call with arguments derived from the user message alone.

### 2. Tool schema injection (CRITICAL BUG FIX)
The original code copied raw messages from the dataset as `initial_messages` but never told
the model what tools were available. The model had no tool definitions in its prompt and
responded with plain text instead of tool calls.

Fixed by using `renderer.create_conversation_prefix_with_tools()` to inject XML tool schemas
into the system message, matching the pattern used by harbor_env and swe_agentic_env.

### 3. max_turns=3 (adjusted from original max_turns=10)
With max_turns=1, the model's first response is its only chance. But even for single-call
ground-truth tasks, the model often wants to gather info first (e.g., for "cancel meeting
with X", it calls `calendar_get_events` before `calendar_delete_event`). With max_turns=3,
the model can do a short info-gathering step, then the action call. The reward function
already checks ALL tool calls in the full conversation.

## Test Results (3 steps, Nano checkpoint, 4 groups x 4 rollouts)

### v1: Before tool schema fix (max_turns=10, max_tokens=4096)
- Model generated plain text (no tool calls at all)
- reward/total = -0.025 (context overflow penalty)
- correct = 0.0

### v2: After tool schema fix (max_turns=1, max_tokens=8192)
- Model now makes tool calls (tool schemas visible in prompt)
- But calls info-gathering tools (calendar_get_events) not action tools (calendar_delete_event)
- reward/total = 0.0, correct = 0.0
- max_turns hit on every episode (1 turn used, episode ends)

### v3: max_turns=3, max_tokens=16384
- Model does multi-step: calls get_events, gets mock response, tries to reason about it
- But mock data doesn't match what ground truth expects for the action arguments
- Still correct = 0.0 in the 4 tasks sampled
- Much slower (~800s per step vs ~400s)

## Why Reward Is Still 0

Even with all fixes, the 3-step test shows 0 reward. Root causes:

1. **Model prefers multi-step reasoning**: For tasks like "cancel meeting with X", the model
   calls `calendar_get_events` first, then tries to use the result to call `calendar_delete_event`.
   The ground truth expects a direct `calendar_delete_event(event_id="EVT_REAL_123")` with
   a specific event_id that the model can't know.

2. **Small test size**: Only 4 tasks per step (due to groups_per_batch=4). With 3,248 tasks
   and many categories (calendar, email, CRM, analytics, project management), we may simply
   not have sampled an easy-enough task where the model directly calls the correct tool.

3. **Argument exactness**: Ground truth requires exact string matches on argument values.
   Even if the model calls the right tool, slight formatting differences (e.g., name casing,
   date format) cause a mismatch.

## Remaining Improvements

### P1: Partial credit for tool name match (high impact)
Many trajectories call the correct tool but with wrong arguments. Adding partial credit
for name-only matches would provide training signal:
```python
name_reward = matched_names / len(ground_truth)  # 0-1
args_reward = matched_name_and_args / len(ground_truth)  # 0-1
reward = 0.5 * name_reward + 0.5 * args_reward
```

### P2: Larger test batch
Run with groups_per_batch=32+ to sample more diverse tasks. Some task categories
(email_send_email, project_management_create_task) should be easier for direct matching.

### P3: Mock backend seeded from ground truth
For single-call tasks, parse the ground truth to extract expected IDs and seed the mock
returns. E.g., if ground truth expects `calendar_delete_event(event_id="EVT123")`, the
mock `calendar_get_events` should return events with `event_id="EVT123"`.

## Data Distribution
- Total workbench tasks with ground_truth: 4,686
- Single-call (len(ground_truth)==1): 3,248 (69%)
- Multi-call (len(ground_truth)>1): 1,438 (31%)
- Categories observed in test: workbench_calendar, workbench_email, workbench_analytics
