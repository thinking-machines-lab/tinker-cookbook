# Workbench Tool-Calling RL Environment Analysis

## Status: REWARD=0 — Mock backend returns fake data, model can't match ground truth args

## Core Problem: Mock Backend Defeats Multi-Step Verification

### How the reward works
```python
def check_tool_calls_in_messages(messages, ground_truth):
    # For each ground_truth call, find a model call with same name AND same arguments
    for gt in ground_truth:
        for mc in model_calls:
            if mc["name"] == gt_name:
                args_match = all(str(mc["arguments"].get(k)) == str(v) for k, v in gt_args.items())
```
Reward = fraction of ground truth calls matched (name + exact argument values).

### Why this fails with mock backend
Consider a 2-step task: "Find John's email, then send him a meeting invite."

Ground truth:
1. `company_directory_find_email_address(name="John Smith")` → should return real email
2. `calendar_create_event(attendees="john.smith@company.com", ...)`

What happens:
1. Model calls `company_directory_find_email_address(name="John Smith")` ✓ name matches
2. Mock returns `{"email": "john.smith@company.com"}` — happens to match ground truth pattern
3. Model uses mock email in next call → MAY match ground truth

**But** for many tasks, the mock data doesn't match ground truth:
- Mock always returns `event_id="EVT001"` but ground truth might reference `event_id="EVT_REAL_123"`
- Mock email search returns generic results, ground truth expects specific `email_id` values
- CRM mock returns `customer_id="CUST001"`, ground truth expects actual IDs

### What fraction of tasks are single-call (verifiable without backend)?

Looking at the tool implementations and ground truth structure:
- **Single-call tasks**: Only need name + args match, no dependency on return values. Examples:
  - "Send an email to X with subject Y" → just needs `email_send_email(to=X, subject=Y)`
  - "Create a task called X" → just needs `project_management_create_task(title=X)`
- **Multi-step tasks**: Need return values from step N to construct step N+1's arguments. Examples:
  - "Find who sent the latest email and add them to the project" (2+ steps, chained)
  - "Get John's email and send him the report" (2 steps, chained)

**Estimated split**: Without examining actual data distribution, tool-calling benchmarks typically have 40-60% multi-step tasks. Even single-call tasks may fail if the ground truth expects specific argument formatting.

## Actionable Improvements

### P0: Filter to single-call tasks only
Add a filter in `WorkbenchRLDataset.__init__`:
```python
ds = ds.filter(lambda x: len(x.get("ground_truth", [])) == 1)
```
Single-call tasks only need correct tool name + arguments, which the model can get right without backend data. This should immediately give non-zero reward.

### P1: Relax argument matching for multi-step tasks
For chained tasks, score tool name correctness separately from argument correctness:
```python
name_reward = matched_names / len(ground_truth)  # 0-1
args_reward = matched_name_and_args / len(ground_truth)  # 0-1
reward = 0.5 * name_reward + 0.5 * args_reward
```
This gives partial credit for calling the right sequence of tools even with wrong arguments.

### P2: Make mock backend return consistent fake data
Instead of generic mock responses, seed the mock with data derived from the ground truth:
- Pre-populate email IDs, customer IDs, event IDs that the ground truth expects
- This requires parsing the ground truth to extract expected return values and configuring mocks accordingly

### P3: Grade on tool call SEQUENCE rather than exact args
Many tool-calling benchmarks grade on:
1. Correct tool sequence (name order)
2. Correct argument types (not exact values)
3. Correct argument values (exact match)

Layer these as partial credit: 0.33 for right sequence, 0.33 for right types, 0.34 for exact values.

## Expected Impact
P0 (single-call filter) should immediately give reward > 0 for a subset of tasks. Combined with P1 (partial credit), the env should produce meaningful training signal for tool-calling capabilities.
