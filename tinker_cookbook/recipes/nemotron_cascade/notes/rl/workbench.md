# Workbench Tool-Calling RL Environment Analysis

## Status: FIXED -- Ground-truth seeded mocks + partial credit reward. Reward 0.44 on step 0 with Super base model.

## Changes Made

### 1. Tool schema injection (from earlier iteration)
The original code never told the model what tools were available. Fixed by using
`renderer.create_conversation_prefix_with_tools()` to inject XML tool schemas into
the system message, matching the pattern used by harbor_env and swe_agentic_env.

### 2. Ground-truth seeded mock backend (NEW)
Root cause of reward=0: mock tools returned fake IDs (e.g. `email_id: "0000250"`) that
didn't match ground-truth expected arguments (e.g. `email_id: "00000259"`). Even for
"single-call" tasks, the model often does a lookup first (calls `email_search_emails`
before `email_delete_email`), so the fake IDs propagate into the action call.

Fix: `_extract_gt_ids()` parses ground-truth tool calls to extract expected IDs and
values. `WorkbenchTools` constructor accepts these IDs and returns them from lookup
tools. For example, if ground truth expects `email_delete_email(email_id="00000259")`,
then `email_search_emails` returns results containing `email_id: "00000259"`.

### 3. Partial credit reward (NEW)
Changed from binary (exact match or 0) to partial credit:
- 0.0 if model makes no tool calls
- 0.5 if model calls the correct tool name(s) but with wrong arguments
- 1.0 if model calls the correct tool(s) with exact argument matches
- Formula: `reward = 0.5 * name_match_rate + 0.5 * exact_match_rate`

This provides gradient signal even when the model picks the right tool but fumbles
an argument (e.g. wrong date format, missing optional field).

### 4. Re-enabled multi-call tasks
Removed the `len(ground_truth) == 1` filter. Multi-call tasks in the workbench
dataset are mostly parallel calls (e.g. two `analytics_create_plot` calls) where
arguments come directly from the user message. The seeded mock backend handles the
few cases that need lookup. Dataset went from 3,248 to 4,686 examples.

### 5. Added analytics_create_plot tool
The dataset contains many `analytics_create_plot` ground-truth calls but the mock
backend was missing this tool. Added it so the model can actually make these calls.

### 6. max_turns=3 (kept from earlier)
Model can do a short info-gathering step, then the action call. Reward function
checks ALL tool calls in the full conversation.

## Test Results

### v4: Ground-truth seeded mocks + partial credit (Super base, 4 groups x 4 rollouts)
Step 0:
- `name_match`: 0.875 (model picks right tool 87.5% of the time)
- `correct` (exact match): 0.0
- `partial_reward`: 0.4375
- `reward/total`: 0.4375
- `frac_mixed`: 1.0 (all groups have variance -- ideal for RL)
- `turns_per_episode`: 2.75 (model uses multi-step reasoning)
- ~233s per step

This is a massive improvement over the previous 0.0 reward. The model correctly
identifies the tool to call most of the time. The remaining gap between name_match
(0.875) and exact_match (0.0) is the argument accuracy problem -- the model calls
the right tool but with slightly wrong arguments.

### Previous results for comparison
- v1 (no tool schemas): reward = -0.025, correct = 0.0 (plain text, no tool calls)
- v2 (schemas, max_turns=1): reward = 0.0, correct = 0.0 (info-gathering only)
- v3 (schemas, max_turns=3): reward = 0.0, correct = 0.0 (fake IDs don't match)

## Data Distribution
- Total workbench tasks with ground_truth: 4,686
- Single-call (len(ground_truth)==1): 3,248 (69%)
- Multi-call (len(ground_truth)>1): 1,438 (31%)
- Multi-call tasks are mostly parallel calls to the same tool (e.g. two analytics_create_plot)
- Categories: workbench_calendar, workbench_email, workbench_analytics, workbench_project_management, workbench_crm

## Remaining Improvements

### P1: Argument fuzzy matching
Many near-misses: model calls correct tool with almost-correct arguments but gets 0
on exact match. Consider fuzzy matching for dates (2023-12-01 vs 12/1/2023), names
(case insensitive), and optional fields.

### P2: Larger batch for production runs
Test used 4 groups x 4 rollouts. For real training, use paper hyperparams:
128 groups x 16 rollouts per step.

### P3: Multi-call task accuracy
The parallel multi-call tasks (analytics plots) should work well since all args come
from the user message. True chained tasks (if any exist) may still struggle.
