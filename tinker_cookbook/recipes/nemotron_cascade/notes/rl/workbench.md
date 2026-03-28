# Workbench Tool-Calling Environment

## Status: Fixed -- reward 0.44 on Super base model

## Configuration
- max_tokens: 49K
- max_turns: 3 (info-gathering + action call)
- Dataset: `nvidia/Nemotron-Cascade-2-RL-data`, workbench split (4,686 examples)
- Categories: calendar, email, analytics, project management, CRM

## Super Base Model Results
- Step 0: name_match=0.875, correct=0.375, partial_reward=0.44, frac_mixed=1.0
- ~240-375s per step

## Fixes Applied
1. **Tool schema injection**: Used `renderer.create_conversation_prefix_with_tools()` to inject XML tool schemas
2. **Ground-truth seeded mocks**: `_extract_gt_ids()` parses ground-truth calls to seed mock tools with expected IDs (solves fake-ID mismatch)
3. **Partial credit reward**: `reward = 0.5 * name_match_rate + 0.5 * exact_match_rate` (was binary)
4. **Re-enabled multi-call tasks**: Removed single-call filter (3,248 -> 4,686 examples)
5. **Added analytics_create_plot tool**: Missing from mock backend

## Known Limitations
- Argument matching is strict (no fuzzy matching for dates, case-insensitive names)
- Multi-call chained tasks (where one call depends on another's output) may still struggle
