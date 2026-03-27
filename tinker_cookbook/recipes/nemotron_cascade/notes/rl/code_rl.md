# Code RL Environment Analysis

## Status: FIXED — Reward improved from 0.094 to 0.500 (5.3x)

Root cause was a critical bug: missing newline in assertion execution scripts caused
SyntaxError on every execution. See `notes/rl/code_rl_deep_dive.md` for full analysis.

## Architecture
- Default dataset: MBPP (sanitized) from HuggingFace (257 problems)
- Paper uses: AtCoder/Codeforces competitive programming (3.5K hard problems, not publicly released)
- Execution: Modal sandbox with bash script
- Reward: Strict binary (default) or partial credit (fraction of tests passed)

## Fixes Applied

### P0: Missing Newline Bug (CRITICAL)
The `build_execution_script` function base64-encodes stripped code (no trailing newline).
When `cat solution.py tests.py > run.py` concatenates them, the last line of the solution
merges with the first test assertion, causing SyntaxError 100% of the time.
**Fix**: Ensure code ends with `\n` before encoding.

### P1: Code Extraction Improvements
- Strip `<think>...</think>` blocks before extraction (reasoning model traces)
- Use last match instead of first for fenced blocks (handles draft code in reasoning)
- Added unfenced code fallback for responses without fences

### P2: Partial Credit Support
- Each assertion now runs in its own try/except block
- Script reports `RESULT: passed/total` for parsing
- `partial_credit=True` mode available on `CodeRLEnv` and `CodeRLDatasetBuilder`

### P3: Error Visibility
- Removed `2>/dev/null` that was hiding all error messages
- Failed assertions report error details to stderr

## Test Results

| Metric | Before | After |
|---|---|---|
| Reward | 0.094 | 0.500 |
| Code extraction rate | Unknown | 75-100% |
| Groups mixed | Low | 100% |
| Error visibility | Hidden by 2>/dev/null | Full stderr captured |

## Remaining Items

### Not yet fixed
- `test_imports` not included (affects 13/257 = 5% of MBPP problems)
- No competitive programming dataset available (paper's data not released)
- Default max_tokens (49K) may be low for competitive programming; paper uses 118K

### Future considerations
- LiveCodeBench as a closer proxy to the paper's competitive programming data
- Increase group_size for harder datasets where pass rate is lower
