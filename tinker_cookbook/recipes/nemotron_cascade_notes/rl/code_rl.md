# Code RL Environment

## Status: Working -- reward 0.75-1.0 on Super base model

## Configuration
- max_tokens: 118K (paper-matched)
- Dataset: MBPP sanitized (257 problems); paper uses unreleased AtCoder/Codeforces (3.5K)
- Execution: Modal sandbox with bash script

## Super Base Model Results
- Reward: 0.75-1.0, frac_mixed: 1.0 (step 1)
- ~620s/step at group=4x2

## Fixes Applied
1. **Missing newline bug (critical)**: `build_execution_script` omitted trailing newline, causing SyntaxError when solution.py and tests.py were concatenated. Fixed by ensuring `\n` suffix.
2. **Code extraction**: Strip `<think>` blocks, use last fenced block match, added unfenced fallback
3. **Partial credit**: Each assertion runs in own try/except, reports `RESULT: passed/total`
4. **Error visibility**: Removed `2>/dev/null` that hid all errors

## Known Limitations
- `test_imports` not included (affects 5% of MBPP problems)
- No competitive programming dataset available (paper's data unreleased)
- MBPP is relatively easy for Super base model
