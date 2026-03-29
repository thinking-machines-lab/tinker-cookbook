# RLHF Environment

## Status: Working -- reward 0.50 on Super base model

## Configuration
- max_tokens: 16K
- KL penalty: 0.03 (only env with non-zero KL)
- Dataset: HelpSteer3 (`context` field, not `prompt`)
- GenRM judge: Kimi K2.5 (paper uses Qwen3-235B)

## Super Base Model Results
- Reward: 0.50, frac_mixed: 1.0
- ~760s/step at group=4x2

## Reward Architecture
RLHF is the only env that computes reward in `compute_group_rewards` (not `Env.step()`). `RLHFEnv.step()` always returns reward=0; the GenRM runs pairwise comparisons after all rollouts complete.

## Fixes Applied
1. **Wrong API parameter**: `model_path=` changed to `base_model=` for GenRM sampling client
2. **Wrong dataset field**: HelpSteer3 uses `context` (list of message dicts), not `prompt`

## Known Limitations
- GenRM verdicts often unparseable: Qwen3.5 thinking consumes all tokens before `VERDICT:` line at `genrm_max_tokens=512`. Options: increase to 2048+, use non-thinking renderer, or add concise-reasoning system prompt.
