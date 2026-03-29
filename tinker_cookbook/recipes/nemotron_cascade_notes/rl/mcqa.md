# MCQA Environment

## Status: Fixed -- reward 1.0 on Super base model

## Configuration
- max_tokens: 8K (49K causes overlong with base model; paper uses 49K)
- Dataset: `nvidia/Nemotron-Cascade-2-RL-data`, multi-domain split

## Super Base Model Results
- Reward: 1.0, frac_mixed: 0.0 (all correct = no GRPO signal at small scale)
- At scale with diverse questions, mixed groups should emerge

## Fixes Applied
1. **Expanded answer extraction**: Added patterns for `Option Selected: X`, `<final_answer>X</final_answer>`, `((X))`, `*X*`, and more
2. **Overlong partial credit**: When stop_reason="length", search thinking content for answer, award 0.5 for correct-but-overlong
3. **Concise system prompt**: Optional (set `system_prompt=None` to disable); not in original paper

## Key Decisions
- Answer extraction searches after `</think>` first, falls back to full text
- `check_answer` uses exact match (removed old containment check that caused false positives)
- Think-content fallback (`include_think=True`) for truncated responses
