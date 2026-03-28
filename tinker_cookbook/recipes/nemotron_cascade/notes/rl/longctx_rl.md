# Long-Context RL Environment

## Status: Working -- reward 0.60-0.65 on Super base model

## Configuration
- max_tokens: 49K
- Dataset: NarrativeQA (test split)
- Judge: Qwen3.5-397B-A17B, scores 0-10 normalized to [0,1]

## Super Base Model Results
- Reward: 0.60-0.65, frac_mixed: 1.0
- ~1350s/step at group=16x32

## Key Fixes Applied
1. **Judge max_tokens**: Increased from 32 to 512. Qwen3.5 is a thinking model and needs space for `<think>` reasoning before outputting the score. At 32 tokens, scores were truncated to 0.
2. **Score parsing**: Changed to extract the last integer in the response (not first), to skip numbers in the judge's reasoning.

## Known Limitations
- Judge context truncated to 12K chars (NarrativeQA docs can be much longer)
- Uses `document.summary` when available (shorter, may not test long-context ability)
- Reference answers are short (1-2 sentences); model's longer answers may be unfairly scored
