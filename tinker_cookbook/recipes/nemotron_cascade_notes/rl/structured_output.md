# Structured Output Environment

## Status: Working -- reward 0.67-0.89 on Super base model

## Configuration
- max_tokens: 49K
- Dataset: `nvidia/Nemotron-Cascade-2-RL-data`, multi-domain split

## Super Base Model Results
- Reward: 0.67-0.89, frac_mixed: 1.0 (good GRPO signal)

## Reward Logic
- JSON extraction with multiple strategies (direct parse, fenced blocks, regex)
- Schema validation via `jsonschema` library (added to replace the original minimal validation)
- Validates field types, required fields, nested constraints, enum/pattern

## Key Fix
Original validation was too lenient (only checked top-level required fields). Adding proper `jsonschema.validate()` dropped reward from ~0.9 to 0.67-0.89, creating the variance needed for GRPO gradients.
