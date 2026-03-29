# IF-RL Environment

## Status: Working -- best performing env

## Configuration
- LR: 3e-5 (LoRA, 10x paper's 3e-6 full FT)
- group_size: 16, batch: 32 (paper=128, validated at 128)
- max_tokens: 49K, temperature: 1.0, KL: 0
- Loss: importance_sampling (GRPO), dynamic filtering: on

## Super Base Model Results
- Reward: 0.77-1.0, frac_mixed: 1.0 at group=16x32
- At batch=128: reward=0.714, 100% mixed groups, ~42 min/step
- ETA for 180 steps at batch=128: ~5 days

## Reward Logic
- 48 IFEval instruction types verified programmatically
- Reward = fraction of instructions satisfied (0.0 to 1.0)
- Overlong penalty: reward=0 if stop_reason="length"
- Dataset: `nvidia/Nemotron-Cascade-2-RL-data`, IF-RL split

## Known Limitations
- Verifier does "loose" matching only (case-insensitive)
- `detectable_format:constrained_response` and `count:counting_composition` always return True (~2-5% false positive rate)
- `language:response_language` falls back to True if `langdetect` not installed
