# Overall Status (2026-03-27)

## Completed
- Recipe: 17 Python files, 28 commits
- SFT v1 on Nano (163K examples, rank 32, 16K tokens) — NLL 0.488
- SFT on gpt-oss-120b — benchmark: GSM8K 94%, IFEval 58% strict
- 9 RL environments built and tested on Nano
- Benchmark eval pipeline (GSM8K, IFEval, MMLU-Pro, MATH-500)
- Wandb integration

## Running
- IF-RL lr=3e-5 paper-matched sweep: 4 steps, reward 0.705→0.787 (+0.082)
- IF-RL lr=1e-5: 3 steps, reward 0.734→0.762 (+0.028)
- MCQA lr=1e-5 and 3e-5: 2 steps each

## Blocked
- SFT v2 (full data): data download incomplete (chat, several others)
- RL cascade: waiting for optimal LR findings + SFT v2

## Key Findings
1. LoRA LR = 10x paper's full-FT LR (SFT: 5e-4, RL: 3e-5)
2. Paper-matched group=16, 49K tokens shows clear RL learning
3. IF-RL lr=3e-5 is the best RL config so far
4. Math SFT data is 229GB for 5.2M examples (very long reasoning chains)

## Checkpoints
- Nano SFT v1 final: tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final
- gpt-oss SFT final: tinker://a27528b8-b83f-59a0-9334-78129ec565d0:train:0/weights/final
- gpt-oss IF-RL Run 1: tinker://6c99cabe-10a4-5e30-a32e-7ffdc39d896a:train:0/weights/final
- gpt-oss IF-RL Run 2: tinker://ec4ef250-02aa-5f91-b141-fe7c6f8f7d95:train:0/weights/final
