# Overall Status

## Model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144` -- MoE (12B active), LoRA rank 64, 262K context.

See `model_decision.md` for rationale.

## SFT Progress

- Data: All 8 subsets downloaded (24.5M examples, 553GB)
- Config: batch=2048, lr=3e-4 cosine, rank=64, 49K tokens
- Training: step 169/33K, NLL 0.687 -> 0.547 (20% improvement), ~5.5 min/step

## RL Environment Status (Super base model, no SFT checkpoint)

| Env | Reward | frac_mixed | max_tokens | Status |
|-----|--------|-----------|------------|--------|
| IF-RL | 0.77-1.0 | 1.0 | 49K | Working |
| MCQA | 1.0 | 0.0 (too easy) | 8K | Fixed (expanded extraction + overlong partial credit) |
| Structured Output | 0.67-0.89 | 1.0 | 49K | Working |
| Code RL | 0.75-1.0 | 1.0 | 118K | Working |
| Long-Context | 0.60-0.65 | 1.0 | 49K | Working |
| RLHF | 0.50 | 1.0 | 16K | Working (GenRM Kimi K2.5) |
| Workbench | 0.44 | 1.0 | 49K | Fixed (ground-truth seeded mocks + partial credit) |
| SWE Agentless | 0.12-0.46 | 1.0 | 98K | Working (LLM judge); sandbox errors in execution mode |
| SWE Agentic | TBD | -- | 262K | R2E-Gym Docker integration wired up, untested at scale |

## Key Findings

1. LoRA LR = 10x paper's full-FT LR (SFT: 3e-4, RL: 3e-5)
2. group_size=16 critical for mixed groups and GRPO signal
3. LLM judge envs need max_tokens >= 256 for thinking models
4. `<think>` tags must be stripped before answer/code/patch extraction
5. R2E-Gym Docker images solve SWE dependency issues

## Next Steps

1. Finish SFT (33K steps) -> launch full RL cascade
2. RL cascade: SFT -> IF-RL (180 steps) -> Multi-domain (70 steps)
3. Benchmark after each stage
4. Test SWE Agentic at scale with R2E-Gym Docker
5. Ground-truth-seeded mocks for Workbench production runs

## Wandb

(internal — see team Wandb project)

## SFT Checkpoint (2026-03-30)

- **Step 500** saved with 14-day TTL
- **State**: `tinker://dcac3236-4699-55be-8375-9e54e071c056:train:0/weights/sft_step500_permanent`
- **Sampler**: `tinker://dcac3236-4699-55be-8375-9e54e071c056:train:0/sampler_weights/sft_step500_permanent`
- **NLL**: 0.687 -> 0.542 (500 steps, Super 120B, LoRA rank 64)
