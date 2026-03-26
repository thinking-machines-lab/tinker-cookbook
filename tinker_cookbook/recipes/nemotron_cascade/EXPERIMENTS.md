# Nemotron-Cascade-2 Replication Experiments

Replicating NVIDIA's Nemotron-Cascade-2 (arxiv:2603.19220) using Tinker API.

## Models
- `openai/gpt-oss-120b:peft:131072` (LoRA fine-tuning, 128K ctx, MoE, hidden=2880)
- `Qwen/Qwen3-8B-Base` (LoRA fine-tuning, 32K ctx, dense base model)

## Paper Pipeline
1. SFT on ~24.8M examples (multi-domain)
2. IF-RL (instruction following, ~180 steps)
3. Multi-domain RL (~70 steps)
4. MOPD (on-policy distillation)
5. RLHF
6. Long-context RL
7. Code RL
8. SWE RL

## Our Replication Plan
Focus on stages 1-3 (SFT -> IF-RL -> Multi-domain RL).

## Experiment Log

### SFT LR Sweep (commit: 5b17edc)
- Date: 2026-03-26
- Data: instruction_following subset, 500 examples
- LRs tested: 1e-4, 3e-4, 5e-4, 1e-3
- Batch size: 16
- Max length: 8192
- LoRA rank: 32
- Renderer: gpt_oss_no_sysprompt (gpt-oss), role_colon (Qwen3)
- Cosine LR schedule, AdamW (beta1=0.9, beta2=0.98)

#### gpt-oss-120b Results

| LR | Steps | First NLL | Min NLL | Final NLL | Notes |
|----|-------|-----------|---------|-----------|-------|
| 1e-4 | 53 | 1.008 | 0.409 | 0.483 | Too conservative |
| 3e-4 | 53 | 0.900 | 0.367 | 0.449 | Good |
| **5e-4** | **31** | **0.900** | **0.362** | **0.452** | **Best - matches paper 5e-5 * 10x LoRA** |
| 1e-3 | 31 | 0.900 | 0.426 | 0.488 | Too aggressive |

#### Qwen3-8B-Base Results

| LR | Steps | First NLL | Min NLL | Final NLL | Notes |
|----|-------|-----------|---------|-----------|-------|
| 1e-4 | 31 | 1.310 | 0.932 | 0.946 | Too conservative |
| 3e-4 | 31 | 1.310 | 0.854 | 0.868 | OK |
| 5e-4 | 31 | 1.310 | 0.824 | 0.839 | Good (matches get_lr ~4.7e-4) |
| 1e-3 | 31 | 1.310 | 0.800 | 0.815 | Good, still improving |

**Decision**: Use lr=5e-4 for gpt-oss-120b, lr=5e-4 for Qwen3-8B-Base.

### Medium-scale SFT (10K examples)
- Date: 2026-03-26
- Data: instruction_following subset, 10K examples
- Model: gpt-oss-120b, lr=5e-4, batch_size=64
- Steps: ~140 (10K/64 - test_size)
- Status: RUNNING

### Full SFT
- Date:
- Data: math (100K) + science (50K) + instruction_following (10K) + safety (3.5K)
- LR: 5e-4
- Steps:
- Checkpoint:

### IF-RL
- Date:
- Load checkpoint:
- Steps:
- Reward:
