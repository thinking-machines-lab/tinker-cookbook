# Nemotron-Cascade-2 Replication Experiments

Replicating NVIDIA's Nemotron-Cascade-2 (arxiv:2603.19220) using Tinker API.

## Model
- `openai/gpt-oss-120b:peft:131072` (LoRA fine-tuning)
- Hidden size: 2880, MoE architecture

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
Focus on stages 1-3 (SFT → IF-RL → Multi-domain RL).

## Experiment Log

### SFT LR Sweep
- Date: 2026-03-26
- Data: instruction_following subset, 10K examples
- LRs tested: 1e-4, 3e-4, 5e-4, 1e-3
- Batch size: 64
- LoRA rank: 32
- Renderer: gpt_oss_no_sysprompt
- Cosine LR schedule, AdamW (beta1=0.9, beta2=0.98)

| LR | Steps | Final NLL | Notes |
|----|-------|-----------|-------|
| 1e-4 | | | |
| 3e-4 | | | |
| 5e-4 | | | |
| 1e-3 | | | |

### Full SFT
- Date:
- Data:
- LR:
- Steps:
- Checkpoint:

### IF-RL
- Date:
- Load checkpoint:
- Steps:
- Reward:
