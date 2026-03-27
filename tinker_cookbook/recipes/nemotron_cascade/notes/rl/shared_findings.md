# Shared RL Findings

## Key Hyperparameters

### Learning Rate
- Paper uses 3e-6 for full fine-tuning across all RL stages
- For LoRA, 10x scaling needed: **3e-5 is our best RL LR**
- IF-RL lr=3e-5: +0.082 reward in 4 steps (paper-matched settings)
- IF-RL lr=1e-5: +0.028 in 3 steps (slower learning)

### Group Size and Batch
- Paper: group=16, batch=128 for most stages
- Critical for RL signal quality — small groups (2-4) give very noisy gradients
- group=16 confirmed working, batch=32 is practical (batch=128 too slow with 6 parallel jobs)
- Code RL and SWE need group=16+ to get any non-zero reward

### Max Tokens
- Paper: 49K for most stages, 118K for Code RL, 256K for SWE Agentic
- Critical for reasoning models — the model needs space for <think> chains
- 8K truncates thinking too much; 49K is the minimum for quality RL

### Dynamic Filtering
- Paper removes groups where all rollouts agree (all correct or all incorrect)
- Ensures every group contributes gradient signal
- Our IF-RL shows frac_mixed=1.0 which is good (all groups have variance)

## RL Environment Status

| Env | Reward | Signal | Key Issue |
|-----|--------|--------|-----------|
| IF-RL | 0.70-0.79 | Strong | Working well with lr=3e-5 |
| MCQA | 0.55-0.62 | Medium | May need different LR |
| StructOut | 0.75-1.0 | Strong | |
| Code RL | 0.09 (g=16) | Weak | Needs more rollouts |
| Long-ctx | 0.10 | Weak | LLM judge slow |
| SWE Agentless | 0.0 | None | Hard task, needs scale |
| SWE Agentic | 0.0 | None | 19min/step, very slow |
| RLHF | ? | Unknown | Logging issue |
| Workbench | 0.0 | None | Mock backend, tool discovery fixed |

## Common Patterns
- All RL envs use GRPO (importance_sampling loss in Tinker)
- KL penalty = 0 for all stages except RLHF (0.03)
- Temperature = 1.0 for all stages except SWE Agentic (0.8)
- Cosine LR schedule (constant in short RL runs)
