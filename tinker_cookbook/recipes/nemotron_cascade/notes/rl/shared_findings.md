# Shared RL Findings

## Key Hyperparameters

### Learning Rate
- Paper uses 3e-6 for full fine-tuning across all RL stages
- For LoRA, 10x scaling needed: **3e-5 is our best RL LR**
- IF-RL lr=3e-5: +0.082 reward in 4 steps (paper-matched settings)

### Group Size and Batch
- Paper: group=16, batch=128 for most stages
- group=16 confirmed working; batch=32 is practical (batch=128 validated for IF-RL)
- Code RL and SWE need group=16+ to get any non-zero reward

### Max Tokens
- Paper: 49K for most stages, 118K for Code RL, 256K for SWE Agentic
- Critical for reasoning models -- the model needs space for `<think>` chains
- 8K truncates thinking too much; 49K is the minimum for quality RL

### Dynamic Filtering
- Paper removes groups where all rollouts agree (all correct or all incorrect)
- Ensures every group contributes gradient signal

## Cross-Cutting Issues

### 1. LLM Judge Token Limits (affects Long-ctx, RLHF)
Both Long-ctx and RLHF use thinking model judges. If max_tokens is too low, reasoning gets truncated before the score/verdict. All LLM-judge envs should set max_tokens >= 256.

### 2. Reward Variance for GRPO (affects StructOut, Code RL, SWE)
GRPO centers rewards within each group. If all rollouts get the same reward, advantages are zero. Fix: use partial/fractional rewards instead of strict binary to create within-group variance.

### 3. `<think>` Tag Handling (affects MCQA, all text extraction)
The model produces `<think>...</think>` reasoning before the answer. Any text extraction should strip the thinking block first, then extract from the actual response.

### 4. Modal Sandbox Reliability (affects Code RL, SWE Agentless, SWE Agentic)
Common issues: missing dependencies, shallow clone failures, timeouts. R2E-Gym Docker images solve dependency issues for SWE envs.

### 5. Reward Function Complexity Ladder
1. **Programmatic exact** (IF-RL, StructOut): Fastest, most reliable
2. **Programmatic with execution** (MCQA, Code RL): Medium speed, execution can fail
3. **LLM judge** (Long-ctx, RLHF): Slow, noisy, expensive
4. **Sandbox execution** (SWE Agentless, SWE Agentic, Workbench): Slowest, most failure modes

## Common Patterns

- All RL envs use GRPO (importance_sampling loss in Tinker)
- KL penalty = 0 for all stages except RLHF (0.03)
- Temperature = 1.0 for all stages except SWE Agentic (0.8)
- Cosine LR schedule (constant in short RL runs)
