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

| Env | Reward | Signal | Key Issue | Priority Fix |
|-----|--------|--------|-----------|--------------|
| IF-RL | 0.70-0.79 | Strong | Working well with lr=3e-5 | Scale batch to 128 |
| MCQA | 0.55-0.62 | Noisy | False-positive answer matching | Fix extract_answer + check_answer |
| StructOut | 0.75-1.0 | Near-zero (too easy) | No real schema validation | Add jsonschema library |
| Code RL | 0.09 (g=16) | Weak | MBPP execution issues, 49K too short | Set max_tokens=118K, verify execution |
| Long-ctx | 0.10 | Weak | Judge truncated at 32 tokens | Set judge max_tokens=256+ |
| RLHF | ? | Unknown | GenRM may not be accessible | Verify GenRM model availability |
| Workbench | 0.0 | None | Mock backend defeats multi-step tasks | Filter to single-call tasks |
| SWE Agentless | 0.0 | None | No codebase context in prompt | Add file contents to prompt |
| SWE Agentic | 0.0 | None | Missing deps, 19min/step | Use SWE-bench Docker images |

## Cross-Cutting Issues

### 1. LLM Judge Token Limits (affects Long-ctx, RLHF)
Both Long-ctx and RLHF use Qwen3.5-397B-A17B as a judge/reward model. Qwen3.5 is a thinking model that uses `<think>` tags. If max_tokens is too low (32 for long-ctx judge), the model's reasoning gets truncated before it outputs the actual score/verdict. **All LLM-judge envs should set max_tokens >= 256.**

### 2. Reward Variance is Critical for GRPO (affects StructOut, Code RL, SWE)
GRPO centers rewards within each group. If all rollouts get the same reward (all 1.0 or all 0.0), the advantages are zero and the group contributes nothing to training. Dynamic filtering removes these groups, but:
- StructOut: Nearly all groups are all-1.0 → effectively no training
- SWE envs: Nearly all groups are all-0.0 → effectively no training
- Code RL: Most groups are all-0.0 with occasional 1 → very sparse signal

**Fix: Use partial/fractional rewards** instead of strict binary to create within-group variance even when most rollouts fail.

### 3. `<think>` Tag Handling (affects MCQA, all text extraction)
The model produces `<think>...</think>` reasoning before the answer. Any text extraction (answer extraction in MCQA, code extraction in Code RL, patch extraction in SWE) should strip the thinking block first, then extract from the actual response.

### 4. Modal Sandbox Reliability (affects Code RL, SWE Agentless, SWE Agentic)
Three envs use Modal sandboxes. Common issues:
- Missing dependencies (`pip install -e .` fails silently)
- Shallow clone doesn't reach target commit
- Timeout (30s for code, 300s for SWE) may be too short for complex cases
**Consider using pre-built Docker images with dependencies for SWE envs.**

### 5. Reward Function Complexity Ladder
Envs naturally order by reward function complexity:
1. **Programmatic exact** (IF-RL, StructOut): Fastest, most reliable
2. **Programmatic with execution** (MCQA, Code RL): Medium speed, execution can fail
3. **LLM judge** (Long-ctx, RLHF): Slow, noisy, expensive
4. **Sandbox execution** (SWE Agentless, SWE Agentic, Workbench): Slowest, most failure modes

**Recommendation: Get all programmatic envs working well first, then tackle LLM-judge and sandbox envs.**

## Recommended Priority Order for Improvements

1. **IF-RL**: Scale batch to 128, run 180 steps → should match paper results
2. **MCQA**: Fix answer extraction (strip think, exact match only) → expect +0.05-0.10
3. **Structured Output**: Add jsonschema validation → expect reward drop to 0.5-0.7 (good)
4. **Long-ctx**: Set judge max_tokens=256 → expect reward jump to 0.3-0.5
5. **Code RL**: Verify MBPP execution, set max_tokens=118K → expect 0.15-0.25
6. **RLHF**: Verify GenRM accessibility → expect 0.3-0.7 once working
7. **Workbench**: Filter to single-call tasks → expect 0.2-0.4 on subset
8. **SWE Agentless**: Add codebase context, use Docker images → expect 0.05-0.15
9. **SWE Agentic**: Needs Docker images + stronger model → defer to later stage

## Common Patterns
- All RL envs use GRPO (importance_sampling loss in Tinker)
- KL penalty = 0 for all stages except RLHF (0.03)
- Temperature = 1.0 for all stages except SWE Agentic (0.8)
- Cosine LR schedule (constant in short RL runs)
