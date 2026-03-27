# SWE RL Research Findings (2026-03-27)

## How Nemotron Cascade 2 Actually Does SWE RL

### Agentless Stage
- **Reward model**: GPT-OSS-120B (execution-free, LLM judge) — NOT execution-based
- **Dataset**: nvidia/Nemotron-Cascade-RL-SWE (~110K instances from 5 sources: SWE-Bench-Train, SWE-reBench, SWE-Smith, R2E-Gym-Subset, SWE-Fixer-Train)
- **Prompt construction**: Uses BOTH golden localization AND top-5 retrieved localizations (codebase context!)
- **Filtering**: Easy samples filtered out (where most rollouts succeed)
- **Config**: batch=128×16=2048, max_seq=98,304 tokens, lr=3e-6, temp=1.0
- **Sparse reward handling**: Mask loss for prompts where NO rollout gets reward > 0.5
- **Steps**: ~40-50 to convergence

### Agentic Stage
- **Framework**: OpenHands (execution-based, Docker environments)
- **Datasets**: SWE-Gym + R2E-Subset
- **Config**: batch=16×64=1024, max_context=256K, up to 200 turns
- **Curriculum**: Remove instances where all rollouts pass; randomly discard 90% of instances where zero rollouts pass
- **Reward**: Execution-based (compilation + unit tests)

### CRITICAL FINDING: Separate SWE Dataset Released!
The paper uses `nvidia/Nemotron-Cascade-RL-SWE` — a SEPARATE dataset from `nvidia/Nemotron-Cascade-2-RL-data`.
- ~110K instances (vs 4,578 in R2E-Gym-Subset)
- Includes `relevant_file_contents` field (file paths + content) — this is the codebase context we're missing!
- Includes `original_prompt` with golden localization
- Max prompt lengths: 16K, 24K, or 32K tokens

## How DeepSWE Trains (Together AI, 2025)

- **Model**: Qwen3-32B, trained on 4,500 R2E-Gym tasks, 64 H100s, 6 days
- **Reward**: Sparse binary ORM (1 if patch passes tests, 0 otherwise)
- **Compact filtering**: Mask out trajectories that hit max context, timeout, or max steps
- **Algorithm**: GRPO++ with clip-high exploration, no KL, length normalization
- **Key insight**: Emergent behaviors appear — agents learn to search for regression tests, allocate more thinking tokens for complex steps

## How Multi-Turn SWE Agents Are Trained (Facebook SWE-RL)

- **Two-stage curriculum**: Stage 1 at 65K context → Stage 2 at 131K context
- **Filtering**: Remove flaky tests, invalid imports, >7 files or >500 lines changes, LLM quality scores
- **Sparse reward**: DAPO with group-normalized advantages (10 trajectories per group)
- **Critical anti-repetition finding**: DO NOT discard long/looping trajectories — keeping negative examples prevents the agent from learning repetitive loops
- **RFT warmup**: Rejection Fine-Tuning first (mask bad-format trajectories), improves baseline from 11% → 20% before RL

## SWE-RM: Execution-Free Reward Model

- **Architecture**: 30B MoE, 3B active (Qwen3-30B-A3B based)
- **Provides continuous [0,1] scores** vs binary execution feedback
- **RL benefit**: +3 absolute points over execution-only rewards
- **Hybrid is best**: Combining execution-free + execution-based rewards outperforms either alone
- **Training data**: ~100K trajectories needed, 2:1 positive:negative ratio

## Actionable Recommendations for Our Project

### For SWE Agentless (Priority: HIGH)

1. **USE THE NVIDIA SWE DATASET** (`nvidia/Nemotron-Cascade-RL-SWE`)
   - ~110K instances with `relevant_file_contents` — the codebase context we need!
   - This is the EXACT dataset the paper uses
   - Includes golden localization + retrieved localizations in prompts
   - Max prompt lengths 16-32K tokens (fits in 49K context)

2. **Use Kimi K2.5 as execution-free reward model** (already switched)
   - Paper uses GPT-OSS-120B; our Kimi K2.5 is a reasonable proxy
   - Execution-free reward is the paper's approach for agentless

3. **Filter easy/impossible instances**
   - Mask loss where no rollout gets reward > 0.5
   - Filter out easy samples where most rollouts succeed

4. **Set max_tokens=98K** (paper uses 98,304)

### For SWE Agentic (Priority: MEDIUM — defer to later cascade stage)

1. **Use OpenHands framework** (paper's approach) — our current tool set is simpler
2. **Start with easy instances**: Remove 90% of zero-reward instances, remove 100% of all-pass instances
3. **Keep negative examples**: Don't discard looping/long trajectories — they prevent repetitive generation
4. **RFT warmup**: Do supervised fine-tuning on successful trajectories first before RL
5. **Two-stage context**: Start at 65K, then extend to 131K+

### For Both Environments

1. **Hybrid rewards**: Combine execution-free (LLM judge) + execution-based for best results
2. **Group-normalized advantages**: DAPO/GRPO with 10-16 rollouts per group
3. **Compact filtering**: Mask out context-overflow and timeout trajectories

## Experiment Results: nvidia/Nemotron-Cascade-RL-SWE Dataset (2026-03-27)

### LLM Judge Mode — SUCCESS

Config: group_size=4, groups_per_batch=3, lr=3e-5, max_tokens=49152, reward_mode=llm_judge
Dataset: nvidia/Nemotron-Cascade-RL-SWE (streaming, ~110K instances)

| Step | Reward | Judge | Has Patch | Frac Mixed | Time (s) |
|------|--------|-------|-----------|------------|----------|
| 0 | 0.463 | 0.463 | 1.0 | 1.0 | 884 |
| 1 | 0.275 | 0.275 | 1.0 | 1.0 | 797 |
| 2 | 0.408 | 0.408 | 1.0 | 1.0 | 821 |

**Mean reward: 0.382** (vs 0.306 with R2E-Gym-Subset — 25% improvement)
**100% mixed groups** — perfect GRPO signal at every step

### Execution Mode — CONTEXT OVERFLOW

The Cascade SWE prompts are ~24K tokens. With max_tokens=49152, total exceeds the 65K context window:
`Prompt length plus max_tokens exceeds the model's context window: 24042 + 49152 > 65536`

Fix: Set max_tokens to min(49152, 65536 - prompt_length) dynamically, or cap at ~40K.
Paper uses 98,304 max_tokens but with a model that likely supports 128K+ context.

### Comparison: R2E-Gym-Subset vs Cascade SWE Data

| Metric | R2E-Gym-Subset | Cascade SWE |
|--------|---------------|-------------|
| Dataset size | 4,578 | ~110,000 |
| Has codebase context | No | Yes (relevant_file_contents) |
| Prompt length | ~2K tokens | ~24K tokens |
| LLM judge reward | 0.306 | 0.382-0.463 |
| Frac mixed | 1.0 | 1.0 |
| Execution reward | 0.0 | Context overflow (fixable) |
