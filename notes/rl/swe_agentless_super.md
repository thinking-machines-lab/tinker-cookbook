# SWE Agentless RL on Nemotron Super (262K context)

**Date:** 2026-03-28
**Model:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144`
**Branch:** `nemotron-cascade-2-replication`
**No SFT checkpoint (base model)**

## Summary

Testing SWE Agentless RL with the Cascade SWE dataset (`nvidia/Nemotron-Cascade-RL-SWE`, ~141K instances) on the Super model. The Cascade dataset includes pre-built prompts with issue descriptions + relevant codebase context + golden patches, matching the Nemotron Cascade 2 paper's approach.

Previous baseline: R2E-Gym dataset on Super = 0.175 reward (execution-based).
Earlier Nano experiments with Cascade data: 0.38-0.46 reward (LLM judge).

## Experiment 1: LLM Judge + Cascade Data

```
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    model_name=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144 \
    env=swe_rl group_size=4 groups_per_batch=2 max_tokens=98304 \
    context_window=262144 learning_rate=3e-5 max_steps=2 \
    log_path=$HOME/experiments/nemotron-cascade-2/rl/swe_agentless_super \
    wandb_project=nemotron-cascade-2-replication \
    wandb_name=super-262k-swe-agentless
```

**WandB:** https://wandb.ai/thinking-machines-lab-inc/nemotron-cascade-2-replication/runs/hgnxwpod

### Results

| Step | judge_reward | has_patch | Total Reward | obs_tokens (avg) | ac_tokens (avg) |
|------|-------------|-----------|-------------|-------------------|-----------------|
| 0    | 0.200       | 1.0       | 0.200       | 9,075             | 6,029           |
| 1    | 0.125       | 1.0       | 0.125       | 10,375            | 5,477           |

**Note:** The `has_patch` and `judge_reward` metrics are batch averages. Looking at per-trajectory data:

### Per-Trajectory Breakdown

**Step 0:**
- Group 0 (24K obs tokens): All 4 trajectories failed to produce patches (judge=0.0). Prompt was very long.
- Group 1 (9K obs tokens): 4/4 produced patches. Rewards: 0.2, 0.0, 0.3, 0.3.

**Step 1:**
- Group 0 (10K obs tokens): 4/4 produced patches. Rewards: 0.1, 0.1, 0.1, 0.2.
- Group 1 (27K obs tokens): All 4 trajectories failed to produce patches (judge=0.0).

### Key Observations

1. **Patch generation correlates with prompt length.** When observation tokens are ~9-10K, the model reliably generates patches. At ~24-27K tokens, it fails entirely. The model may be spending too much context on the prompt, leaving insufficient space/attention for generating a proper diff.

2. **Non-zero rewards when patches are produced.** When the model does generate patches, it gets 0.1-0.3 judge rewards. The best individual trajectory scored 0.3/1.0 (equivalent to 3/10 from the judge).

3. **Very slow iteration time.** Each step took ~25 min with group_size=4, groups_per_batch=2. Policy sampling dominated at ~10-20 min per group due to long contexts. Full-scale training (128x16 batch) would be impractical without async/streaming.

4. **Mixed groups prevent complete collapse.** All groups had frac_mixed=1.0 (some trajectories got reward, others didn't), which is good for GRPO learning signal.

### Timing

- Sampling: ~23 min per batch (2 groups of 4)
- Judge rewards: ~2-4 min per group
- Training step: ~1 min
- Total per step: ~25 min

## Experiment 2: Execution-Based Reward (R2E-Gym)

```
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    model_name=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144 \
    env=swe_rl group_size=4 groups_per_batch=2 max_tokens=98304 \
    context_window=262144 learning_rate=3e-5 max_steps=1 \
    swe_reward_mode=execution swe_use_cascade_data=false swe_use_r2e_gym=true \
    log_path=$HOME/experiments/nemotron-cascade-2/rl/swe_agentless_super_exec \
    wandb_project=nemotron-cascade-2-replication \
    wandb_name=super-262k-swe-execution
```

**Status:** FAILED -- killed after 50+ minutes stuck on batch 0 group 2.

**WandB:** https://wandb.ai/thinking-machines-lab-inc/nemotron-cascade-2-replication/runs/d7wdnnmx

All Modal sandbox attempts for pandas R2E-Gym Docker images timed out (300s default).
Group 1 completed (all 4 trajectories got reward=0 due to sandbox errors). Group 2 was
stuck in a cycle of sandbox creation -> timeout -> retry with the same pandas instances.

The execution-based approach with R2E-Gym on Super is not viable without:
- Increasing Modal sandbox timeout (>300s for heavy repos like pandas)
- Filtering out heavy repos (pandas, scikit-learn) that exceed sandbox limits
- Using LLM judge as the primary reward signal instead

## Comparison to Previous Results

| Setup | Dataset | Reward Mode | Reward |
|-------|---------|------------|--------|
| Super (R2E-Gym, prev) | R2E-Gym (4.5K) | Execution | 0.175 |
| **Super (Cascade, this)** | **Cascade SWE (141K)** | **LLM Judge** | **0.163 avg** |
| Nano (Cascade, prev) | Cascade SWE | LLM Judge | 0.38-0.46 |

The Super model's initial Cascade SWE judge reward (0.163 averaged over 2 steps) is lower than the Nano experiments (0.38-0.46). This is likely because:
1. Only 2 steps on base model (no SFT warmup)
2. Super model with 262K context may need different hyperparameters
3. The model struggles with the longest prompts (>20K obs tokens)

## Recommendations

1. **Filter by prompt length.** Skip instances with prompts > 15K tokens to improve patch generation rate.
2. **Start from SFT checkpoint.** The paper trains on SFT first, then RL. Base model may not be code-fluent enough.
3. **Increase group_size after SFT.** More rollouts per problem = better variance reduction.
4. **Consider async training** to overlap sampling and training for faster iteration.
