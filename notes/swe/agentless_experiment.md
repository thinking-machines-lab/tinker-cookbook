# SWE Agentless RL Experiment Results (2026-03-27)

## Overview

Ran SWE Agentless RL training experiments comparing execution-based reward
(Modal Docker sandboxes) vs LLM judge reward (Qwen3.5-397B). Both used the
SFT checkpoint from IF-RL training as the starting point.

## Configuration

Both experiments:
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Checkpoint: `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final`
- LoRA rank: 32
- LR: 3e-5
- Max tokens: 16384
- Temperature: 1.0
- Group size: 4
- Groups per batch: 4
- Steps: 3
- Dataset: R2E-Gym-Subset (4578 instances, `use_r2e_gym=True`)
- Loss: importance_sampling
- `remove_constant_reward_groups=True`

## Commands Used

The CLI (`train_rl.py`) does not expose `reward_mode` as a flag, so custom
launcher scripts were used. See `/tmp/run_swe_exec.py` and
`/tmp/run_swe_llm_judge.py`.

Core difference: `SWERLDatasetBuilder(reward_mode="execution")` vs
`SWERLDatasetBuilder(reward_mode="llm_judge")`.

## Results: Execution-Based Reward

Log path: `/tmp/swe_agentless_execution_test`

| Step | Reward | has_patch | correct | frac_all_bad | entropy | total_time (s) |
|------|--------|-----------|---------|--------------|---------|----------------|
| 0    | 0.000  | 1.000     | 0.000   | 1.000        | 0.802   | 682            |
| 1    | 0.000  | 1.000     | 0.000   | 1.000        | 0.479   | 530            |
| 2    | 0.000  | 1.000     | 0.000   | 1.000        | 0.661   | 668            |

**Zero reward signal across all 3 steps.** No patches passed execution tests.
All groups filtered as constant-reward (frac_all_bad=1.0), meaning only 4
episodes per step contributed to metrics (the minimum kept for logging).

### Failure Analysis (Execution Mode)

Three failure modes observed:

1. **Modal ARG_MAX error** (~50% of instances): R2E-Gym test files are too
   large to embed in the bash command argument. Modal's sandbox has a 65536-byte
   limit on CMD arguments. Test scripts ranged from 76K to 168K bytes.
   ```
   Modal error: Total length of CMD arguments cannot exceed 65536 bytes (ARG_MAX). Got 98872 bytes.
   ```

2. **Patches don't pass tests** (~40% of instances): The patch applied
   successfully, but the generated fix was incorrect and tests failed. This is
   expected for a model that hasn't been specifically trained on SWE tasks.

3. **No patch extracted** (~10% of instances): The model generated a response
   that didn't contain a recognizable unified diff.

### Bug Found: Modal ARG_MAX Limit

`run_swe_test_in_modal()` in `swe_rl_env.py` embeds test file contents as
base64-encoded strings in the bash command passed to `modal.Sandbox.create()`.
When R2E-Gym test files are large (which is common), this exceeds Modal's
65536-byte CMD argument limit.

**Fix needed**: Write test files using Modal's filesystem API or split the
sandbox execution into multiple steps (first write files, then run tests).
This would likely unblock 50%+ of execution-mode instances.

### Timing (Execution Mode)

- Policy sampling (generation): 3-11 min per step (dominates)
- Modal sandbox execution (env_step): 2-84s per instance
- Training step: 3-12s
- Total per step: 9-11 minutes

## Results: LLM Judge Reward

Log path: `/tmp/swe_agentless_llm_judge_test`

| Step | judge_reward | has_patch | frac_mixed | entropy | total_time (s) |
|------|-------------|-----------|------------|---------|----------------|
| 0    | 0.463       | 1.000     | 1.000      | 0.720   | 884            |
| 1    | 0.275       | 1.000     | 1.000      | 0.580   | 797            |
| 2    | 0.408       | 1.000     | 1.000      | 0.618   | 821            |

**Meaningful reward signal!** The LLM judge (Qwen3.5-397B) provides graded
rewards from 0.1 to 1.0 across patches.

### Judge Reward Distribution (Step 0)

Individual trajectory rewards: [0.2, 0.3, 0.2, 0.1, 0.6, 0.4, 0.3, 0.7,
0.1, 0.7, 0.3, 1.0, 0.1, 0.8, 0.8, 0.8]

- Mean: 0.463
- Min: 0.1, Max: 1.0
- All 4 groups had mixed rewards (frac_mixed=1.0)
- All 16 trajectories contributed to training (vs only 4 in execution mode)

### Reward Trajectory

The judge rewards fluctuated: 0.463 -> 0.275 -> 0.408. With only 3 steps and
4 groups per batch, this is within normal variance. Too few steps to determine
if the model is learning.

### Timing (LLM Judge Mode)

- Policy sampling (generation): 8-14 min per step (dominates)
- LLM judge scoring (compute_group_rewards): 4-5 min per step
- env_step: <10ms (no sandbox)
- Training step: 3-8s
- Total per step: 13-15 minutes

## Key Findings

### 1. LLM Judge >> Execution for Current Setup

The LLM judge mode is clearly superior for training:
- Provides graded reward signal (0.1-1.0) vs binary 0/1
- No ARG_MAX bug affecting half the instances
- All groups have reward variance (frac_mixed=1.0) enabling learning
- Slightly slower (13-15 min/step vs 9-11 min/step) but all the extra time
  is productive (judge scoring)

### 2. Execution Mode Has Two Blocking Issues

a) **ARG_MAX bug**: ~50% of R2E-Gym instances fail because test files are
   too large to embed in the bash command. This needs a code fix.

b) **No reward signal**: Even for instances that run, the model (with SFT
   checkpoint) produces zero correct patches out of 48 attempts. This makes
   execution-based reward useless at this training stage -- all groups are
   constant (all-zero) and get filtered, providing no gradient signal.

### 3. The Model Produces Patches, Just Not Correct Ones

`has_patch=1.0` consistently -- the model learned to output diff-formatted
patches from SFT. The patches are just not correct enough to pass tests.
This aligns with the paper's approach of using a softer LLM judge reward
during early RL training.

### 4. Scaling Concerns

At production scale (128 groups x 16 rollouts = 2048 per step, 98K tokens):
- Each step would take ~1-2 hours for generation
- LLM judge adds ~1 hour per step
- Execution would add significant time + Docker Hub rate limits
- Paper ran 40-50 steps total (~40-100 hours)

## Recommendations

1. **Use LLM judge mode for SWE RL training.** It matches the paper's
   approach and provides the only viable reward signal.

2. **Fix the ARG_MAX bug** for execution-based validation/eval (even if not
   used for training rewards). Use Modal's file writing API instead of
   embedding test files in bash arguments.

3. **Consider a curriculum**: Start with LLM judge, then switch to execution
   after the model improves enough to pass some tests.

4. **Expose `reward_mode` in the CLI** (`train_rl.py`'s `get_dataset_builder`
   should forward this parameter).

5. **Scale gradually**: Start with groups_per_batch=16, group_size=4 (64
   rollouts/step) before going to the paper's 2048 rollouts/step.
