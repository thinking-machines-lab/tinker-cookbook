# SWE Agentic RL Experiment (2026-03-27)

## Setup

- **Model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **SFT Checkpoint**: `tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final`
- **Dataset**: R2E-Gym/R2E-Gym-Subset (4578 instances, 10 repos)
- **Branch**: `nemotron-cascade-2-replication` (commit 8d63613 for Docker fixes)

## Command

```bash
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=swe_agentic \
    model_name='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16' \
    load_checkpoint_path='tinker://9814478b-c54c-5c5c-9967-40ab181a0b80:train:0/weights/final' \
    group_size=2 \
    groups_per_batch=2 \
    max_steps=2 \
    learning_rate=3e-5 \
    max_tokens=16384 \
    log_path=/tmp/swe_agentic_test \
    temperature=0.8 \
    save_every=0 \
    eval_every=0
```

Note: `chz` entrypoints use bare `key=value` syntax (no `--` prefix).

## Results

### Step 0

| Metric | Value |
|--------|-------|
| Reward | 0.0 (all trajectories) |
| Pass rate | 0.0 |
| Tests total | 1.0 |
| Turns per episode | 12.0 |
| Total turns (all trajs) | 24 |
| AC tokens/turn | 917.7 |
| OB tokens/turn | 18,927 |
| Total AC tokens | 22,024 |
| Total OB tokens | 454,249 |
| Max tokens reached | 1.0 (all hit limit) |
| Sampling time | 1,103s (18.4 min) |
| Train step time | 10.6s |
| Total step time | 1,117s (18.6 min) |

Group rollout times: max 1,103s (~18 min), mean 569s (~9.5 min).
Policy sample time: max 795s (~13 min), mean 44s.

### Step 1

| Metric | Value |
|--------|-------|
| Reward | 0.0 (all trajectories) |
| Pass rate | 0.0 |
| Tests total | 2.0 |
| Turns per episode | 3.0 |
| Total turns (all trajs) | 6 |
| AC tokens/turn | 850.0 |
| OB tokens/turn | 3,825.8 |
| Total AC tokens | 5,100 |
| Total OB tokens | 22,955 |
| Sampling time | 292s (4.9 min) |
| Train step time | 4.4s |
| Total step time | 301s (5.0 min) |

Group rollout times: max 292s (~4.9 min), mean 190s (~3.2 min).

### Per-Trajectory Breakdown

Step 0:
- group=0, traj=0: 3 steps, reward=0.0
- group=0, traj=1: 21 steps, reward=0.0 (this is the one that took ~17 min)
- group=1, traj=0: 1 step, reward=0.0
- group=1, traj=1: 1 step, reward=0.0

Step 1:
- group=0, traj=0: 5 steps, reward=0.0
- group=0, traj=1: 1 step, reward=0.0
- group=1, traj=0: 1 step, reward=0.0
- group=1, traj=1: 1 step, reward=0.0

### Gradient Signal

Both steps produced the warning: "All rewards are uniform. There will be no gradient."
With all rewards at 0.0, there is no RL training signal. The KL divergence increased
slightly (0.004 -> 0.011) due to the optimizer step with zero gradient + entropy effects.

## Model Behavior Observations

### Tool Usage

The model uses the correct tools:
- `read_file`: used frequently to examine source files
- `run_command`: used for `ls`, `find`, `grep` to explore the codebase
- `write_file`: used to attempt patches

Tool call counts across the full run: read_file ~198, run_command ~253, write_file ~72.

The model also tried `str_replace_editor` 3 times (a tool from SWE-bench's default agent
setup), which returned "Tool not found" errors. This suggests the SFT checkpoint was
partially trained on SWE-bench agent traces that use a different tool set.

### Thinking Behavior

The model emits `<think>` tags with its reasoning, but the thinking tokens are
sometimes garbled (e.g., `<think> garrison`, `<think> favoring`, `<think> grap`). This
may be a tokenizer/renderer issue or an artifact of the SFT training data.

### Sample Trajectory (Step 0, Group 0, Traj 1 - 21 steps)

Problem: pandas Series.loc with too many indices raising TypeError instead of ValueError.

The model:
1. Ran `ls -la` to explore the repo structure
2. Used `find` and `grep` to locate indexing-related code
3. Read `pandas/core/indexing.py` (the relevant file)
4. Searched for "unhashable type: slice" and "Too many indexers" error messages
5. Located the relevant code at lines 764 and 1366
6. Continued exploring the `_getitem_axis` method
7. Got stuck in a repetitive loop ("Let's look at the _getitem_axis method at line 910")
   repeated many times until hitting the token limit

The model correctly identified the issue and the relevant code but failed to make the
fix. The repetitive generation loop consumed most of the 16K token budget without
producing a write_file call with the actual fix.

### Observation Token Dominance

Observation tokens (tool outputs) vastly outnumber action tokens (model outputs):
- Step 0: 454K OB tokens vs 22K AC tokens (20:1 ratio)
- Step 1: 23K OB tokens vs 5K AC tokens (4.5:1 ratio)

`read_file` on large source files returns the entire file content, which quickly fills
the context. The model effectively spends most of its token budget on reading files.

## Error Messages

- "Tool 'str_replace_editor' not found" (3 occurrences) -- model tried wrong tool
- No Modal sandbox errors or crashes
- No Docker image pull failures

## Assessment

### What Works

1. **Modal sandbox integration**: Docker images pull correctly, sandboxes start and run
   without errors.
2. **Tool execution**: read_file, write_file, run_command all work correctly in the
   sandbox. The model can explore the repo, read files, and run commands.
3. **End-to-end pipeline**: Dataset loading, environment creation, rollouts, reward
   computation, and training steps all complete successfully.
4. **Test execution**: pytest runs correctly in the sandbox for reward evaluation.

### What Doesn't Work (Yet)

1. **Zero reward**: The model cannot solve any of the problems attempted. All 8
   trajectories across 2 steps got reward=0. This means no RL training signal.

2. **Repetitive generation**: The model gets stuck in repetitive loops, generating the
   same reasoning text over and over until the token limit is hit. This wastes the
   token budget without producing useful edits.

3. **Wrong tool calls**: The model occasionally tries `str_replace_editor` (from
   SWE-bench) instead of the available `write_file` tool. This suggests the SFT
   training didn't fully align the model with this tool set.

4. **Observation bloat**: Large files consumed via read_file dominate the context.
   The MAX_OUTPUT_CHARS=16384 cap helps but doesn't prevent the model from reading
   multiple large files that overwhelm the trajectory.

### Feasibility for Training

**Speed**: At current config (group_size=2, groups_per_batch=2), one step takes 5-18
minutes. With paper hyperparameters (group_size=64, batch_size=16), a single step
would take hours. This is consistent with the paper's use of max_turns=200 and
max_tokens=256K.

**Training signal**: The fundamental problem is zero reward. Without any successful
solves, there's no gradient signal for GRPO. Possible mitigations:

- **Easier problems**: Filter for simpler instances (fewer files to change, smaller
  repos, shorter test files).
- **Partial rewards**: Add intermediate rewards for correct tool usage, file
  identification, or partial test passage.
- **Better SFT**: The SFT checkpoint may not have been trained on agentic SWE data
  with this specific tool set. Fine-tuning on traces using read_file/write_file/
  run_command (instead of str_replace_editor) could help.
- **Smaller max_tokens**: 16K is quite small for SWE tasks. The paper uses 256K.
  However, even with more tokens, the repetitive generation bug would waste them.
- **Temperature**: 0.8 may be too low for exploration. The paper uses 0.8, but the
  initial policy may need higher temperature to generate diverse attempts.

**Recommendation**: The pipeline is end-to-end functional. The bottleneck is model
capability, not infrastructure. Next steps should focus on:
1. Curating an easier problem subset (or using a pass@k metric to estimate solvability)
2. Running with larger max_tokens (at least 65K)
3. Potentially retraining SFT with the correct tool definitions
4. Adding a warm-up phase with expert demonstrations

## Log Files

- Metrics: `/tmp/swe_agentic_test/metrics.jsonl`
- Rollout summaries: `/tmp/swe_agentic_test/iteration_00000{0,1}/train_rollout_summaries.jsonl`
- Full transcripts: `/tmp/swe_agentic_test/iteration_00000{0,1}/train_logtree.json`
- HTML reports: `/tmp/swe_agentic_test/iteration_00000{0,1}/train.html`
- Timing: `/tmp/swe_agentic_test/timing_spans.jsonl`
