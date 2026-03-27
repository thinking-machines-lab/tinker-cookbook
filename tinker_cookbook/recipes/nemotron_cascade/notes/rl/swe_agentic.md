# SWE Agentic RL Environment Analysis

## Status: WORKS E2E, REWARD=0, 19min/step — R2E-Gym path verified

## Architecture
- Multi-turn: model uses read_file, write_file, run_command tools in Modal sandbox
- Up to 200 turns per episode
- group_size: 64 (paper), sandbox_timeout: 600s per sandbox
- Reward: binary (FAIL_TO_PASS tests pass = 1, else = 0)
- Paper: batch=16, group=64, max_tokens=256K, temp=0.8

## Why It's Expensive
- Each rollout creates a Modal sandbox, clones repo, installs deps
- 64 sandboxes per group (paper's group_size)
- Each sandbox runs up to 200 turns of tool interaction
- Then runs test suite for grading
- Total: ~19 minutes per training step

## Why Reward = 0

### 1. Same dependency issues as agentless SWE
The Modal sandbox installs `pytest`, `pytest-timeout`, `setuptools` but not project-specific deps. Many repos need Django, Flask, numpy, etc.

### 2. Model explores but doesn't fix
With 200 turns and 256K context, the model can read files and understand the codebase. But Nano-30B (3B active params) likely struggles with:
- Identifying the root cause from issue descriptions
- Making precise code changes that fix the bug without breaking other things
- Understanding complex test setups

### 3. Setup script may fail silently
```bash
pip install -e . 2>/dev/null || true
```
Silent failure means the model's `run_command("python -m pytest ...")` also fails, but it can't distinguish "test failed because of my change" from "test failed because deps are missing."

### 4. Context overflow
With 200 turns of read_file/run_command, context can easily exceed 256K tokens. The `context_overflow_reward=0.0` parameter gives 0 reward when this happens, but doesn't help the model learn to be more efficient.

## Actionable Improvements

### P0: Use SWE-bench Docker images
Replace the generic Modal sandbox with SWE-bench's pre-built Docker images. These have all dependencies installed and the correct Python version.

### P1: Reduce group_size for cost efficiency
With reward=0 across all 64 rollouts, there's no GRPO signal anyway. Use group_size=4 or 8 to reduce cost while debugging. Only scale up once reward > 0.

### P2: Start with the easiest SWE-bench instances
Filter to instances that are:
- Single-file fixes
- Have short problem statements
- Are in well-known repos (django, flask, etc.)
- Have been solved by other models at high rates

### P3: Pre-warm the sandbox
Run setup + dep installation in the Docker image build step, not during rollout. This saves ~2-3 minutes per sandbox.

### P4: Add intermediate rewards
Instead of binary final reward:
- +0.1 for reading relevant files (matching files mentioned in the issue)
- +0.2 for making a write_file call (attempting a fix)
- +0.3 for patch that doesn't break existing passing tests
- +1.0 for all FAIL_TO_PASS tests pass

### P5: Limit turns for cost control
Set max_turns=50 instead of 200. Most successful SWE-bench solutions take <30 tool calls. 200 turns means the model can waste compute on unproductive exploration.

### P6: Consider cheaper alternatives first
Before investing in expensive SWE agentic RL:
1. Get agentless SWE-RL working with codebase context (much cheaper per step)
2. Use agentless to pre-filter easy instances
3. Only run agentic on instances where agentless showed some promise

## R2E-Gym Integration (verified 2026-03-27)

The R2E-Gym path is fully wired up in `swe_agentic_env.py`:
- `SWEAgenticDatasetBuilder` loads `R2E-Gym/R2E-Gym-Subset` when `use_r2e_gym=True` (default)
- `docker_image` from dataset -> `modal.Image.from_registry()` with workdir `/testbed`
- Test files from `execution_result_content` JSON -> written to `r2e_tests/` in sandbox
- `SWEAgenticReward` runs these test files for binary pass/fail grading

Key data fields used from R2E-Gym:
- `docker_image`: pre-built Docker image with repo + deps at `/testbed`
- `problem_statement`: issue description for the prompt
- `execution_result_content`: JSON with `test_file_names` and `test_file_codes`
- `repo_name`, `commit_hash`, `instance_id`: metadata

The `expected_output_json` field is not used directly -- the env runs the actual
test files rather than comparing against expected outputs.

## Expected Impact
The R2E-Gym Docker images solve the dependency problem (P0). However, the agentic env
remains expensive (~19min/step at group=64). For Nano-30B, getting non-zero reward
likely requires:
1. SFT on coding tasks first (the agentless SWE-RL with LLM judge can help here)
2. Starting with easy instances
3. Reducing group_size to 4-8 during debugging
