# SWE-RL (Agentless) Environment Analysis

## Status: WORKING with LLM judge (reward ~0.3)

## Architecture
- Single-turn: model generates a unified diff patch in one shot
- Reward modes:
  - `llm_judge` (default): Qwen3.5-397B judges patch quality on 0-10 scale, normalized to [0,1].
    Matches the paper's approach of using GPT-OSS-120B as reward model.
  - `execution`: Modal sandbox clones repo, applies patch, runs FAIL_TO_PASS tests (binary 0/1).
- Dataset: R2E-Gym/R2E-Gym-Subset (4578 instances with docker_image), or legacy nvidia/Nemotron-Cascade-2-RL-data SWE-RL split (622 instances)

## LLM Judge Results (2026-03-27)

Nano checkpoint, group=4, batch=4, max_tokens=16384, 2 steps:
- `judge_reward`: 0.306 (mean across all 16 samples in step 0)
- `has_patch`: 1.0 (all samples produced valid diff patches)
- `frac_mixed`: 1.0 (all groups had reward variance -- good GRPO signal)
- Step time: ~15 min (most in sampling + judge calls)
- Judge call time: ~252s per group (4 sequential judge queries per group)

The LLM judge gives meaningful, differentiated rewards even when the model lacks
codebase context. This unblocks RL training for the agentless SWE path.

## Why Patches Don't Pass Tests

### 1. Patch extraction is fragile
```python
def extract_patch(response: str) -> str | None:
    # Try fenced code block
    match = re.search(r'```(?:diff|patch)?\s*\n(.*?)\n```', response, re.DOTALL)
    # Try to find unified diff lines directly
    ...
```
Problems:
- If model outputs a diff inside ````python` block (common mistake), extraction fails
- If model outputs multiple code blocks (explanation + diff), only the first fenced block is checked
- Direct diff detection breaks on blank lines within the diff

### 2. Shallow clone + commit checkout may fail
```bash
git clone --depth=1 https://github.com/{repo}.git /workspace/repo
git fetch --depth=1 origin {base_commit} && git checkout {base_commit}
```
With `--depth=1`, the specific `base_commit` may not be in the shallow history. The `|| true` means checkout failure is silently ignored, so the patch is applied to the wrong commit → test failures.

### 3. `pip install -e .` may fail
Many SWE-bench repos have complex dependency chains. The Modal sandbox only has `pytest`, `pytest-timeout`, and `setuptools`. Missing dependencies (like `numpy`, `scipy`, `django`, etc.) cause import errors during test execution.

### 4. Model lacks codebase context
The prompt only contains the problem statement — no file contents, no directory structure. For an agentless approach, the model must hallucinate the exact file paths and code structure from memory. This is extremely difficult for all but the most popular repos.

### 5. Patch syntax sensitivity
`git apply` is strict about whitespace, context lines, and line numbers. If the model's patch has even slightly wrong context lines (because it doesn't know the exact code at `base_commit`), `git apply` fails. The `|| echo "PATCH_APPLY_FAILED"` catches this but doesn't abort — tests run on unpatched code → guaranteed failure.

## Are Patches Syntactically Valid?

The model likely generates syntactically plausible diff blocks, but they fail at application because:
1. Wrong file paths (model doesn't know exact repo structure)
2. Wrong context lines (model doesn't see actual code)
3. Wrong line numbers

Even syntactically valid diffs with correct format (`---`, `+++`, `@@`, etc.) will fail `git apply` if context doesn't match.

## Actionable Improvements

### P0: Add codebase context to prompt
The agentless approach is nearly impossible without file context. Options:
- Include the actual file contents referenced in the issue (fetch from GitHub before prompting)
- Include the repo structure (`find . -name "*.py"` output)
- Include the failing test file contents

### P1: Fix checkout reliability
Replace shallow clone with targeted fetch:
```bash
git clone --depth=50 https://github.com/{repo}.git /workspace/repo
```
Or better, use SWE-bench's docker images which have pre-built environments.

### P2: Use SWE-bench docker images
SWE-bench provides per-instance Docker images with all dependencies pre-installed. This eliminates the `pip install -e .` failure mode. See `swebench/harness/docker_build.py`.

### P3: Add partial credit for patch quality
Instead of strict binary, give credit for:
- 0.2: Valid diff syntax
- 0.4: Patch applies cleanly
- 0.7: Some tests pass
- 1.0: All tests pass

### P4: Start with easier instances
Filter SWE-bench to "easy" instances (those with high pass rates in existing benchmarks). The full dataset includes very hard bugs that even frontier models solve at <5%.

### P5: Abort if patch doesn't apply
Change `|| echo "PATCH_APPLY_FAILED"` to `|| exit 1`. Currently, failed patches still run tests against unpatched code, wasting time.

## Expected Impact
P0 (adding codebase context) is essential — without it, this env will stay at reward=0. Even with context, agentless SWE-bench is hard. The paper uses GPT-OSS-120B (larger model) and runs ~40-50 steps. For Nano-30B, starting with easy instances (P4) and partial credit (P3) is more realistic.
