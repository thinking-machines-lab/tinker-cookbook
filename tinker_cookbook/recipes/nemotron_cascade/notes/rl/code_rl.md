# Code RL Environment Analysis

## Status: LOW REWARD (0.094 with g=16) — Working but needs tuning

## Architecture
- Default dataset: MBPP (sanitized) from HuggingFace
- Paper uses: AtCoder/Codeforces competitive programming (3.5K hard problems)
- Execution: Modal sandbox with bash script
- Reward: Strict binary (all tests pass = 1, else = 0)

## Key Observations

### 1. Dataset Mismatch: MBPP vs competitive programming
The paper uses 3.5K *hard* competitive programming problems. MBPP is an easier function-writing benchmark. This is actually advantageous for getting initial signal — MBPP should be *easier* than the paper's data, so 0.094 reward suggests deeper issues.

### 2. Two test formats, different execution paths
- **Assertion-based** (MBPP): `assert func(x) == y` — solution + assertions concatenated and run
- **Stdin/stdout** (competitive programming): read from stdin, diff expected output

For MBPP assertion tests, the execution script does:
```bash
cat /tmp/solution.py /tmp/tests.py > /tmp/run.py
timeout 30 python3 /tmp/run.py
```
This works only if the solution defines the function before the assertions. If the model wraps code in `if __name__ == "__main__"` or uses different function names, assertions fail silently.

### 3. Code extraction may miss solutions
`extract_code` looks for fenced code blocks (```python...```). If the model outputs raw code without fences (common for shorter solutions), extraction returns None → reward=0 regardless of correctness.

### 4. Prompt design for MBPP
For assertion-based tests, the prompt says:
> "Write a Python function that satisfies the following requirements."

But it shows example assertions like `assert func(3) == 9`. The model needs to infer the function name from the assertions. If the MBPP `prompt` field already names the function, this works. If not, name mismatch = guaranteed failure.

### 5. Paper uses 118K max tokens, we likely use 49K
The paper specifies 118K for Code RL. With the default CLI config of 49K, long reasoning chains may get truncated, triggering the overlong penalty.

## Actionable Improvements

### P0: Verify MBPP assertion execution locally
Run a few MBPP examples through `build_execution_script` manually to confirm:
- The function names in the prompt match the assertion function names
- The concatenated solution+test file actually runs
- The timeout is sufficient

### P1: Improve code extraction robustness
Add fallback for unfenced code: if no fenced block found, try the full response as Python code (after stripping any markdown or natural language preamble).

### P2: Set max_tokens=118K for code_rl
Paper specifically uses 118K for this stage. The CLI default is 49K.

### P3: Add partial credit reward
Instead of strict binary (all tests or nothing), use fraction of tests passed. This gives smoother signal:
```python
reward = passed_count / total_count
```
The paper uses strict binary, but for LoRA training with small batch, partial credit may learn faster.

### P4: Switch to paper's competitive programming dataset
The paper's dataset is `nvidia/Nemotron-Cascade-2-RL-data` Code-RL subset (if it exists). Check if this split is available on HuggingFace. If not, LiveCodeBench is a closer proxy than MBPP.

### P5: Increase group_size
With 0.094 reward at g=16, most groups have 0 or 1 passing rollouts. Increasing to g=32 or g=64 would give more mixed groups and better GRPO signal. The paper uses g=16 with batch=128, but their model is stronger after full multi-stage training.

## Expected Impact
P0-P2 should increase reward from ~0.09 to ~0.2-0.3 by fixing execution issues and allowing longer reasoning. P3 (partial credit) could further improve gradient quality even at low absolute reward.
