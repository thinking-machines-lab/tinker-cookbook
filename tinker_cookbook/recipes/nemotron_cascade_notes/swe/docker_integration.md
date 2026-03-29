# SWE Docker Integration Investigation (2026-03-27)

## Summary

R2E-Gym Docker images work end-to-end via Modal sandboxes. Two bugs were found and
fixed that were preventing correct test execution in both agentless and agentic SWE
environments.

## What Works

### R2E-Gym Docker Images via Modal

- **Image pull**: Modal can pull R2E-Gym images from Docker Hub (`namanjain12/*`
  namespace). First pull takes ~30-60s; subsequent runs use Modal's cache.
- **Repo at /testbed**: All images have the repository pre-installed at `/testbed`
  with dependencies already set up.
- **Python environment**: Images include the correct Python version (e.g. 3.7.9 for
  older repos) and all project-specific dependencies.
- **Test execution**: pytest works correctly in the sandbox. Tests properly fail on
  unpatched code and pass on patched code.

Verified with multiple repos: `orange3`, `coveragepy`. All 4578 instances in
`R2E-Gym/R2E-Gym-Subset` have valid `docker_image` and test files.

### Dataset Schema

`R2E-Gym/R2E-Gym-Subset` fields used:
- `repo_name`: e.g. "orange3", "aiohttp"
- `docker_image`: e.g. "namanjain12/orange3_final:2d9617bd..."
- `commit_hash`: target commit
- `problem_statement`: natural-language issue description
- `execution_result_content`: JSON string containing:
  - `test_file_names`: list of test filenames (e.g. ["test_1.py"])
  - `test_file_codes`: list of test file source code
- `parsed_commit_content`: JSON with file diffs (ground truth patch)

Note: `instance_id` field does NOT exist in this dataset. The code correctly
falls back to `f"{repo}:{base_commit[:8]}"`.

Unique repos in dataset: orange3, coveragepy, numpy, datalad, pyramid, aiohttp,
scrapy, tornado, pillow, pandas (10 repos, 4578 instances total).

### Agentless SWE (swe_rl_env.py)

The `run_swe_test_in_modal()` function works with R2E-Gym images:
1. Creates Modal sandbox with `modal.Image.from_registry(docker_image)`
2. Writes test files from `execution_result_content` into `/testbed/r2e_tests/`
3. Applies patch via `git apply`
4. Runs pytest on test files
5. Returns binary pass/fail

### Agentic SWE (swe_agentic_env.py)

The `ModalSandbox.create()` path works with R2E-Gym images:
1. Creates persistent sandbox with R2E-Gym image
2. `write_file()` works for writing test files
3. `run_command()` works for executing pytest
4. Sandbox cleanup works correctly

## Bugs Found and Fixed

### Bug 1: `--timeout=60` flag breaks pytest (FIXED)

**Impact**: All R2E-Gym test execution was silently broken.

R2E-Gym Docker images include `pytest 7.4.4` but NOT `pytest-timeout`. The
`--timeout=60` flag caused pytest to exit with an "unrecognized arguments" error
(exit code 4). Because stderr was suppressed with `2>/dev/null`, the error was
invisible, and the test was counted as "failed" for the wrong reason.

**Files affected**:
- `swe_rl_env.py`: Both R2E-Gym path (line ~200) and legacy path (line ~241)
- `swe_agentic_env.py`: `SWEAgenticReward.__call__()` (line ~164)

**Fix**: Removed `--timeout=60` from all pytest commands. Test timeouts are
handled at the sandbox/Modal level instead (sandbox_timeout, command_timeout).

### Bug 2: Failed `git apply` silently continued (FIXED)

**Impact**: When a patch failed to apply, tests ran on unpatched code, wasting
time and returning misleading results.

The `run_swe_test_in_modal()` R2E-Gym path had:
```bash
git apply - 2>/dev/null || echo "PATCH_APPLY_FAILED"
```
This printed a message but continued execution, running tests on unpatched code.

**Fix**: Changed to:
```bash
git apply - || { echo "PATCH_APPLY_FAILED"; exit 1; }
```
Now a failed patch application immediately exits with failure.

Also changed `2>/dev/null` to `2>&1` on pytest commands so error messages are
captured in the output rather than being silently swallowed.

## Docker Hub Rate Limiting

Anonymous Docker Hub pulls: 100 pulls/6h. Authenticated: 200 pulls/6h.
With group_size=64, each training step could hit rate limits quickly.

**Mitigations**:
- Modal caches images after first pull, so repeated use of the same image is free.
- The 10 repos in R2E-Gym-Subset mean ~10 unique image bases (with different tags
  per commit, but Modal may deduplicate layers).
- For large-scale training, consider: authenticating Docker Hub in Modal config,
  mirroring images to a private registry, or pre-pulling all images.

## Modal Configuration

No special Modal configuration needed beyond authentication (`modal token new`).
The code creates apps with `create_if_missing=True` and uses `Image.from_registry()`
which handles pull/cache automatically.

Sandbox defaults:
- `timeout=300` (agentless), `timeout=600` (agentic)
- Working directory: `/testbed` (R2E-Gym mode), `/workspace/repo` (legacy mode)

## Remaining Work

1. **Full RL training test**: The execution-based reward path is now correctly wired
   up. A full RL training run with `reward_mode="execution"` should be tested with
   a small group_size (2-4) to verify end-to-end training signal.

2. **Image pre-warming**: First pull of each unique image tag adds ~30-60s latency.
   For training, consider a warm-up step that pre-pulls all unique images.

3. **Test timeout at sandbox level**: Without `--timeout=60`, a runaway test could
   block indefinitely. The sandbox-level timeout (300s/600s) provides a safety net,
   but individual test timeouts could be added by installing `pytest-timeout` in a
   derived image or using `timeout` command wrapper.
