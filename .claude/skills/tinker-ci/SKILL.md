---
name: tinker-ci
description: Guide for testing conventions and CI pipelines — unit tests, integration smoke tests, pytest markers, and GitHub Actions workflows. Use when the user asks about testing, CI, running tests, or adding tests for a recipe.
---

# Testing & CI

The repo has two layers of testing and two CI workflows.

## Reference

Read these for details:
- `tests/helpers.py` — `run_recipe()` helper for smoke tests
- `tests/conftest.py` — Pytest configuration and API key handling
- `tests/recipes/` — Existing recipe smoke tests
- `.github/workflows/pytest.yaml` — Unit test CI (every PR)
- `.github/workflows/smoke-test-recipes.yaml` — Smoke test CI (daily)
- `CONTRIBUTING.md` — Development setup and test commands
- `pyproject.toml` — Pytest configuration (testpaths, markers, file patterns)

## Test structure

```
tinker-cookbook/
├── tinker_cookbook/
│   ├── renderers/parsing_test.py     # Unit tests: *_test.py next to source
│   ├── recipes/math_rl/math_env_test.py
│   └── ...
└── tests/
    ├── conftest.py                   # Skips integration tests without API key
    ├── helpers.py                    # run_recipe() helper
    └── recipes/
        ├── test_recipe_chat_sl.py    # Integration tests: test_recipe_*.py
        ├── test_recipe_dpo.py
        └── ...
```

## Unit tests (`*_test.py`)

Colocated with source code. Run without API key.

```bash
uv run pytest tinker_cookbook/
```

**Conventions:**
- File naming: `<module>_test.py` next to the code it tests
- No network calls, no `TINKER_API_KEY` required
- Fast (< 1s per test)
- Use standard pytest features (fixtures, parametrize, marks)
- Test picklability for components used in distributed rollout

**Example:** `tinker_cookbook/renderers/parsing_test.py`

## Integration / smoke tests (`test_recipe_*.py`)

Live in `tests/recipes/`. Require `TINKER_API_KEY`. Verify recipes can run.

```bash
# Run all integration tests
uv run pytest tests/ -v -x -s

# Run a specific recipe test
uv run pytest tests/recipes/test_recipe_chat_sl.py -v -x -s
```

**Conventions:**
- File naming: `tests/recipes/test_recipe_<name>.py`
- Mark with `@pytest.mark.integration`
- Use `run_recipe()` from `tests/helpers.py`
- `run_recipe()` passes `max_steps=2` by default — recipe runs 2 training steps and exits
- Always pass `behavior_if_log_dir_exists=delete` to avoid CI conflicts
- Override batch sizes to small values for fast execution

**Template:**

```python
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_my_recipe():
    run_recipe(
        "tinker_cookbook.recipes.my_recipe.train",
        [
            "behavior_if_log_dir_exists=delete",
            "groups_per_batch=4",
        ],
    )
```

### How `run_recipe()` works
1. Launches `uv run python -m <module> <args> max_steps=2` as a subprocess
2. Streams stdout in real time for CI debuggability
3. Waits for clean exit (exit code 0) within timeout (default: 1800s)
4. Fails if process exits non-zero or times out

## Pytest markers

Defined in `pyproject.toml`:
- `@pytest.mark.integration` — Requires API key, skipped locally without `TINKER_API_KEY`
- `@pytest.mark.slow` — Long-running tests

`tests/conftest.py` auto-skips integration tests when `TINKER_API_KEY` is not set (fails on CI if missing).

## CI workflows

### `pytest.yaml` — Unit tests (every PR/push to main)
```
Trigger: push to main, pull requests
Runs: uv run pytest tinker_cookbook/
Requires: HF_TOKEN (for tokenizer access)
```

### `smoke-test-recipes.yaml` — Integration tests (daily + manual)
```
Trigger: daily at 6am UTC, manual dispatch
Runs: Each test_recipe_*.py in parallel (matrix strategy)
Requires: TINKER_API_KEY, HF_TOKEN
Timeout: 20 min per recipe
Concurrency: 1 (avoid API contention)
```

Adding `tests/recipes/test_recipe_<name>.py` is all that's needed — CI auto-discovers it.

## Running pre-commit checks

```bash
uv run ruff check tinker_cookbook/
uv run ruff format tinker_cookbook/
uv run pyright tinker_cookbook/
pre-commit run --all-files
```
