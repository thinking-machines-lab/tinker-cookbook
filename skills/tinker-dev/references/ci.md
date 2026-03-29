# Testing & CI

Complete reference for testing conventions and CI pipelines.

## Reference

- `tests/helpers.py` — `run_recipe()` helper
- `tests/conftest.py` — Pytest configuration and API key handling
- `.github/workflows/pytest.yaml` — Unit test CI
- `.github/workflows/smoke-test-recipes.yaml` — Smoke test CI
- `pyproject.toml` — Pytest configuration

## Test structure

```
tinker-cookbook/
├── tinker_cookbook/
│   ├── renderers/parsing_test.py      # Unit tests: *_test.py next to source
│   ├── recipes/math_rl/math_env_test.py
│   └── ...
└── tests/
    ├── conftest.py                    # Skips integration tests without API key
    ├── helpers.py                     # run_recipe() helper
    └── recipes/
        ├── test_recipe_chat_sl.py     # Integration: test_recipe_*.py
        └── ...
```

## Unit tests

Colocated with source code. No API key needed.

```bash
uv run pytest tinker_cookbook/
```

Conventions:
- File naming: `<module>_test.py` next to the code
- No network calls, no `TINKER_API_KEY`
- Fast (< 1s per test)
- Test picklability for distributed components

## Integration / smoke tests

Live in `tests/recipes/`. Require `TINKER_API_KEY`.

```bash
uv run pytest tests/ -v -x -s
uv run pytest tests/recipes/test_recipe_chat_sl.py -v -x -s
```

Conventions:
- File naming: `tests/recipes/test_recipe_<name>.py`
- Mark with `@pytest.mark.integration`
- `run_recipe()` passes `max_steps=2` by default
- Always pass `behavior_if_log_dir_exists=delete`
- Override batch sizes to small values

### How `run_recipe()` works

1. Launches `uv run python -m <module> <args> max_steps=2`
2. Streams stdout in real time
3. Waits for clean exit within timeout (default: 1800s)
4. Fails on non-zero exit or timeout

## Pytest markers

- `@pytest.mark.integration` — Requires API key, skipped locally without it
- `@pytest.mark.slow` — Long-running tests

`tests/conftest.py` auto-skips integration tests when `TINKER_API_KEY` is not set.

## CI workflows

### `pytest.yaml` — Unit tests (every PR/push)
- Trigger: push to main, pull requests
- Runs: `uv run pytest tinker_cookbook/`
- Requires: `HF_TOKEN`

### `smoke-test-recipes.yaml` — Integration (daily + manual)
- Trigger: daily at 6am UTC, manual dispatch
- Runs: Each `test_recipe_*.py` in parallel (matrix strategy)
- Requires: `TINKER_API_KEY`, `HF_TOKEN`
- Timeout: 20 min per recipe
- Concurrency: 1

Adding `tests/recipes/test_recipe_<name>.py` is all that's needed — CI auto-discovers it.

## Pre-commit checks

```bash
uv run ruff check tinker_cookbook/
uv run ruff format tinker_cookbook/
uv run pyright tinker_cookbook/
pre-commit run --all-files
```
