# Development Reference — Contributing, Tests, CI, New Recipes

Consolidated reference for contributing to tinker-cookbook: development setup, code style, testing, CI pipelines, and new recipe scaffolding.

---

## Development setup

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
uv sync --extra dev
pre-commit install
```

---

## Code style

- **Formatter/Linter:** ruff (line length: 100)
- **Type checker:** pyright
- **Pre-commit hooks** run automatically

```bash
uv run ruff check tinker_cookbook/
uv run ruff format tinker_cookbook/
uv run pyright tinker_cookbook/
pre-commit run --all-files
```

### Rules

- Explicit typing everywhere — avoid `Any` and `type: ignore`
- Builder pattern: config objects (`@chz.chz`) build runtime objects
- Config/runtime separation: configs are serializable, runtime objects are heavyweight
- Env objects are single-use (no reset)
- Dimension notation: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums)
- Use `safezip`, `timed`, `scope` helpers
- Use `ml_log.log_metrics` for metrics, `logtree` for transcripts

---

## PR process

1. Create a feature branch from `main`
2. Make changes with tests
3. Run `pre-commit run --all-files`
4. Open PR with clear description

CI runs pre-commit, pyright, and pytest on every PR.

---

## Testing

Two layers of tests.

### Reference files

- `tests/helpers.py` — `run_recipe()` helper
- `tests/conftest.py` — Pytest configuration and API key handling
- `.github/workflows/pytest.yaml` — Unit test CI
- `.github/workflows/smoke-test-recipes.yaml` — Smoke test CI
- `pyproject.toml` — Pytest configuration

### Test structure

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

### Unit tests (`*_test.py`)

Colocated with source code. No API key needed.

```bash
uv run pytest tinker_cookbook/
```

Conventions:
- File naming: `<module>_test.py` next to the code
- No network calls, no `TINKER_API_KEY`
- Fast (< 1s per test)
- Test picklability for distributed components

### Integration / smoke tests (`test_recipe_*.py`)

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

### Pytest markers

- `@pytest.mark.integration` — Requires API key, skipped locally without it
- `@pytest.mark.slow` — Long-running tests

`tests/conftest.py` auto-skips integration tests when `TINKER_API_KEY` is not set.

---

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

---

## Creating a new recipe

### Step 1: Understand the request

Determine:
- **Recipe name**: Directory/file name under `recipes/`
- **Training type**: SL, RL, DPO, distillation, or hybrid
- **Key details**: Model, dataset, environment, reward signal

### Step 2: Read existing recipes

Before writing code, read the most relevant recipe:
- **SL**: `tinker_cookbook/recipes/sl_basic.py` and `tinker_cookbook/recipes/chat_sl/train.py`
- **RL**: `tinker_cookbook/recipes/rl_basic.py` and `tinker_cookbook/recipes/math_rl/train.py`
- **DPO**: `tinker_cookbook/recipes/preference/dpo/train.py`
- **Distillation**: `tinker_cookbook/recipes/distillation/on_policy_distillation.py`
- **Multi-turn RL**: `tinker_cookbook/recipes/harbor_rl/train.py`

### Step 3: Follow conventions

#### File structure
```
tinker_cookbook/recipes/<recipe_name>/
├── __init__.py
├── train.py           # CLIConfig + cli_main
└── <env_or_data>.py   # Dataset/environment definitions
```

#### Required elements
1. `@chz.chz` config class with sensible defaults
2. `model_info.get_recommended_renderer_name()` — never hardcode
3. `cli_utils.check_log_dir()` before training
4. `checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async()` if loading checkpoints
5. Explicit typing — no `Any` or `type: ignore`
6. Auto-generated log paths

#### CLI pattern

```python
@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    learning_rate: float = 1e-4

async def cli_main(cli_config: CLIConfig):
    # Build full config, call training main

if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
```

Entry point: `python -m tinker_cookbook.recipes.<name>.train [chz overrides]`

#### Naming conventions
- Subscript suffixes: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums)
- Use `safezip`, `timed`, `scope` helpers
- Use `ml_log.log_metrics` for metrics, `logtree` for transcripts

### Step 4: Add tests

#### Smoke test (required)

Create `tests/recipes/test_recipe_<name>.py`:

```python
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        [
            "behavior_if_log_dir_exists=delete",
            # Override params for fast execution:
            # "groups_per_batch=4", "group_size=2",
        ],
    )
```

#### Unit tests (for testable components)

Place next to code: `tinker_cookbook/recipes/<name>/<component>_test.py`

### Step 5: Verify

```bash
python -c "from tinker_cookbook.recipes.<name> import train"
python -m tinker_cookbook.recipes.<name>.train --help
uv run pytest tests/recipes/test_recipe_<name>.py -v -x -s
```
