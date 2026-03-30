---
name: dev
description: Contributing to tinker-cookbook — development setup, code style, creating new recipes, testing, CI, and skill management. Use when the user wants to contribute code, create a new recipe, run tests, understand CI pipelines, or manage skills in this repo.
---

# Development & Contributing

Everything for contributing to the tinker-cookbook repo.

## Development setup

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
uv sync --extra dev
pre-commit install
```

## Code style

- **Formatter/Linter:** ruff (line length: 100)
- **Type checker:** pyright
- **Pre-commit hooks** run automatically

```bash
uv run ruff check tinker_cookbook/
uv run ruff format tinker_cookbook/
uv run pyright tinker_cookbook/
```

### Rules
- Explicit typing everywhere — avoid `Any` and `type: ignore`
- Builder pattern: config objects (`@chz.chz`) build runtime objects
- Config/runtime separation: configs are serializable, runtime objects are heavyweight
- Env objects are single-use (no reset)
- Dimension notation: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums)

## Creating a new recipe

### File structure

```
tinker_cookbook/recipes/<recipe_name>/
├── __init__.py
├── train.py           # Main entry point with CLIConfig + cli_main
└── <env_or_data>.py   # Dataset/environment definitions
```

### Required patterns

1. `@chz.chz` config class with sensible defaults
2. `model_info.get_recommended_renderer_name(model_name)` — never hardcode
3. `cli_utils.check_log_dir()` before training
4. `checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async()` if loading checkpoints
5. Explicit typing — no `Any` or `type: ignore`

### CLI pattern

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

For the full recipe scaffolding guide with smoke test templates, read `references/new-recipe.md`.

## Testing

Two layers:

### Unit tests (`*_test.py`)

Colocated with source code. No API key needed.
```bash
uv run pytest tinker_cookbook/
```

### Integration / smoke tests (`test_recipe_*.py`)

Live in `tests/recipes/`. Require `TINKER_API_KEY`.
```bash
uv run pytest tests/recipes/test_recipe_<name>.py -v -x -s
```

Template:
```python
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_my_recipe():
    run_recipe(
        "tinker_cookbook.recipes.my_recipe.train",
        ["behavior_if_log_dir_exists=delete", "groups_per_batch=4"],
    )
```

`run_recipe()` launches the module with `max_steps=2` and verifies clean exit.

For the full testing guide (CI workflows, pytest markers, `run_recipe()` details), read `references/ci.md`.

## PR process

1. Create a feature branch from `main`
2. Make changes with tests
3. Run `pre-commit run --all-files`
4. Open PR with clear description

CI runs pre-commit, pyright, and pytest on every PR.

## Existing recipe patterns

Read these for reference before writing new recipes:
- **SL**: `tinker_cookbook/recipes/sl_basic.py`, `tinker_cookbook/recipes/chat_sl/train.py`
- **RL**: `tinker_cookbook/recipes/rl_basic.py`, `tinker_cookbook/recipes/math_rl/train.py`
- **DPO**: `tinker_cookbook/recipes/preference/dpo/train.py`
- **Distillation**: `tinker_cookbook/recipes/distillation/on_policy_distillation.py`
- **Multi-turn RL**: `tinker_cookbook/recipes/harbor_rl/train.py`

## Code references

- `CONTRIBUTING.md` — Full contributing guide
- `tests/helpers.py` — `run_recipe()` helper
- `tests/conftest.py` — Pytest configuration
- `.github/workflows/pytest.yaml` — Unit test CI
- `.github/workflows/smoke-test-recipes.yaml` — Smoke test CI
