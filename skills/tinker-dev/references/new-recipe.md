# Create a New Training Recipe

Step-by-step guide for scaffolding a new recipe.

## Step 1: Understand the request

Ask the user:
- **Recipe name**: Directory/file name under `recipes/`
- **Training type**: SL, RL, DPO, distillation, or hybrid
- **Key details**: Model, dataset, environment, reward signal

## Step 2: Read existing recipes

Before writing code, read the most relevant recipe:
- **SL**: `tinker_cookbook/recipes/sl_basic.py` and `tinker_cookbook/recipes/chat_sl/train.py`
- **RL**: `tinker_cookbook/recipes/rl_basic.py` and `tinker_cookbook/recipes/math_rl/train.py`
- **DPO**: `tinker_cookbook/recipes/preference/dpo/train.py`
- **Distillation**: `tinker_cookbook/recipes/distillation/on_policy_distillation.py`
- **Multi-turn RL**: `tinker_cookbook/recipes/harbor_rl/train.py`

## Step 3: Follow conventions

### File structure
```
tinker_cookbook/recipes/<recipe_name>/
├── __init__.py
├── train.py           # CLIConfig + cli_main
└── <env_or_data>.py   # Dataset/environment definitions
```

### Required elements
1. `@chz.chz` config class with sensible defaults
2. `model_info.get_recommended_renderer_name()` — never hardcode
3. `cli_utils.check_log_dir()` before training
4. `checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async()` if loading checkpoints
5. Explicit typing — no `Any` or `type: ignore`
6. Auto-generated log paths

### Naming conventions
- Subscript suffixes: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums)
- Use `safezip`, `timed`, `scope` helpers
- Use `ml_log.log_metrics` for metrics, `logtree` for transcripts

### Entry point
```bash
python -m tinker_cookbook.recipes.<recipe_name>.train [chz overrides]
```

## Step 4: Add tests

### Smoke test (required)

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

### Unit tests (for testable components)

Place next to code: `tinker_cookbook/recipes/<name>/<component>_test.py`

## Step 5: Verify

```bash
python -c "from tinker_cookbook.recipes.<name> import train"
python -m tinker_cookbook.recipes.<name>.train --help
uv run pytest tests/recipes/test_recipe_<name>.py -v -x -s
```
