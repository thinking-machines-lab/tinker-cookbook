---
name: new-recipe
description: Scaffold a new training recipe for the Tinker cookbook following repo conventions. Use when the user wants to create a new recipe, training script, or experiment.
disable-model-invocation: true
argument-hint: "[recipe-name]"
---

# Create a New Training Recipe

Scaffold a new training recipe in `tinker_cookbook/recipes/` following repo conventions.

## Step 1: Understand the request

Ask the user:
- **Recipe name**: What to call it (becomes the directory/file name under `recipes/`)
- **Training type**: SL, RL, DPO, distillation, or hybrid
- **Key details**: Model, dataset, environment, reward signal, etc.

## Step 2: Read existing recipes for patterns

Before writing any code, read the most relevant existing recipe:
- **SL-based**: Read `tinker_cookbook/recipes/sl_basic.py` and `tinker_cookbook/recipes/chat_sl/train.py`
- **RL-based**: Read `tinker_cookbook/recipes/rl_basic.py` and `tinker_cookbook/recipes/math_rl/train.py`
- **DPO-based**: Read `tinker_cookbook/recipes/preference/dpo/train.py`
- **Distillation-based**: Read `tinker_cookbook/recipes/distillation/on_policy_distillation.py`
- **Multi-turn RL**: Read `tinker_cookbook/recipes/harbor_rl/train.py`

Also read `CLAUDE.md` for conventions.

## Step 3: Follow repo conventions

Every recipe MUST follow these patterns:

### File structure
```
tinker_cookbook/recipes/<recipe_name>/
├── __init__.py        # Empty or minimal
├── train.py           # Main entry point with CLIConfig + cli_main
└── <env_or_data>.py   # Dataset/environment definitions (if needed)
```

Or for simple recipes: `tinker_cookbook/recipes/<recipe_name>.py`

### CLI pattern (use `@chz.chz` for config)
```python
@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    learning_rate: float = 1e-4
    # ... all configurable parameters with defaults

async def cli_main(cli_config: CLIConfig):
    # Build full config from CLI config
    # Call training main function

if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
```

### Required elements
1. **`@chz.chz` config class** with sensible defaults
2. **`model_info.get_recommended_renderer_name(model_name)`** for renderer — never hardcode
3. **`cli_utils.check_log_dir()`** before training to avoid clobbering
4. **`checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async()`** if loading checkpoints
5. **Explicit typing** — no `Any` or `type: ignore`
6. **Auto-generated log paths** with model name, hyperparams, and timestamp

### Naming conventions
- Subscript suffixes for tensors: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums)
- Use `safezip`, `timed`, `scope` helpers where appropriate
- Use `ml_log.log_metrics` for metrics, `logtree` for transcripts

### Entry point
Recipe must be runnable as:
```bash
python -m tinker_cookbook.recipes.<recipe_name>.train [chz overrides]
```

## Step 4: Create the recipe

Write the recipe files following the patterns above. Place them in `tinker_cookbook/recipes/$ARGUMENTS/`.

## Step 5: Verify

- Ensure the recipe is importable: `python -c "from tinker_cookbook.recipes.<name> import train"`
- Check that CLI help works: `python -m tinker_cookbook.recipes.<name>.train --help`
- For unit-testable components, add tests in `tinker_cookbook/tests/`
