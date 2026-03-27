---
name: tinker-contributing
description: Guide for contributing to the tinker-cookbook repo — development setup, code style, type checking, PR process, and design conventions. Use when the user asks about how to contribute, set up the dev environment, code style, or project conventions.
---

# Contributing

Guide for developing and contributing to tinker-cookbook.

## Reference

Read `CONTRIBUTING.md` for the full guide.

## Development setup

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
uv sync --extra dev
pre-commit install
```

This installs dev dependencies and registers pre-commit hooks (`ruff` formatting/linting).

## Code style

- **Formatter/Linter:** [ruff](https://docs.astral.sh/ruff/) (line length: 100)
- **Type checker:** [pyright](https://github.com/microsoft/pyright)
- **Pre-commit hooks** run automatically on every commit

```bash
uv run ruff check tinker_cookbook/
uv run ruff format tinker_cookbook/
uv run pyright tinker_cookbook/
```

### Typing rules
- Use explicit types everywhere
- Avoid `Any` and `type: ignore` — prefer casting
- Prefer single types over union types
- Don't add convoluted generics just to satisfy the type checker

## Design conventions

### Builder pattern
Config objects build runtime objects:
- `SupervisedDatasetBuilder` → `SupervisedDataset`
- `RLDatasetBuilder` → `RLDataset`
- `EnvGroupBuilder` → group of `Env` objects

Config objects use `@chz.chz` decorator. They have a `__call__` method that builds the runtime object.

### Config/runtime separation
- **Config:** `@chz.chz` dataclasses, serializable, lightweight
- **Runtime:** Regular classes or dataclasses, heavyweight (datasets, clients)

### Training script organization
- **`tinker_cookbook/<module>/train.py`** — Main training loop with detailed `Config` (not CLI-constructable)
- **`tinker_cookbook/recipes/<name>/train.py`** — Launch script with `CLIConfig` from command line

### Async
- All methods that take nontrivial time should be async (especially in RL)
- Some beginner-oriented code (e.g., `sl_loop.py`) uses sync for simplicity

### Env lifecycle
- `Env` objects are single-use (no reset)
- Shared resources managed by `EnvGroupBuilder`, not individual `Env`s

### Dimension notation
Subscript suffixes on variable names:
- `_P` = problems, `_G` = groups, `_T` = tokens, `_D` = datums
- Example: `tokens_P_G_T[p][g][t]` = token t of group g of problem p
- Flattened: `tokens_PG_T` = problems and groups merged into one dimension

## PR process

1. Create a feature branch from `main`
2. Make changes with tests
3. Run `pre-commit run --all-files`
4. Open PR with clear description

CI runs pre-commit, pyright, and pytest on every PR.

## Testing

See the `/tinker-ci` skill for full testing details.

```bash
# Unit tests (no API key needed)
uv run pytest tinker_cookbook/

# Integration tests (requires TINKER_API_KEY)
uv run pytest tests/
```
