# Contributing to Tinker

This project is built in the spirit of open science and collaborative development. We believe that the best tools emerge through community involvement and shared learning.

## How contributions work

Tinker is developed across an internal monorepo and this public GitHub repository. Changes sync bidirectionally on a daily cadence:

- **Internal -> GitHub**: Internal commits are synced out to the public repo daily.
- **GitHub -> Internal**: Merged PRs on GitHub are synced back into the internal monorepo daily.

This means your contributions land in both places automatically — you just open a PR on GitHub.

## Contribution process

### 1. Find or propose work

- Check [open issues](https://github.com/thinking-machines-lab/tinker-cookbook/issues) for bugs, feature requests, or improvements.
- For small fixes (typos, doc clarifications, bug fixes with obvious solutions), go straight to a PR.
- For larger changes (new recipes, API changes, architectural decisions), open an issue first to discuss the approach. This saves everyone time.

### 2. Set up your environment

```bash
git clone git@github.com:thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .
```

You'll need a Tinker API key for anything that hits the service. Sign up at [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker/) and export it as `TINKER_API_KEY`.

### 3. Make your changes

Create a branch from `main`:

```bash
git checkout -b your-branch-name
```

Follow the conventions below when writing code.

### 4. Test

```bash
pytest tinker-cookbook/tests/test_renderers.py
pytest tinker-cookbook/tests/test_utils.py
```

### 5. Submit a PR

- Keep PRs focused. One logical change per PR.
- Write a clear description: what changed and why.
- Link to the relevant issue if there is one.

A maintainer will review your PR. We aim to respond within a few business days. Once approved and merged, your changes will sync into the internal monorepo in the next daily sync.

## Code conventions

### Organization of training scripts

The codebase is designed around three goals:

1. **Low barrier to entry**: dead simple to run something and see numbers go up.
2. **Extensible**: pass in custom datasets, evals, and control all the hyperparameters.
3. **Science-friendly**: easy to run sweeps and analyze results.

The structure:

- A main training function (e.g., [rl/train.py](tinker-cookbook/rl/train.py), [supervised/train.py](tinker-cookbook/supervised/train.py)) contains the main loop with a detailed `Config` object.
- Launch scripts (e.g., [recipes/math_rl/train.py](tinker-cookbook/recipes/math_rl/train.py)) assemble training configs and expose a smaller `CLIConfig` for the command line.

Config members that specify datasets and evals should be `chz` configs (with `.build()`) or callables (we recommend `functools.partial`), keeping configs serializable for sweeps.

### Typing

Use typing wherever possible. Avoid `Any` and `type: ignore`; prefer casting. Don't write convoluted generics or overly verbose code just to satisfy the type checker. Prefer single types over union types.

### Async

Async is useful for RL, where it allows many parallel queries (e.g., sampling calls). All methods that take nontrivial time in RL interfaces (like `Env`) should be async. Some non-RL code (e.g., [recipes/sl_loop.py](tinker-cookbook/recipes/sl_loop.py)) intentionally avoids async to stay beginner-friendly.

### Classes and the builder pattern

There are a lot of classes, but they follow the builder pattern:

- `SupervisedDatasetBuilder` builds a `SupervisedDataset`.
- `RLDatasetBuilder` builds an `RLDataset`, which generates `EnvGroupBuilder` objects, which each generate `Env` objects.

Config objects use `chz` (like dataclasses with extra features for configs). Runtime objects use dataclasses or regular classes.

### Notation

Subscript suffixes indicate tensor shapes: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums). Example: `tokens_P_G_T[p][g][t]` is a single token. Arrays may be ragged; `tokens_PG_T` means the `P` and `G` dimensions are flattened.

### Envs

An `Env` is an RL environment, roughly analogous to OpenAI Gym but single-use (no `reset`). Discard after a rollout. Shared resources should be maintained by whatever object creates the envs. Created via `EnvGroupBuilder`.

## What we're looking for

Good contributions to Tinker tend to be:

- **New recipes** that demonstrate real fine-tuning workflows (math reasoning, tool use, preference learning, etc.)
- **Improved documentation** — clearer explanations, better examples, fixed errors
- **Bug fixes** with minimal, focused diffs
- **Evaluation integrations** — new evaluators or benchmark support

## Contact

Questions or feedback: tinker@thinkingmachines.ai

For rendered documentation: [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai)
