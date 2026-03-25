---
name: tinker-update-skills
description: Install, update, or uninstall Tinker Claude Code skills. Use when the user wants to manage their Tinker skill installation.
---

# Manage Tinker Skills Installation

Help the user install, update, or uninstall Tinker Claude Code skills.

## Install

```
/plugin marketplace add thinking-machines-lab/tinker-cookbook
```

Then install the `tinker-training` plugin bundle. Skills update automatically from the repo.

## Plugin bundles

| Bundle | Skills |
|--------|--------|
| **tinker-training** | All user-facing skills: setup, models, hyperparams, logging, sdk, types, cli, renderers, environments, weights, completers, checkpoints, evals, datasets, sft, grpo, distillation, dpo, rlhf, multiturn-rl, update-skills |
| **tinker-dev** | Development skills: ci, contributing, new-recipe, manage-skills |

## Uninstall

```
/plugin marketplace remove thinking-machines-lab/tinker-cookbook
```
