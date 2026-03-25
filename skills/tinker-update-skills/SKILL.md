---
name: tinker-update-skills
description: Install, update, or uninstall globally-installed Tinker Claude Code skills. Use when the user wants to manage their Tinker skill installation.
---

# Manage Tinker Skills Installation

Help the user install, update, or uninstall Tinker Claude Code skills globally.

## Installation methods

### Method 1: Plugin marketplace (recommended)

```bash
# In Claude Code, run:
/plugin marketplace add thinking-machines-lab/tinker-cookbook
```

Then browse and install the `tinker-training` plugin. This is the cleanest approach — Claude Code handles updates automatically.

### Method 2: Symlink installer

For manual control or environments without plugin support:

```bash
# One-liner (no clone needed):
curl -fsSL https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/.claude/install-skills.sh | bash

# From a local clone:
bash .claude/install-skills.sh

# Remove:
bash .claude/install-skills.sh --remove
```

Two modes: standalone (sparse-clones to `~/.claude/tinker-cookbook/`) or local clone (symlinks directly, `git pull` updates skills).

## Checking status

```bash
# List installed tinker skills
ls ~/.claude/skills/tinker-* 2>/dev/null | xargs -I{} basename {} | sort

# Check where symlinks point (local clone vs cached)
readlink ~/.claude/skills/tinker-sft 2>/dev/null || echo "tinker-sft not installed"
```

## Plugin bundles

| Bundle | Skills |
|--------|--------|
| **tinker-training** | All Layer 0–3 skills: setup, models, hyperparams, logging, sdk, types, cli, renderers, environments, weights, completers, checkpoints, evals, datasets, sft, grpo, distillation, dpo, rlhf, multiturn-rl, update-skills |
| **tinker-dev** | Dev skills: ci, contributing, new-recipe, manage-skills |

## Updating

- **Plugin method**: Claude Code updates plugins automatically on next activation.
- **Symlink method (local clone)**: `git pull && bash .claude/install-skills.sh`
- **Symlink method (standalone)**: Re-run the curl one-liner — it pulls the cached clone.
