---
name: update-skills
description: Install, update, or uninstall globally-installed Tinker Claude Code skills. Use when the user wants to manage their Tinker skill installation.
argument-hint: "[install|update|uninstall|status]"
---

# Manage Tinker Skills Installation

Help the user install, update, or uninstall Tinker Claude Code skills globally.

## How global skills work

Tinker skills live in this repo at `.claude/skills/`. The install script (`.claude/install-skills.sh`) symlinks them into `~/.claude/skills/` with a `tinker-` prefix so they're available in any project.

There are two installation modes:
1. **From a local clone** — symlinks point into the user's clone, so `git pull` updates skills automatically
2. **Standalone (via curl)** — a shallow clone is cached at `~/.claude/tinker-cookbook/`, and re-running the installer updates it

## Step 1: Determine what the user wants

- **install** or **update**: Run the install script (same command for both — it refreshes existing symlinks)
- **uninstall** or **remove**: Run the install script with `--remove`
- **status**: Check what's currently installed

## Step 2: Check current state

```bash
# See what tinker skills are currently symlinked
ls -la ~/.claude/skills/tinker-* 2>/dev/null || echo "No tinker skills installed"
```

## Step 3: Execute the action

### Install / Update

If the user has a local clone of tinker-cookbook (check if `.claude/install-skills.sh` exists in the current repo or a known path):

```bash
bash .claude/install-skills.sh
```

If not, use the one-liner:

```bash
curl -fsSL https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/.claude/install-skills.sh | bash
```

For updates from a local clone, pull first:

```bash
git pull && bash .claude/install-skills.sh
```

For standalone installs, the script handles the update internally (pulls the cached clone).

### Uninstall

```bash
bash .claude/install-skills.sh --remove
```

Or standalone:

```bash
curl -fsSL https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/.claude/install-skills.sh | bash -s -- --remove
```

### Status

```bash
# List installed tinker skills
ls ~/.claude/skills/tinker-* 2>/dev/null | xargs -I{} basename {} | sort

# Check where symlinks point (local clone vs cached)
readlink ~/.claude/skills/tinker-sft 2>/dev/null || echo "tinker-sft not installed"
```

## Included skills (Layers 0–3)

| Prefix | Skills |
|--------|--------|
| Fundamentals | `tinker-setup`, `tinker-models`, `tinker-hyperparams`, `tinker-logging` |
| SDK | `tinker-sdk`, `tinker-types`, `tinker-cli` |
| Primitives | `tinker-renderers`, `tinker-environments`, `tinker-weights`, `tinker-completers`, `tinker-checkpoints`, `tinker-evals`, `tinker-datasets` |
| Recipes | `tinker-sft`, `tinker-grpo`, `tinker-distillation`, `tinker-dpo`, `tinker-rlhf`, `tinker-multiturn-rl` |

Dev-only skills (`ci`, `contributing`, `new-recipe`, `manage-skills`) are excluded from global install — they're only available when working inside the tinker-cookbook repo itself.
