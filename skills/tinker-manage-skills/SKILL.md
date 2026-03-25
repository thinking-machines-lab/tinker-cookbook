---
name: tinker-manage-skills
description: Create, update, or organize Claude Code skills in this repo. Use when adding a new skill, reviewing existing skills for consistency, or maintaining the skill taxonomy.
---

# Manage Claude Code Skills

This meta-skill governs how skills are created and maintained in the tinker-cookbook repo.

## Skill taxonomy

All skills in `skills/` are organized into 5 layers:

### Layer 0: Fundamentals (`setup`, `models`, `hyperparams`, `logging`)
**Scope:** Getting started, model selection, hyperparameter guidance, training output analysis. Cross-cutting concerns needed before touching any code.
**Auto-invocation:** Yes — triggers when users ask about setup, models, hyperparameters, or debugging.
**Key principle:** These inform all other layers. Reference `docs/`, `README.md`, `tinker_cookbook/hyperparam_utils.py`.

### Layer 1: Tinker SDK (`tinker-sdk`, `tinker-types`, `tinker-cli`)
**Scope:** Raw Tinker Python SDK APIs — ServiceClient, TrainingClient, SamplingClient, RestClient, types, errors, and CLI commands.
**Auto-invocation:** Yes — triggers when users ask about Tinker API basics or CLI usage.
**Key principle:** Reference `docs/api-reference/` for authoritative API docs.

### Layer 2: Cookbook Primitives (`renderers`, `environments`, `weights`, `completers`, `checkpoints`, `evals`, `datasets`)
**Scope:** Building blocks in `tinker_cookbook/` — renderers, RL environments, weight lifecycle, completers, checkpointing, evaluators, dataset construction.
**Auto-invocation:** Yes — triggers when users ask about specific primitives.
**Key principle:** Reference source code in `tinker_cookbook/` and docs in `docs/`.

### Layer 3: Algorithm / Task Recipes (`sft`, `grpo`, `distillation`, `dpo`, `rlhf`, `multiturn-rl`)
**Scope:** End-to-end training workflows built on Layer 1 + Layer 2.
**Auto-invocation:** Yes — triggers when users want to set up a specific training method.
**Key principle:** Reference recipes in `tinker_cookbook/recipes/` and defer primitive details to Layer 2 skills.

### Layer 4: Repo Development (`new-recipe`, `ci`, `contributing`, `manage-skills`)
**Scope:** Development workflow — scaffolding, testing, CI, code style, skill maintenance.
**Auto-invocation:** `contributing` and `ci` auto-invoke; `new-recipe` and `manage-skills` are manual-only.
**Key principle:** Reference `CONTRIBUTING.md`, `tests/`, `.github/workflows/`.

## Creating a new skill

### Step 1: Determine the layer
Which layer does this skill belong to? Skills should have a clear, non-overlapping scope. If it spans layers, split it.

### Step 2: Check for overlap
Read existing skills in `skills/` to ensure the new skill doesn't duplicate content. If there's overlap, update the existing skill instead.

### Step 3: Create the skill file

Create `skills/<skill-name>/SKILL.md` with this structure:

```yaml
---
name: <skill-name>
description: <Clear description of what the skill does and when to use it>
---

# <Skill Title>

<Brief description of what this skill helps with>

## Step 1: Understand the request
<What to ask the user if not specified>

## Step 2: Reference existing code
<Which files to read for patterns — be specific with file paths>

## Step 3: Key concepts
<Core APIs, parameters, patterns>

## Step 4: Implementation
<Code examples following repo conventions>

## Step N: Add tests
<Testing guidance — smoke tests and unit tests>
```

### Step 4: Follow these conventions

**Naming:**
- Lowercase, hyphenated: `tinker-sdk`, `new-recipe`, `manage-skills`
- Layer 0: named after the fundamental concept
- Layer 1: named after the SDK concept
- Layer 2: named after the primitive
- Layer 3: named after the algorithm/method
- Layer 4: named after the dev action

**Content rules:**
- Always reference **actual file paths** in the repo — never describe APIs from memory
- Include code examples that follow repo conventions (`@chz.chz`, explicit typing, etc.)
- For Layer 3 skills: defer primitive details to Layer 2 skills (e.g., say "see `/tinker-renderers` skill" instead of re-explaining renderers)
- Include a testing section pointing to `tests/recipes/` for smoke tests and `*_test.py` for unit tests
- Keep skills under 200 lines — move detailed reference material to separate files in the skill directory

**Frontmatter rules:**
- `description` is required and must clearly state **when** to trigger the skill
- Follow the [Agent Skills spec](https://agentskills.io/specification) for frontmatter fields

## Auditing existing skills

When auditing, check each skill for:

1. **Accuracy:** Do file paths and API references match the current codebase? Run `ls` or `grep` to verify.
2. **Freshness:** Has the referenced code changed since the skill was written? Check git log for the referenced files.
3. **Taxonomy compliance:** Is the skill in the correct layer? Does it overlap with other skills?
4. **Convention compliance:** Does it follow the structure above? Does it include testing guidance?
5. **Cross-references:** Do Layer 3 skills reference Layer 2 skills where appropriate?

## Current skill inventory

```
skills/
├── Layer 0: Fundamentals
│   ├── tinker-setup/            # Installation, API key, first run
│   ├── tinker-models/           # Model lineup, selection, families
│   ├── tinker-hyperparams/      # LR formulas, batch size, LoRA rank
│   └── tinker-logging/          # Training outputs, metrics, debugging
├── Layer 1: SDK
│   ├── tinker-sdk/              # ServiceClient, TrainingClient, SamplingClient, RestClient APIs
│   ├── tinker-types/            # Datum, ModelInput, TensorData, response types, error types
│   └── tinker-cli/              # tinker CLI: run/checkpoint management, download, publish
├── Layer 2: Primitives
│   ├── tinker-renderers/        # Renderer setup, TrainOnWhat, vision
│   ├── tinker-environments/     # Env, EnvGroupBuilder, custom RL envs
│   ├── tinker-weights/          # download, build_hf_model, publish
│   ├── tinker-completers/       # TokenCompleter, MessageCompleter
│   ├── tinker-checkpoints/      # save/load, CheckpointRecord, resume
│   ├── tinker-evals/            # Evaluators, Inspect AI
│   └── tinker-datasets/         # SupervisedDatasetBuilder, RLDatasetBuilder
├── Layer 3: Recipes
│   ├── tinker-sft/              # Supervised fine-tuning
│   ├── tinker-grpo/             # RL with verifiable rewards
│   ├── tinker-distillation/     # Knowledge distillation
│   ├── tinker-dpo/              # Direct Preference Optimization
│   ├── tinker-rlhf/             # RLHF pipeline
│   └── tinker-multiturn-rl/     # Multi-turn RL
├── Layer 4: Development
│   ├── tinker-new-recipe/       # Scaffold new recipe
│   ├── tinker-ci/               # Testing and CI
│   ├── tinker-contributing/     # Dev setup and code style
│   └── tinker-manage-skills/    # This skill
└── Meta
    └── tinker-update-skills/    # Install/update/uninstall global skills
```

## Maintenance schedule

When the codebase changes significantly (new modules, API changes, renamed files):
1. Run `/tinker-manage-skills audit` to check all skills
2. Update affected skills
3. Commit changes with a descriptive message
