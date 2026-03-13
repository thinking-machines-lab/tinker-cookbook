---
name: multiturn-rl
description: Set up and run multi-turn RL training for interactive environments (terminal tasks, tool use, search/RAG, games) using the Tinker API. Use when the user wants multi-turn RL, agentic training, tool-use RL, or interactive environment training.
argument-hint: "[model-name] [environment-type]"
---

# Multi-Turn RL Training

Help the user set up RL training for multi-turn interactive environments using the Tinker API.

## Step 1: Understand the request

Ask the user (if not already specified):
- **Model**: Which model to train (e.g., `moonshotai/Kimi-K2-Thinking`, `Qwen/Qwen3-8B`)
- **Environment type**:
  - **Terminal/sandbox tasks**: Model executes shell commands (Harbor)
  - **Search/RAG**: Model uses retrieval tools (Search-R1)
  - **Multiplayer games**: Two models compete (TicTacToe, Twenty Questions, Guess Number)
  - **Custom multi-turn**: User-defined interactive environment
- **Turn structure**: Max turns, tool outputs, observation handling

## Step 2: Reference existing recipes

Read these files for patterns:
- `tinker_cookbook/recipes/harbor_rl/train.py` — Terminal task RL with sandbox execution
- `tinker_cookbook/recipes/harbor_rl/harbor_env.py` — HarborDatasetBuilder, sandbox factory
- `tinker_cookbook/recipes/search_tool/train.py` — Search-R1 with Chroma vector DB
- `tinker_cookbook/recipes/multiplayer_rl/text_arena/train.py` — Two-player games
- `tinker_cookbook/recipes/multiplayer_rl/twenty_questions/train.py` — Twenty Questions
- `tinker_cookbook/recipes/multiplayer_rl/guess_number/train.py` — Guess the Number
- `tinker_cookbook/rl/message_env.py` — Message-based environment interface
- `docs/rl/sequence-extension.mdx` — Multi-turn RL and KV-cache
- `docs/rl/rl-envs.mdx` — Custom environments

## Step 3: Configure the environment

### Harbor (Terminal Tasks)
Interactive sandbox where model runs shell commands and gets outputs:

```python
from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDatasetBuilder, HarborTask

dataset_builder = HarborDatasetBuilder(
    tasks=tasks,                    # List of HarborTask objects
    batch_size=8,                   # groups_per_batch
    group_size=4,                   # rollouts per task
    model_name=model_name,
    renderer_name=renderer_name,
    max_turns=10,                   # max interaction turns
    sandbox_timeout=3600,           # sandbox lifetime (seconds)
    command_timeout=120,            # per-command timeout
    grader_timeout=60,              # grading timeout
)
```

### Search/RAG (Search-R1)
Model queries a vector database during generation:

See `recipes/search_tool/train.py` for Chroma integration and streaming minibatch config.

### Multiplayer Games
Two models play against each other:

See `recipes/multiplayer_rl/text_arena/train.py` for the competitive RL pattern.

### Key Multi-Turn Parameters

- `max_turns`: Maximum number of interaction turns
- `max_tokens`: Max tokens per generation step
- `kl_penalty_coef`: KL penalty (often 0.0 for multi-turn to allow exploration)
- `max_steps_off_policy`: Enable async rollouts for expensive environments

### Async Rollouts
Multi-turn envs are slow due to tool execution. Use async config:
```python
config = Config(
    ...
    async_config=AsyncConfig(
        max_steps_off_policy=cli_config.max_steps_off_policy,
        groups_per_batch=cli_config.groups_per_batch,
    ) if cli_config.max_steps_off_policy is not None else None,
)
```

## Step 4: Write the training script

Follow the Harbor pattern:

```python
import asyncio
import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

@chz.chz
class CLIConfig:
    model_name: str = "moonshotai/Kimi-K2-Thinking"
    lora_rank: int = 32
    max_tokens: int = 8192
    max_turns: int = 10
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    max_steps_off_policy: int | None = None

async def cli_main(cli_config: CLIConfig):
    renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)

    dataset_builder = ...  # Your multi-turn dataset builder

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        log_path="/tmp/tinker-examples/multiturn/my_run",
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        ) if cli_config.max_steps_off_policy is not None else None,
    )

    await main(config)
```

## Step 5: Run

```bash
# Harbor terminal RL
python -m tinker_cookbook.recipes.harbor_rl.train

# Search-R1
python -m tinker_cookbook.recipes.search_tool.train

# Multiplayer games
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.train
```

## Step 6: Add tests

If you created a new multi-turn recipe, add a smoke test:

```python
# tests/test_recipe_<name>.py
from tests.helpers import run_recipe

def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        ["behavior_if_log_dir_exists=delete", "groups_per_batch=4", "group_size=2"],
    )
```

See `tests/test_recipe_text_arena.py` and `tests/test_recipe_twenty_questions.py` for existing multi-turn examples. For environment-specific logic (sandbox setup, tool parsing), add unit tests as `*_test.py` next to the source code.

## Common pitfalls
- Multi-turn envs are expensive — start with small `groups_per_batch` (4-8)
- Use `max_steps_off_policy` for async rollouts when env execution is slow
- `Env` objects are single-use — the builder creates fresh envs each batch
- Sandbox timeouts need to be generous enough for complex tasks
- KV-cache (sequence extension) is key for multi-turn efficiency — see `docs/rl/sequence-extension.mdx`
- `kl_penalty_coef=0.0` is common for multi-turn since you want the model to explore tool use
