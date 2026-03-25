---
name: tinker-rlhf
description: Set up and run the full RLHF pipeline (SFT, reward model training, RL from reward model) using the Tinker API. Use when the user wants to do RLHF, train a reward model, or run the full preference-based RL pipeline.
---

# RL from Human Feedback (RLHF) Pipeline

Help the user set up and run the full 3-stage RLHF pipeline using the Tinker API.

## Overview

RLHF is a multi-stage pipeline:
1. **SFT Stage** — Fine-tune base model on instruction data
2. **Reward Model (RM) Stage** — Train a reward model on preference comparisons
3. **RL Stage** — Optimize the SFT policy using the reward model

## Step 1: Understand the request

Ask the user (if not already specified):
- **Base model**: Which model to start from (e.g., `meta-llama/Llama-3.2-3B`)
- **Preference data**: Which comparison dataset (HHH, HelpSteer3, UltraFeedback, or custom)
- **Which stages to run**: All 3, or skip SFT/RM if checkpoints exist
- **LoRA rank**: Typically 64 for RLHF

## Step 2: Reference existing recipes

Read these files:
- `tinker_cookbook/recipes/preference/rlhf/rlhf_pipeline.py` — Complete 3-stage pipeline
- `tinker_cookbook/rl/preference_envs.py` — Preference-based RL environments
- `tinker_cookbook/preference/types.py` — PreferenceModelBuilder
- `tinker_cookbook/preference/comparison_policy_evaluator.py` — RM evaluation
- `docs/preferences/rlhf-example.mdx` — RLHF guide

## Step 3: Configure each stage

### Stage 1: SFT
Standard supervised fine-tuning (see `/tinker-sft` skill). Key settings:
- Dataset: NoRobots or similar instruction data
- `sft_learning_rate`: 2e-4 (LoRA)
- `train_on_what`: `TrainOnWhat.ALL_ASSISTANT_MESSAGES`

### Stage 2: Reward Model
Train on preference comparisons:
- Uses `ChatDatasetBuilderFromComparisons` with a comparison builder (e.g., `HHHComparisonBuilder`)
- `rm_learning_rate`: 3e-4
- Produces a reward model checkpoint used in Stage 3

### Stage 3: RL from Reward Model
Optimize SFT policy using RM scores:
- Load SFT checkpoint as starting policy
- Load RM weights for scoring
- `PreferenceModelBuilderFromChatRenderer` wraps the RM
- `PairwisePreferenceRLDatasetBuilder` creates the RL environment
- `rl_learning_rate`: 1e-5 (much lower than SFT)
- `tournament_pattern`: `ALL_PAIRS_BOTH_WAYS` for pairwise comparison

### Typical Hyperparameters
```python
@chz.chz
class CLIConfig:
    base_model: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 64
    batch_size: int = 256
    max_length: int = 16384
    sft_learning_rate: float = 2e-4
    rm_learning_rate: float = 3e-4
    rl_learning_rate: float = 1e-5
    rl_max_tokens: int = 1024
    rl_group_size: int = 4
```

## Step 4: Write the training script

Follow the pipeline pattern from `rlhf_pipeline.py`:

```python
import asyncio
import os
import chz
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer
from tinker_cookbook.rl import preference_envs, train
from tinker_cookbook.supervised import train as supervised_train

# Stage 1: SFT
def sft_stage(log_path, base_model, ...):
    # Standard SFT config + supervised_train.main()
    ...

# Stage 2: Reward Model
def train_rm(log_path, base_model, ...):
    # Train on preference comparisons
    ...

# Stage 3: RL
async def train_rl(log_path, sft_log_path, rm_log_path, base_model, ...):
    sft_checkpoint = checkpoint_utils.get_last_checkpoint(sft_log_path)["state_path"]
    rm_weights = checkpoint_utils.get_last_checkpoint(rm_log_path)["sampler_path"]

    preference_model_builder = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name,
        model_name=base_model,
        rm_weights_path=rm_weights,
    )
    rl_dataset_builder = preference_envs.PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        preference_model_builder=preference_model_builder,
        batch_size=batch_size,
        group_size=group_size,
        tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        ...
    )
    config = train.Config(
        load_checkpoint_path=sft_checkpoint,
        dataset_builder=rl_dataset_builder,
        learning_rate=1e-5,
        ...
    )
    await train.main(config)
```

## Step 5: Run

```bash
# Full pipeline
python -m tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline

# Skip SFT (already have checkpoint)
python -m tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline run_sft=False

# Skip SFT and RM
python -m tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline run_sft=False run_rm=False
```

## Step 6: Add tests

If you created a new RLHF recipe, add a smoke test:

```python
# tests/recipes/test_recipe_<name>.py
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        ["behavior_if_log_dir_exists=delete"],
    )
```

`run_recipe()` automatically passes `max_steps=2` so the recipe runs 2 training steps and exits. See `tests/recipes/test_recipe_rlhf_pipeline.py` for the existing example.

## Common pitfalls
- RL learning rate must be **much lower** than SFT (1e-5 vs 2e-4)
- Checkpoints flow between stages: SFT → RL policy init, RM → RL reward scoring
- Use `checkpoint_utils.get_last_checkpoint()` to find checkpoints from previous stages
- RM quality directly impacts RL — validate RM before running Stage 3
- `group_size` in RL stage affects variance of reward estimates
