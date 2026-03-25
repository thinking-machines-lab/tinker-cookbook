---
name: tinker-sft
description: Set up and run supervised fine-tuning (SFT) on instruction or chat datasets using the Tinker API. Use when the user wants to do instruction tuning, chat fine-tuning, or supervised learning.
argument-hint: "[model-name] [dataset]"
---

# Supervised Fine-Tuning (SFT)

Help the user set up and run supervised fine-tuning using the Tinker API.

## Step 1: Understand the request

Ask the user (if not already specified):
- **Model**: Which base model to fine-tune (e.g., `meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-8B`). See `docs/model-lineup.mdx` for available models.
- **Dataset**: What data to train on — built-in datasets (NoRobots, Tulu3) or custom JSONL file.
- **Goal**: General instruction tuning, domain-specific fine-tuning, or chat quality improvement.

## Step 2: Reference existing recipes

Read these files for patterns and conventions:
- `tinker_cookbook/recipes/sl_basic.py` — Minimal SFT example
- `tinker_cookbook/recipes/chat_sl/train.py` — Full-featured chat SFT with eval
- `tinker_cookbook/supervised/train.py` — Core training loop
- `tinker_cookbook/supervised/data.py` — Dataset construction helpers
- `docs/supervised-learning/sl-basic.mdx` — Getting started guide
- `docs/supervised-learning/sl-hyperparams.mdx` — Learning rate and batch size guidance

## Step 3: Configure the training run

Key configuration decisions:

### Renderer
Match renderer to model family using `model_info.get_recommended_renderer_name(model_name)`. Never hardcode renderer names.

### Learning Rate
- Use `hyperparam_utils.get_lr(model_name)` for recommended LR
- LoRA fine-tuning typically needs ~10x higher LR than full fine-tuning (e.g., 2e-4 for LoRA vs 2e-5 for full)

### TrainOnWhat
- `TrainOnWhat.ALL_ASSISTANT_MESSAGES` — Train on all assistant turns (most common)
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — Train only on final assistant response
- `TrainOnWhat.EVERYTHING` — Train on entire conversation including user messages

### Dataset
- Built-in: `NoRobotsBuilder`, `Tulu3Builder`
- Custom JSONL: Use `FromConversationFileBuilder(common_config=..., file_path="path/to/data.jsonl")`
- Format: Same as `tinker_cookbook/example_data/conversations.jsonl`

### Batch Size & Epochs
- `batch_size`: Number of tokens per training batch (default: 128 for basic, scale up as needed)
- `num_epochs`: Number of passes through the dataset
- `eval_every`: Evaluate every N batches

## Step 4: Write the training script

Follow the pattern from `sl_basic.py`:

```python
import asyncio
import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # Configure dataset builder here
    dataset = ...

    return chz.Blueprint(train.Config).apply({
        "log_path": "/tmp/tinker-examples/my_sft_run",
        "model_name": model_name,
        "renderer_name": renderer_name,
        "dataset_builder": dataset,
        "learning_rate": 2e-4,
        "lr_schedule": "linear",
        "num_epochs": 1,
        "eval_every": 8,
    })

def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
```

## Step 5: Run and iterate

```bash
python -m tinker_cookbook.recipes.<recipe_name>
```

Override parameters from CLI: `python -m tinker_cookbook.recipes.<recipe_name> learning_rate=1e-4 batch_size=256`

## Step 6: Add tests

If you created a new recipe, add a smoke test so CI catches regressions:

```python
# tests/recipes/test_recipe_<name>.py
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        ["behavior_if_log_dir_exists=delete", "batch_size=16"],
    )
```

`run_recipe()` automatically passes `max_steps=2` to the recipe, so it runs 2 training steps and exits cleanly. Tests are auto-discovered by CI and run daily. For unit-testable components (dataset processing, custom logic), add `*_test.py` files next to the source code.

## Step 7: Export weights (optional)

After training, export weights using the `tinker_cookbook.weights` API:

```python
from tinker_cookbook import weights

adapter_dir = weights.download(tinker_path="tinker://run-id/sampler_weights/final", output_dir="./adapter")
weights.build_hf_model(base_model="meta-llama/Llama-3.1-8B", adapter_path=adapter_dir, output_path="./model")
weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-finetuned-model")
```

## Common pitfalls
- Always use `model_info.get_recommended_renderer_name()` — never hardcode renderer
- Use `cli_utils.check_log_dir()` to avoid clobbering previous runs
- For custom datasets, ensure JSONL matches the conversation format in `example_data/conversations.jsonl`
- LR too high causes instability; LR too low wastes compute
