---
name: tinker-sft
description: Set up and run supervised fine-tuning (SFT) on instruction or chat datasets using the Tinker API. Use when the user wants to do instruction tuning, chat fine-tuning, or supervised learning.
---

# Supervised Fine-Tuning (SFT)

Help the user set up and run supervised fine-tuning using the Tinker API.

## Key concepts

**Renderer:** Converts chat messages to tokens. Always resolve automatically:
```python
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

**TrainOnWhat** controls which tokens get trained on:
- `TrainOnWhat.ALL_ASSISTANT_MESSAGES` — Train on all assistant turns (most common)
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — Train only on final assistant response
- `TrainOnWhat.ALL_TOKENS` — Train on entire conversation including user messages

**ChatDatasetBuilderCommonConfig** bundles tokenizer, renderer, and dataset settings:
```python
common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=renderer_name,
    max_length=32768,
    batch_size=128,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
```

**Built-in datasets:** `NoRobotsBuilder`, `Tulu3Builder` (from `tinker_cookbook.recipes.chat_sl.chat_datasets`)

**Custom JSONL dataset:** Use `FromConversationFileBuilder` with a JSONL file where each line is:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Hyperparameters:**
- `learning_rate`: Use `hyperparam_utils.get_lr(model_name)` or ~2e-4 for LoRA
- `batch_size`: Number of tokens per batch (128 is a reasonable starting point)
- `num_epochs`: Number of passes through the dataset
- `eval_every`: Evaluate every N batches

## Minimal working example

This is a complete, runnable SFT script:

```python
import asyncio
import sys

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
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
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    if 0:  # To swap in your own dataset:
        dataset = FromConversationFileBuilder(
            common_config=common_config, file_path="/path/to/your/dataset.jsonl"
        )
    return chz.Blueprint(train.Config).apply({
        "log_path": "/tmp/tinker-examples/sl_basic",
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

Run it: `python my_sft.py` or override params: `python my_sft.py learning_rate=1e-4 batch_size=256`

## Customization

**Change the model:** Replace `model_name` — renderer resolves automatically. See `/tinker-models` for available models.

**Use Tulu3 dataset:** Replace `NoRobotsBuilder` with `chat_datasets.Tulu3Builder(common_config=common_config)`.

**Use custom JSONL:** Set `if 0` to `if 1` in the example above and point to your file.

**Add LoRA rank:** Add `"lora_rank": 32` to the blueprint `.apply({...})` dict.

**Add evaluators:** Add `"evaluator_builders": [...]` — see the [GitHub repo](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/chat_sl) for examples with inline evaluators.

For testing and weight export patterns, see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook).

## Common pitfalls
- Always use `model_info.get_recommended_renderer_name()` — never hardcode renderer names
- Use `cli_utils.check_log_dir()` to avoid clobbering previous runs
- LR too high causes instability; LR too low wastes compute. Use `hyperparam_utils.get_lr(model_name)` for recommendations.
