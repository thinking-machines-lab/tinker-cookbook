---
name: tinker-sft
description: Set up and run supervised fine-tuning (SFT), knowledge distillation, or any supervised learning workflow using the Tinker API. Covers datasets, renderers, completers, and distillation. Use when the user wants to do instruction tuning, chat fine-tuning, supervised learning, dataset preparation, rendering, text generation, or knowledge distillation — even if they don't say "SFT" explicitly.
---

# Supervised Learning

Everything for supervised fine-tuning, datasets, renderers, completers, and distillation.

## Minimal SFT example

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

Run: `python my_sft.py` or override: `python my_sft.py learning_rate=1e-4 batch_size=256`

## Renderers

Renderers convert chat messages to token sequences for training and generation. Always resolve automatically:

```python
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

renderer_name = model_info.get_recommended_renderer_name(model_name)
tokenizer = get_tokenizer(model_name)
renderer = get_renderer(renderer_name, tokenizer)
```

Key methods:
```python
model_input = renderer.build_generation_prompt(messages)              # For sampling
model_input, weights = renderer.build_supervised_example(messages)    # For training
message, is_complete = renderer.parse_response(token_ids)             # Parse output
stop = renderer.get_stop_sequences()                                  # Stop tokens
```

### TrainOnWhat

Controls which tokens receive training signal:
- `TrainOnWhat.ALL_ASSISTANT_MESSAGES` — Most common
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — Train only on final response
- `TrainOnWhat.ALL_TOKENS` — Train on everything including user messages
- `TrainOnWhat.CUSTOMIZED` — Set `trainable=True/False` on individual messages

For the full renderer table (18 renderers across all model families), vision input handling, and custom renderer registration, read `references/renderers.md`.

## Datasets

The cookbook uses the builder pattern: a `*DatasetBuilder` (config) builds a `*Dataset` (runtime).

### Built-in datasets

```python
from tinker_cookbook.recipes.chat_sl.chat_datasets import NoRobotsBuilder, Tulu3Builder

dataset = NoRobotsBuilder(common_config=common_config)
dataset = Tulu3Builder(common_config=common_config)
```

### Custom JSONL file

```python
from tinker_cookbook.supervised.data import FromConversationFileBuilder

dataset = FromConversationFileBuilder(
    common_config=common_config,
    file_path="/path/to/data.jsonl",
    test_size=100, shuffle_seed=42,
)
```

JSONL format — each line:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Low-level datum construction

```python
from tinker_cookbook.supervised.data import conversation_to_datum

datum = conversation_to_datum(messages, renderer, max_length, train_on_what)

# Or step by step:
model_input, weights = renderer.build_supervised_example(messages)
datum = datum_from_model_input_weights(model_input, weights, max_length)
```

For HuggingFace dataset loading, DPO datasets, and more details, read `references/datasets.md`.

## Knowledge distillation

On-policy distillation: student generates, teacher scores via KL divergence. No correctness rewards needed.

```python
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig, PromptOnlyDatasetBuilder, TeacherConfig,
)

teacher_config = TeacherConfig(base_model="Qwen/Qwen3-8B")
dataset_builder = PromptOnlyDatasetBuilder(
    dataset_name="deepmath",  # or "tulu3"
    groups_per_batch=1024, group_size=4,
    model_name_for_tokenizer=model_name, renderer_name=renderer_name,
)
dataset_config = DistillationDatasetConfig(
    dataset_builder=dataset_builder, teacher_config=teacher_config,
    groups_per_batch=1024,
)
config = train_on_policy.Config(
    dataset_configs=[dataset_config],
    model_name="Qwen/Qwen3-8B-Base",  # Student
    renderer_name=renderer_name,
    learning_rate=1e-4, lora_rank=128,
    kl_penalty_coef=1.0, kl_discount_factor=0.0,
    log_path="/tmp/tinker-examples/distillation",
)
await train_on_policy.main(config)
```

**Multi-teacher:** Pass multiple `DistillationDatasetConfig` objects with different teachers.
**Off-policy:** Use standard SFT on teacher-generated reasoning traces.

For the full distillation guide, read `references/distillation.md`.

## Completers

Completers wrap SamplingClient for convenient text generation:
- **TokenCompleter** — low-level, returns tokens + logprobs (used in RL rollouts)
- **MessageCompleter** — high-level, returns parsed Message objects (used in eval, tool-use)

```python
from tinker_cookbook.completers import TinkerTokenCompleter, TinkerMessageCompleter

# Token level
completer = TinkerTokenCompleter(sampling_client=sc, max_tokens=256, temperature=1.0)
result = await completer(model_input=prompt, stop=stop_sequences)

# Message level
completer = TinkerMessageCompleter(sampling_client=sc, renderer=renderer, max_tokens=256)
response_message = await completer(messages=[{"role": "user", "content": "What is 2+2?"}])
```

For custom completer subclassing, read `references/completers.md`.

## Customization

- **Change model**: Replace `model_name` — renderer resolves automatically
- **Add LoRA rank**: Add `"lora_rank": 32` to blueprint
- **Add evaluators**: Add `"evaluator_builders": [...]` to config

## Async patterns (important for throughput)

The built-in `supervised/train.py` already uses async internally. For custom SL loops or evaluation, always overlap API calls:

```python
# CORRECT: overlap forward_backward with data prep
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=adam_params)
next_batch = dataset.get_batch(i + 1)  # Prepare while GPU works
fb_result = fb_future.result()
optim_result = optim_future.result()

# WRONG: sequential calls waste GPU cycles
result = tc.forward_backward(data=batch, loss_fn="cross_entropy")
tc.optim_step(adam_params=adam_params)
next_batch = dataset.get_batch(i + 1)  # GPU idle during data prep
```

For evaluation, run samples concurrently rather than one-by-one:
```python
import asyncio
eval_tasks = [evaluate_sample(sc, sample) for sample in eval_set]
results = await asyncio.gather(*eval_tasks)
```

## Common pitfalls

- **Sequential API calls**: Always use `_async` variants in training loops and overlap GPU work with data preparation. Sequential `.result()` chains waste GPU cycles.
- **Sampler desync**: Create a **new** SamplingClient (and new completer) after every weight save. A stale client silently uses old weights.
- Always use `model_info.get_recommended_renderer_name()` — never hardcode
- Use `cli_utils.check_log_dir()` to avoid clobbering previous runs
- `batch_size` is in tokens, not examples
- Custom JSONL must use the messages format shown above
- `forward()` computes loss without gradients — use it for eval only, not in training loops

## Code references

- `tinker_cookbook/supervised/train.py` — SL training loop and Config
- `tinker_cookbook/supervised/types.py` — SupervisedDatasetBuilder, ChatDatasetBuilder
- `tinker_cookbook/supervised/data.py` — Dataset construction helpers
- `tinker_cookbook/renderers/` — All renderer implementations
- `tinker_cookbook/completers.py` — TokenCompleter, MessageCompleter
- `tinker_cookbook/distillation/` — Distillation training
- `tinker_cookbook/recipes/chat_sl/` — SFT recipes with built-in datasets
- `tinker_cookbook/recipes/distillation/` — Distillation recipes
