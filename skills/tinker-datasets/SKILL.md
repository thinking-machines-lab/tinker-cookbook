---
name: tinker-datasets
description: Guide for dataset construction — SupervisedDatasetBuilder, RLDatasetBuilder, ChatDatasetBuilder, and custom dataset creation from JSONL, HuggingFace, or conversation data. Use when the user asks about datasets, data loading, data preparation, or custom data formats.
---

# Datasets

The cookbook uses the builder pattern for datasets: a `*DatasetBuilder` (config) builds a `*Dataset` (runtime).

## Reference

Read these for details:
- `tinker_cookbook/supervised/types.py` — SupervisedDatasetBuilder, ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
- `tinker_cookbook/supervised/data.py` — Dataset construction helpers, FromConversationFileBuilder
- `tinker_cookbook/rl/types.py` — RLDatasetBuilder, RLDataset
- `docs/training-sampling.mdx` — Data preparation basics

## Supervised datasets

### ChatDatasetBuilderCommonConfig

Shared config for all chat-based dataset builders:

```python
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat

common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
    renderer_name="llama3",
    max_length=32768,        # Max sequence length
    batch_size=128,          # Tokens per batch
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
```

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
    test_size=100,       # Hold out 100 examples for eval
    shuffle_seed=42,
)
```

JSONL format — each line is a conversation:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

See `tinker_cookbook/example_data/conversations.jsonl` for the expected format.

### From HuggingFace datasets

```python
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset

dataset = SupervisedDatasetFromHFDataset(
    hf_dataset=hf_dataset,
    batch_size=128,
    map_fn=lambda example: conversation_to_datum(
        example["messages"], renderer, max_length, train_on_what
    ),
)
```

### Low-level datum construction

```python
from tinker_cookbook.supervised.data import conversation_to_datum

# Full pipeline: messages → datum
datum = conversation_to_datum(messages, renderer, max_length, train_on_what)

# Or step by step:
model_input, weights = renderer.build_supervised_example(messages)
datum = datum_from_model_input_weights(model_input, weights, max_length)
```

## RL datasets

RL datasets return batches of `EnvGroupBuilder` objects. See the `/tinker-environments` skill for details.

```python
@chz.chz
class MyRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 128
    group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Return (train_dataset, optional_test_dataset)
        ...
```

## DPO datasets

DPO uses comparison pairs (chosen vs rejected):

```python
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons

dataset = DPODatasetBuilderFromComparisons(
    common_config=common_config,
    comparison_builder=HHHComparisonBuilder(),
)
```

See `tinker_cookbook/preference/dpo_datasets.py` and `tinker_cookbook/recipes/preference/datasets.py`.

## Common pitfalls
- Always use `ChatDatasetBuilderCommonConfig` for consistent tokenizer/renderer setup
- `batch_size` is in tokens, not examples — larger sequences mean fewer examples per batch
- Custom JSONL must match the format in `example_data/conversations.jsonl`
- Use `test_size` to hold out evaluation data from the same distribution
- Dataset builders must be serializable (`@chz.chz`) for config persistence and sweeps
