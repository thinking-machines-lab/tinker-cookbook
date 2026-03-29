# Datasets

Complete reference for dataset construction patterns.

## Reference

- `tinker_cookbook/supervised/types.py` — SupervisedDatasetBuilder, ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
- `tinker_cookbook/supervised/data.py` — Dataset construction helpers
- `tinker_cookbook/rl/types.py` — RLDatasetBuilder, RLDataset

## ChatDatasetBuilderCommonConfig

Shared config for all chat-based dataset builders:

```python
from tinker_cookbook import model_info
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat

model_name = "meta-llama/Llama-3.1-8B"
common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=model_info.get_recommended_renderer_name(model_name),
    max_length=32768,
    batch_size=128,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
```

## Built-in datasets

```python
from tinker_cookbook.recipes.chat_sl.chat_datasets import NoRobotsBuilder, Tulu3Builder

dataset = NoRobotsBuilder(common_config=common_config)
dataset = Tulu3Builder(common_config=common_config)
```

## Custom JSONL file

```python
from tinker_cookbook.supervised.data import FromConversationFileBuilder

dataset = FromConversationFileBuilder(
    common_config=common_config,
    file_path="/path/to/data.jsonl",
    test_size=100,
    shuffle_seed=42,
)
```

Format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

## From HuggingFace datasets

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

## Low-level datum construction

```python
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.common import datum_from_model_input_weights

datum = conversation_to_datum(messages, renderer, max_length, train_on_what)

# Or step by step:
model_input, weights = renderer.build_supervised_example(messages)
datum = datum_from_model_input_weights(model_input, weights, max_length)
```

## RL datasets

RL datasets return batches of `EnvGroupBuilder` objects:

```python
@chz.chz
class MyRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 128
    group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return MyDataset(...), None
```

## DPO datasets

```python
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons

dataset = DPODatasetBuilderFromComparisons(
    common_config=common_config,
    comparison_builder=HHHComparisonBuilder(),
)
```

See `tinker_cookbook/preference/dpo_datasets.py` and `tinker_cookbook/recipes/preference/datasets.py`.
