# SFT Reference — Renderers, Datasets, Completers

---

## Renderers

### Available renderers

| Renderer name | Model family | Notes |
|---|---|---|
| `llama3` | Llama 3.x | |
| `qwen3` | Qwen3 | Thinking enabled |
| `qwen3_disable_thinking` | Qwen3 | Thinking disabled |
| `qwen3_instruct` | Qwen3 Instruct 2507 | No thinking |
| `qwen3_vl` | Qwen3 VL | Vision + thinking |
| `qwen3_vl_instruct` | Qwen3 VL Instruct | Vision, no thinking |
| `qwen3_5` | Qwen3.5 VL | Thinking enabled |
| `qwen3_5_disable_thinking` | Qwen3.5 VL | Thinking disabled |
| `deepseekv3` | DeepSeek V3 | Defaults to non-thinking |
| `deepseekv3_thinking` | DeepSeek V3 | Thinking mode |
| `kimi_k2` | Kimi K2 | Thinking format |
| `kimi_k25` | Kimi K2.5 | Thinking enabled |
| `kimi_k25_disable_thinking` | Kimi K2.5 | Thinking disabled |
| `nemotron3` | Nemotron-3 | Thinking enabled |
| `nemotron3_disable_thinking` | Nemotron-3 | Thinking disabled |
| `gpt_oss_no_sysprompt` | GPT-OSS | No system prompt |
| `gpt_oss_low_reasoning` | GPT-OSS | Low reasoning |
| `gpt_oss_medium_reasoning` | GPT-OSS | Medium reasoning |
| `gpt_oss_high_reasoning` | GPT-OSS | High reasoning |
| `role_colon` | Generic | Simple `role: content` format |

### Key methods

```python
# Build generation prompt (for sampling)
model_input = renderer.build_generation_prompt(messages, role="assistant")

# Build supervised example (for training)
model_input, weights = renderer.build_supervised_example(
    messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
)

# Parse model output back to a message
message, is_complete = renderer.parse_response(token_ids)

# Get stop sequences for sampling
stop = renderer.get_stop_sequences()

# Tool calling support
prefix_messages = renderer.create_conversation_prefix_with_tools(tool_specs)
```

### TrainOnWhat (all variants)

```python
from tinker_cookbook.renderers import TrainOnWhat

TrainOnWhat.ALL_ASSISTANT_MESSAGES        # Most common
TrainOnWhat.LAST_ASSISTANT_MESSAGE        # Final response only
TrainOnWhat.ALL_TOKENS                    # Everything including user messages
TrainOnWhat.LAST_ASSISTANT_TURN           # Last assistant turn only
TrainOnWhat.ALL_MESSAGES                  # All messages
TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES  # User + system only
TrainOnWhat.CUSTOMIZED                    # Per-message trainable flag
```

### Vision inputs

For VLM models, use image content parts:
```python
message = {
    "role": "user",
    "content": [
        {"type": "image", "image_url": "https://..."},
        {"type": "text", "text": "What is in this image?"},
    ],
}
```

Use a VL renderer (`qwen3_vl`, `qwen3_5`, etc.) and pass `image_processor` to `get_renderer()`.

### Custom renderers

```python
from tinker_cookbook.renderers import register_renderer

def my_renderer_factory(tokenizer, image_processor):
    return MyCustomRenderer(tokenizer)

register_renderer("my_renderer", my_renderer_factory)
```

### Renderer code references

- `tinker_cookbook/renderers/__init__.py` — Factory, registry
- `tinker_cookbook/renderers/base.py` — Renderer base class, Message, ContentPart types
- `tinker_cookbook/renderers/qwen3.py` — Qwen3 renderers
- `tinker_cookbook/renderers/qwen3_5.py` — Qwen3.5 renderers
- `tinker_cookbook/renderers/llama3.py` — Llama3 renderer
- `tinker_cookbook/renderers/deepseek_v3.py` — DeepSeek renderers
- `tinker_cookbook/renderers/kimi_k2.py`, `kimi_k25.py` — Kimi renderers
- `tinker_cookbook/renderers/nemotron3.py` — Nemotron renderer
- `tinker_cookbook/renderers/gpt_oss.py` — GPT-OSS renderers

---

## Datasets

### ChatDatasetBuilderCommonConfig

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
    test_size=100,
    shuffle_seed=42,
)
```

Format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

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
from tinker_cookbook.supervised.common import datum_from_model_input_weights

datum = conversation_to_datum(messages, renderer, max_length, train_on_what)

# Or step by step:
model_input, weights = renderer.build_supervised_example(messages)
datum = datum_from_model_input_weights(model_input, weights, max_length)
```

### RL datasets

RL datasets return batches of `EnvGroupBuilder` objects:

```python
@chz.chz
class MyRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int = 128
    group_size: int = 4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        return MyDataset(...), None
```

### DPO datasets

```python
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons

dataset = DPODatasetBuilderFromComparisons(
    common_config=common_config,
    comparison_builder=HHHComparisonBuilder(),
)
```

See `tinker_cookbook/preference/dpo_datasets.py` and `tinker_cookbook/recipes/preference/datasets.py`.

### Dataset code references

- `tinker_cookbook/supervised/types.py` — SupervisedDatasetBuilder, ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
- `tinker_cookbook/supervised/data.py` — Dataset construction helpers
- `tinker_cookbook/rl/types.py` — RLDatasetBuilder, RLDataset

---

## Completers

### TokenCompleter

Generates tokens from a ModelInput prompt. Used internally by RL rollouts.

```python
from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs

completer = TinkerTokenCompleter(
    sampling_client=sc, max_tokens=256, temperature=1.0,
)

result: TokensWithLogprobs = await completer(
    model_input=prompt,
    stop=stop_sequences,
)
# result.tokens: list[int]
# result.maybe_logprobs: list[float] | None
```

### MessageCompleter

Higher-level: takes a conversation, returns a Message. Handles rendering and parsing internally.

```python
from tinker_cookbook.completers import TinkerMessageCompleter

completer = TinkerMessageCompleter(
    sampling_client=sc, renderer=renderer,
    max_tokens=256, temperature=1.0, stop_condition=None,
)

response_message: Message = await completer(messages=[
    {"role": "user", "content": "What is 2+2?"},
])
```

### When to use which

- **TokenCompleter**: RL rollouts, custom generation loops needing logprobs and token-level control
- **MessageCompleter**: Evaluation, tool-use environments, multi-turn RL with Messages

### Custom completers

Both are abstract base classes for non-Tinker backends:

```python
from tinker_cookbook.completers import TokenCompleter, MessageCompleter

class MyTokenCompleter(TokenCompleter):
    async def __call__(self, model_input, stop) -> TokensWithLogprobs:
        ...

class MyMessageCompleter(MessageCompleter):
    async def __call__(self, messages) -> Message:
        ...
```

### Completer pitfalls

- Create a new completer (with a new SamplingClient) after saving weights
- `TokensWithLogprobs.maybe_logprobs` can be `None` if logprobs weren't requested
- MessageCompleter uses the renderer for both prompt construction and response parsing

### Completer code references

- `tinker_cookbook/completers.py` — Implementation
