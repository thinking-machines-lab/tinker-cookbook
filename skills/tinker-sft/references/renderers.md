# Renderers

Complete reference for all available renderers.

## Available renderers

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

## Key methods

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

## TrainOnWhat (all variants)

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

## Vision inputs

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

## Custom renderers

```python
from tinker_cookbook.renderers import register_renderer

def my_renderer_factory(tokenizer, image_processor):
    return MyCustomRenderer(tokenizer)

register_renderer("my_renderer", my_renderer_factory)
```

## Code references

- `tinker_cookbook/renderers/__init__.py` — Factory, registry
- `tinker_cookbook/renderers/base.py` — Renderer base class, Message, ContentPart types
- `tinker_cookbook/renderers/qwen3.py` — Qwen3 renderers
- `tinker_cookbook/renderers/qwen3_5.py` — Qwen3.5 renderers
- `tinker_cookbook/renderers/llama3.py` — Llama3 renderer
- `tinker_cookbook/renderers/deepseek_v3.py` — DeepSeek renderers
- `tinker_cookbook/renderers/kimi_k2.py`, `kimi_k25.py` — Kimi renderers
- `tinker_cookbook/renderers/nemotron3.py` — Nemotron renderer
- `tinker_cookbook/renderers/gpt_oss.py` — GPT-OSS renderers
