---
name: tinker-renderers
description: Guide for using renderers — the bridge between chat-style messages and token sequences. Covers renderer setup, TrainOnWhat, vision inputs, model family matching, and custom renderers. Use when the user asks about renderers, tokenization, message formatting, or vision inputs.
---

# Renderers

Renderers convert chat-style messages into token sequences for training and generation.

## Reference

Read these for details:
- `tinker_cookbook/renderers/base.py` — Renderer base class and API
- `tinker_cookbook/renderers/__init__.py` — Registry, factory, TrainOnWhat enum
- `docs/rendering.mdx` — Rendering guide with examples

## Getting a renderer

Always use `model_info.get_recommended_renderer_name()` — never hardcode:

```python
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

renderer_name = model_info.get_recommended_renderer_name(model_name)
tokenizer = get_tokenizer(model_name)
renderer = get_renderer(renderer_name, tokenizer)
```

**Available renderers:** `llama3`, `qwen3`, `deepseekv3`, `kimi_k2`, `kimi_k25`, `nemotron3`, `nemotron3_disable_thinking`, `role_colon`, and more. See `tinker_cookbook/renderers/__init__.py` for the full registry.

## Key renderer methods

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

## TrainOnWhat

Controls which tokens receive training signal:

```python
from tinker_cookbook.renderers import TrainOnWhat

# Most common — train on all assistant responses
TrainOnWhat.ALL_ASSISTANT_MESSAGES

# Train only on the final assistant response
TrainOnWhat.LAST_ASSISTANT_MESSAGE

# Train on everything (including user messages)
TrainOnWhat.ALL_TOKENS

# Other options
TrainOnWhat.LAST_ASSISTANT_TURN
TrainOnWhat.ALL_MESSAGES
TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES
TrainOnWhat.CUSTOMIZED  # Set trainable=True/False on individual messages
```

## Vision inputs

For VLM models, use `ImageChunk` in messages:

```python
message = {
    "role": "user",
    "content": [
        {"type": "image", "image_url": "https://..."},  # or local path
        {"type": "text", "text": "What is in this image?"},
    ],
}
```

See `docs/rendering.mdx` and `tinker_cookbook/recipes/vlm_classifier/train.py` for VLM examples.

## Custom renderers

Register a custom renderer:

```python
from tinker_cookbook.renderers import register_renderer

def my_renderer_factory(tokenizer, image_processor):
    return MyCustomRenderer(tokenizer)

register_renderer("my_renderer", my_renderer_factory)
```

## Picklability

Renderers must be pickleable for distributed rollout execution. The codebase tests this — see `tinker_cookbook/renderers/renderer_pickle_test.py`.

## Common pitfalls
- Always use `model_info.get_recommended_renderer_name()` — renderer must match model family
- After loading a checkpoint trained with a specific renderer, use the same renderer name
- `build_supervised_example()` returns weights as `list[float]` — wrap with `TensorData.from_numpy()` if needed
- For tool calling, use `create_conversation_prefix_with_tools()` to inject tool definitions
