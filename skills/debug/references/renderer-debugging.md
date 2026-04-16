# Renderer Debugging Reference

How to diagnose and fix renderer issues that cause silent training quality degradation.

## How renderers work

A renderer converts a list of chat messages into model-specific token sequences for training and sampling. Each model family (Llama3, Qwen3, DeepSeek, etc.) has its own token format with specific role delimiters, special tokens, and conventions.

Key renderer methods:
- `build_generation_prompt(messages)` → `ModelInput` (tokens for sampling)
- `build_supervised_example(messages)` → `(ModelInput, weights)` (tokens + loss mask for training)
- `parse_response(token_ids)` → `(Message, success)` (decode sampled tokens back to a message)

## Check your transformers version first

Different `transformers` versions have known bugs that affect specific models:
- `transformers == 5.3.0`: Incorrect `tokenizer_class` for DeepSeek V2/V3 on the hub (huggingface/transformers#44801, fixed in 5.3.1). Causes tokenizer loading to fail or use the wrong tokenizer class.
- `transformers < 5.0`: Bug in `Qwen2VLImageProcessor` — miscounts image tokens for VL models
- Always print `transformers.__version__` in your debug output

```python
import transformers
print(f"transformers: {transformers.__version__}")
```

## Full token comparison recipe

This script compares the cookbook renderer against HuggingFace's `apply_chat_template` for a given model and conversation:

```python
"""
Renderer parity check — verifies cookbook renderer matches HuggingFace tokenizer.
Usage: python renderer_check.py
"""
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

MODEL_NAME = "Qwen/Qwen3-8B"  # Change this

# Setup
tokenizer = get_tokenizer(MODEL_NAME)
renderer_name = get_recommended_renderer_name(MODEL_NAME)
renderer = get_renderer(renderer_name, tokenizer)

# Test conversations — try several patterns
test_cases = [
    # Basic conversation
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ],
    # With system message
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    # Multi-turn
    [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ],
]

for i, messages in enumerate(test_cases):
    print(f"\n=== Test case {i + 1} ===")

    # Cookbook tokens
    cookbook_mi = renderer.build_generation_prompt(messages)
    cookbook_tokens = cookbook_mi.to_ints()

    # HuggingFace tokens
    hf_tokens = list(tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    ))

    if cookbook_tokens == hf_tokens:
        print(f"PASS: {len(cookbook_tokens)} tokens match")
    else:
        print(f"FAIL: cookbook={len(cookbook_tokens)} tokens, HF={len(hf_tokens)} tokens")

        # Find divergence point
        min_len = min(len(cookbook_tokens), len(hf_tokens))
        for j in range(min_len):
            if cookbook_tokens[j] != hf_tokens[j]:
                ctx_start = max(0, j - 3)
                print(f"  First diff at position {j}:")
                print(f"    cookbook[{ctx_start}:{j+3}]: {cookbook_tokens[ctx_start:j+3]}")
                print(f"    HF[{ctx_start}:{j+3}]:      {hf_tokens[ctx_start:j+3]}")
                print(f"    cookbook decoded: {tokenizer.decode(cookbook_tokens[max(0,j-5):j+5])!r}")
                print(f"    HF decoded:      {tokenizer.decode(hf_tokens[max(0,j-5):j+5])!r}")
                break

        if len(cookbook_tokens) != len(hf_tokens):
            print(f"  Length diff: cookbook has {len(cookbook_tokens) - len(hf_tokens):+d} tokens")
```

## Thinking mode

Models with thinking capabilities (Qwen3, DeepSeek V3, Kimi K2.5, Nemotron3) have two renderer variants:
- **With thinking** (`qwen3`, `deepseekv3_thinking`): Model produces `<think>...</think>` blocks before responding
- **Without thinking** (`qwen3_disable_thinking`, `deepseekv3`): Thinking is suppressed

Common issues:
- Training on data with `<think>` blocks using a non-thinking renderer → Thinking tokens get wrong loss weights
- Using a thinking renderer but passing `thinking=False` to HF template → Token mismatch
- Historical assistant messages may have thinking stripped by default (`strip_thinking_from_history=True` in Qwen3)

When comparing against HF for thinking models:
```python
# For Qwen3 with thinking enabled:
hf_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, thinking=True)

# For Qwen3 with thinking disabled:
hf_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, thinking=False)
```

## Tool calling

Each model family has a different tool call format:
- **Qwen3**: `<tool_call>\n{json}\n</tool_call>` tags
- **DeepSeek V3**: Special tokens like `<tool_calls_begin>`, `<tool_calls_end>`
- **Kimi K2**: `## FunctionCall {json}` format
- **Llama3**: Bare JSON (unreliable parsing — tool calling not well-supported)

When comparing tool-calling conversations against HF:
```python
# Must pass tools parameter to HF
hf_tokens = tokenizer.apply_chat_template(
    messages,
    tools=tool_specs,  # List of tool schema dicts
    add_generation_prompt=True,
    tokenize=True,
)
```

Not all renderers have tool formats that match HF — some intentionally diverge. Check the model's renderer implementation if tool calling quality is poor.

## KL divergence at step 0

If KL divergence is high at the very first training step (before any gradient updates), the renderer is likely producing tokens the model doesn't expect. The model's log-probabilities on the "correct" tokens are low because those tokens don't match its learned distribution.

Diagnosis:
1. Run the token comparison above
2. If tokens match HF, the renderer is correct — high step-0 KL may indicate the data distribution is far from the model's pre-training distribution (especially common with gpt-oss models on specialized data)
3. If tokens don't match, fix the renderer first

## Available renderers

Use `get_recommended_renderer_name()` — never hardcode:
- **Llama 3.x**: `llama3`
- **Qwen3**: `qwen3`, `qwen3_disable_thinking`, `qwen3_instruct`, `qwen3_vl`, `qwen3_vl_instruct`
- **Qwen3.5**: `qwen3_5`, `qwen3_5_disable_thinking`
- **DeepSeek V3**: `deepseekv3` (no thinking), `deepseekv3_thinking`
- **Kimi K2**: `kimi_k2`
- **Kimi K2.5**: `kimi_k25`, `kimi_k25_disable_thinking`
- **Nemotron3**: `nemotron3`, `nemotron3_disable_thinking`
- **GPT-OSS**: `gpt_oss_no_sysprompt`, `gpt_oss_low_reasoning`, `gpt_oss_medium_reasoning`, `gpt_oss_high_reasoning`
- **Generic fallback**: `role_colon`

## Vision model considerations

Vision-language (VL) models add image tokens alongside text tokens, creating additional points of failure.

### Setup

VL renderers require an image processor from the `transformers` library:

```python
from transformers import AutoProcessor
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

model_name = "Qwen/Qwen3-VL-7B-Instruct"
tokenizer = get_tokenizer(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

renderer = get_renderer(
    "qwen3_vl_instruct",
    tokenizer,
    image_processor=processor.image_processor,  # Required for VL renderers
)
```

Forgetting `image_processor` raises `RendererError("qwen3_vl renderer requires an image_processor")`.

### VL renderers by model

| Model | Renderer | Notes |
|-------|----------|-------|
| Qwen3-VL | `qwen3_vl` | Thinking-enabled VL. For instruct-only: `qwen3_vl_instruct`. |
| Qwen3.5 (VL variants) | `qwen3_5` | Same renderer handles VL when image_processor provided |
| Kimi K2.5 | `kimi_k25` | Supports vision natively |

### Common VL issues

**`Expected X tokens, got Y from image`**

This is the most frequently reported VL bug. It's caused by a bug in HuggingFace's `Qwen2VLImageProcessor` in `transformers < 5.0` that miscounts image tokens.

Fix: `pip install 'transformers>=5.0'` (or install `torchvision` as an alternative workaround).

**Image token count mismatch between training and serving**

Different image resolutions produce different numbers of image tokens. The image processor determines this. Ensure:
- Same `transformers` version during training and serving
- Same image processor configuration (max resolution, etc.)
- Images are not re-encoded between training data preparation and actual training

**Token comparison for VL**

When comparing VL renderer output against HuggingFace, you need to include images in the comparison:

```python
from PIL import Image

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": Image.open("test.png")},
        {"type": "text", "text": "What's in this image?"},
    ]},
]

# Cookbook
cookbook_mi = renderer.build_generation_prompt(messages)
cookbook_tokens = cookbook_mi.to_ints()

# HuggingFace — use the full processor, not just tokenizer
hf_inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
# Compare input_ids
```

Note: VL token comparison is more complex because image tokens may be represented differently. Focus on verifying that text tokens surrounding images match and that the total image token count is correct.

**Weight export for VL models**

VL models add a `model.language_model.*` prefix to language model weights. The cookbook's `weights.build_hf_model()` handles this automatically. Custom merge scripts commonly miss this prefix, causing the LoRA adapter to silently not be applied (adapter weight names don't match model weight names).
