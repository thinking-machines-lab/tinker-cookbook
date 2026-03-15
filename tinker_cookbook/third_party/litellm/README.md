# LiteLLM Integration

A [LiteLLM](https://docs.litellm.ai/) custom provider that routes calls through Tinker's native `SamplingClient` for optimal sampling performance.

## Why use this?

If you have an agent or application built on LiteLLM (or frameworks that use it, like LangChain, CrewAI, or AutoGen), this integration lets you:

1. **Run your existing code against Tinker** without rewriting it to use the Tinker SDK directly
2. **Get raw token IDs** from every request, which you can feed into Tinker's training APIs for supervised learning or RL

Tinker also offers an [OpenAI-compatible endpoint](https://tinker-docs.thinkingmachines.ai/compatible-apis/openai), which works with LiteLLM out of the box. However, the native `SamplingClient` used by this integration provides better performance.

## Installation

```bash
pip install 'tinker_cookbook[litellm]'
```

## Quick start

```python
from tinker_cookbook.third_party.litellm import register_tinker_provider
import litellm

# Register once at startup
register_tinker_provider()

# Use litellm as normal — the "tinker/" prefix routes to this provider
response = await litellm.acompletion(
    model="tinker/my-model",
    messages=[{"role": "user", "content": "Hello!"}],
    base_model="Qwen/Qwen3-8B",  # determines renderer and sampling client
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

## Accessing raw tokens for training

The key feature of this integration is token-level access for training workflows:

```python
response = await litellm.acompletion(
    model="tinker/my-model",
    messages=messages,
    base_model="Qwen/Qwen3-8B",
)

# Raw token IDs are in provider_specific_fields
fields = response.choices[0].message.provider_specific_fields
prompt_token_ids = fields["prompt_token_ids"]       # list[int]
completion_token_ids = fields["completion_token_ids"]  # list[int]

# Use these directly with Tinker's training APIs
```

## Parameters

| LiteLLM parameter | Description |
|---|---|
| `model` | Any string with `tinker/` prefix (e.g., `"tinker/my-agent"`) |
| `base_model` | **Required.** Tinker model name (e.g., `"Qwen/Qwen3-8B"`). Determines the renderer and sampling client. |
| `temperature` | Sampling temperature |
| `max_tokens` / `max_completion_tokens` | Maximum tokens to generate |
| `top_p` | Nucleus sampling parameter |
| `top_k` | Top-k sampling parameter |
| `stop` | Stop sequences (defaults to model's stop sequences) |
| `tools` | OpenAI-format tool definitions |

## Injecting a custom SamplingClient

After saving new weights during training, you can point the provider at a specific checkpoint:

```python
import tinker

provider = register_tinker_provider()

# Later, after saving weights...
service = tinker.ServiceClient()
new_sampler = service.create_sampling_client(base_model="Qwen/Qwen3-8B")
provider.set_client("Qwen/Qwen3-8B", new_sampler)

# Subsequent litellm calls will use the new sampler
```

## Tool calling

Tool declarations are supported for models whose renderers implement `create_conversation_prefix_with_tools` (Qwen3, DeepSeek V3, Kimi K2/K2.5, GPT-OSS):

```python
response = await litellm.acompletion(
    model="tinker/my-agent",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    base_model="Qwen/Qwen3-8B",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
```

## Sync and async

Both `litellm.completion()` and `litellm.acompletion()` are supported.
