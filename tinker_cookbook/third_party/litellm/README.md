# LiteLLM Integration

A [LiteLLM](https://docs.litellm.ai/) custom provider that routes calls through Tinker's native `SamplingClient` for optimal sampling performance.

## Why use this?

If you have an agent or application built on LiteLLM (or frameworks that use it, like LangChain, CrewAI, or AutoGen), this integration lets you:

1. **Run your existing code against Tinker** without rewriting it to use the Tinker SDK directly
2. **Get raw token IDs** from every request, which you can feed into Tinker's training APIs for supervised learning or RL

Tinker also offers an [OpenAI-compatible endpoint](/compatible-apis/openai), which works with LiteLLM out of the box. However, the native `SamplingClient` used by this integration provides better performance.

## Setup

This integration requires `litellm` as an additional dependency. From the tinker-cookbook repo root:

```bash
uv pip install -e ".[litellm]"
```

You also need a `TINKER_API_KEY` — see [Getting an API key](https://tinker-docs.thinkingmachines.ai/install#getting-an-api-key).

## Quick start

```python
from tinker_cookbook.third_party.litellm import register_litellm_provider
import litellm

# Register once at startup
register_litellm_provider()

# The "tinker/" prefix routes to this provider.
# base_model is the Tinker model to sample from.
response = await litellm.acompletion(
    model="tinker/my-label",
    messages=[{"role": "user", "content": "Hello!"}],
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

## How `model` and `base_model` work

LiteLLM uses the `model` parameter to decide which provider handles the request. The `tinker/` prefix routes to this provider — everything after the prefix is an arbitrary label that appears in the response metadata (it does **not** select the model).

The actual model is determined by `base_model`, which is passed directly to Tinker's `ServiceClient.create_sampling_client(base_model=...)`. This must be a model name from Tinker's [model lineup](https://tinker-docs.thinkingmachines.ai/model-lineup), e.g.:

- `Qwen/Qwen3-4B-Instruct-2507`
- `meta-llama/Llama-3.1-8B-Instruct`
- `moonshotai/Kimi-K2.5`

You can list available models with:

```python
import tinker
service = tinker.ServiceClient()
for m in service.get_server_capabilities().supported_models:
    print(m.model_name)
```

### Sampling from a fine-tuned checkpoint

When you pass `base_model`, the provider creates a `SamplingClient` for that pretrained model. To sample from a **fine-tuned** checkpoint instead, use `set_client()` to inject a `SamplingClient` created with a `model_path`:

```python
import tinker

provider = register_litellm_provider()

# Create a sampling client pointing at your fine-tuned checkpoint.
# The model_path comes from training_client.save_weights_for_sampler().
service = tinker.ServiceClient()
sampler = service.create_sampling_client(
    model_path="tinker://<experiment-id>/sampler_weights/000080"
)

# The provider reads the base model from the sampling client automatically
# to resolve the correct renderer and tokenizer.
provider.set_client(sampler)

# Now litellm calls will sample from your fine-tuned checkpoint.
# base_model must still match so the provider finds the right client bundle.
response = await litellm.acompletion(
    model="tinker/my-finetuned",
    messages=[{"role": "user", "content": "Hello!"}],
    base_model="Qwen/Qwen3-4B-Instruct-2507",
)
```

See [Saving and loading weights](https://tinker-docs.thinkingmachines.ai/save-load) for how to obtain checkpoint paths.

## Accessing raw tokens for training

The key feature of this integration is token-level access for training workflows:

```python
response = await litellm.acompletion(
    model="tinker/my-label",
    messages=messages,
    base_model="Qwen/Qwen3-4B-Instruct-2507",
)

# Raw token IDs are in provider_specific_fields
fields = response.choices[0].message.provider_specific_fields
prompt_token_ids = fields["prompt_token_ids"]       # list[int]
completion_token_ids = fields["completion_token_ids"]  # list[int]

# Use these directly with Tinker's training APIs
```

## Supported parameters

| LiteLLM parameter | Description |
|---|---|
| `model` | Must start with `tinker/` to route to this provider. The rest is a label for the response metadata. |
| `base_model` | **Required.** Tinker model name passed to `create_sampling_client()`. See [model lineup](https://tinker-docs.thinkingmachines.ai/model-lineup). |
| `temperature` | Sampling temperature |
| `max_tokens` / `max_completion_tokens` | Maximum tokens to generate |
| `top_p` | Nucleus sampling parameter |
| `top_k` | Top-k sampling parameter |
| `stop` | Stop sequences (defaults to model's stop sequences) |
| `tools` | OpenAI-format tool definitions |

## Tool calling

Tool declarations are supported for models whose renderers implement `create_conversation_prefix_with_tools` (Qwen3, DeepSeek V3, Kimi K2/K2.5, GPT-OSS):

```python
response = await litellm.acompletion(
    model="tinker/my-agent",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    base_model="Qwen/Qwen3-4B-Instruct-2507",
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
