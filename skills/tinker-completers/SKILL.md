---
name: tinker-completers
description: Guide for using completers — TokenCompleter and MessageCompleter for text generation during RL rollouts and evaluation. Use when the user asks about generating text, completing messages, or using completers in RL environments.
---

# Completers

Completers wrap SamplingClient for convenient text generation. Two levels of abstraction:
- **TokenCompleter** — low-level, returns tokens + logprobs
- **MessageCompleter** — high-level, returns parsed Message objects

## Reference

Read these for details:
- `tinker_cookbook/completers.py` — Implementation
- `docs/completers.mdx` — Usage guide

## TokenCompleter

Generates tokens from a ModelInput prompt. Used internally by RL rollouts.

```python
from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs

completer = TinkerTokenCompleter(
    sampling_client=sc,
    max_tokens=256,
    temperature=1.0,
)

result: TokensWithLogprobs = await completer(
    model_input=prompt,
    stop=stop_sequences,  # list[str] or list[int]
)
# result.tokens: list[int]
# result.maybe_logprobs: list[float] | None
```

## MessageCompleter

Higher-level: takes a conversation (list of Messages), returns a Message. Handles rendering and parsing internally.

```python
from tinker_cookbook.completers import TinkerMessageCompleter

completer = TinkerMessageCompleter(
    sampling_client=sc,
    renderer=renderer,
    max_tokens=256,
    temperature=1.0,
    stop_condition=None,  # Override stop sequences
)

response_message: Message = await completer(messages=[
    {"role": "user", "content": "What is 2+2?"},
])
# response_message = {"role": "assistant", "content": "4"}
```

## When to use which

- **TokenCompleter**: RL rollouts, custom generation loops where you need logprobs and token-level control
- **MessageCompleter**: Evaluation, tool-use environments, multi-turn RL where you work with Messages

## Custom completers

Both are abstract base classes you can subclass for non-Tinker backends:

```python
from tinker_cookbook.completers import TokenCompleter, MessageCompleter

class MyTokenCompleter(TokenCompleter):
    async def __call__(self, model_input, stop) -> TokensWithLogprobs:
        ...

class MyMessageCompleter(MessageCompleter):
    async def __call__(self, messages) -> Message:
        ...
```

## Common pitfalls
- Create a new completer (with a new SamplingClient) after saving weights
- `TokensWithLogprobs.maybe_logprobs` can be `None` if logprobs weren't requested
- MessageCompleter uses the renderer for both prompt construction and response parsing
