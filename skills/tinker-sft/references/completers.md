# Completers

Complete reference for TokenCompleter and MessageCompleter.

## Reference

- `tinker_cookbook/completers.py` — Implementation

## TokenCompleter

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

## MessageCompleter

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

## When to use which

- **TokenCompleter**: RL rollouts, custom generation loops needing logprobs and token-level control
- **MessageCompleter**: Evaluation, tool-use environments, multi-turn RL with Messages

## Custom completers

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

## Pitfalls

- Create a new completer (with a new SamplingClient) after saving weights
- `TokensWithLogprobs.maybe_logprobs` can be `None` if logprobs weren't requested
- MessageCompleter uses the renderer for both prompt construction and response parsing
