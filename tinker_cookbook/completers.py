"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import tinker

from tinker_cookbook import renderers

logger = logging.getLogger(__name__)

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]

# ---------------------------------------------------------------------------
# Sample sink — an optional observer for every successful sample made through
# the Tinker completers. Used by the token DB
# (``tinker_cookbook.tokendb.capture.capture_samples``) to persist sampled
# tokens; ``None`` (the default) is a single falsy check per call.
# ---------------------------------------------------------------------------

SampleSink: TypeAlias = Callable[
    [tinker.ModelInput, Sequence[tinker.types.SampledSequence], dict[str, Any]], None
]
"""Callback ``(model_input, sampled_sequences, sampling_metadata)`` invoked
after each successful sample. Sink exceptions are logged and swallowed so a
sink can never break sampling."""

_sample_sink: SampleSink | None = None


def set_sample_sink(sink: SampleSink | None) -> None:
    """Register a sink observing every completer sample (``None`` disables)."""
    global _sample_sink
    _sample_sink = sink


def get_sample_sink() -> SampleSink | None:
    """Return the registered sample sink, or ``None`` when disabled."""
    return _sample_sink


def _notify_sample_sink(
    model_input: tinker.ModelInput,
    sequences: Sequence[tinker.types.SampledSequence],
    metadata: dict[str, Any],
) -> None:
    """Invoke the sample sink if registered; never raises."""
    if _sample_sink is None:
        return
    try:
        _sample_sink(model_input, sequences, metadata)
    except Exception:
        logger.exception("Sample sink failed; continuing without capture")


@dataclass
class TokensWithLogprobs:
    """A sequence of token IDs with optional log-probabilities and a stop reason."""

    tokens: list[int]
    maybe_logprobs: list[float] | None
    stop_reason: tinker.StopReason = "stop"

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    """Abstract interface for generating tokens from a prompt."""

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Generate a token sequence from the given model input.

        Args:
            model_input (tinker.ModelInput): The tokenized prompt to complete from.
            stop (StopCondition): Stop sequences (strings) or stop token IDs
                that terminate generation.

        Returns:
            TokensWithLogprobs: The generated tokens with their log-probabilities
                and stop reason.
        """
        raise NotImplementedError


class MessageCompleter:
    """Abstract interface for generating message responses."""

    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        """Generate an assistant message given a conversation history.

        Args:
            messages (list[renderers.Message]): The conversation history as a
                list of message dicts with ``role`` and ``content`` keys.

        Returns:
            renderers.Message: The generated assistant message, which may include
                ``tool_calls`` if the model produced them.
        """
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """Token completer that uses a tinker.SamplingClient to sample actions.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        max_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature. Default: 1.0.
        context_window (int | None): Model's total context window size. When
            set, ``max_tokens`` is dynamically capped per request so that
            ``prompt_tokens + max_tokens <= context_window``. This prevents
            "prompt + max_tokens exceeds context window" API errors.

    Example::

        completer = TinkerTokenCompleter(sampling_client, max_tokens=512)
        result = await completer(model_input, stop=["<|endoftext|>"])
        print(result.tokens, result.logprobs)
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    context_window: int | None = None

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        max_tokens = self.max_tokens
        if self.context_window is not None:
            max_tokens = min(max_tokens, self.context_window - model_input.length)
            if max_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({model_input.length}) exceeds context window "
                    f"({self.context_window}). No room for generation."
                )

        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
            ),
        )

        _notify_sample_sink(
            model_input,
            sample_result.sequences,
            {
                "completer": "TinkerTokenCompleter",
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            },
        )

        # Extract tokens, logprobs, and stop_reason from the first (and only) sample
        sampled_seq = sample_result.sequences[0]
        assert sampled_seq.logprobs is not None

        return TokensWithLogprobs(
            tokens=sampled_seq.tokens,
            maybe_logprobs=sampled_seq.logprobs,
            stop_reason=sampled_seq.stop_reason,
        )


class TinkerMessageCompleter(MessageCompleter):
    """Message completer that uses a tinker.SamplingClient to generate responses.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        renderer (renderers.Renderer): Renderer that converts between messages
            and token sequences.
        max_tokens (int): Maximum number of tokens to generate per call.
        stop_condition (StopCondition | None): Custom stop condition. If ``None``,
            uses the renderer's default stop sequences.
        temperature (float): Sampling temperature. Default: 1.0.

    Example::

        completer = TinkerMessageCompleter(sampling_client, renderer, max_tokens=512)
        response = await completer([
            {"role": "user", "content": "What is 2+2?"}
        ])
        print(response["content"])
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        _notify_sample_sink(
            model_input,
            response.sequences,
            {
                "completer": "TinkerMessageCompleter",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        # Decode the response
        parsed_message, _termination = self.renderer.parse_response(response.sequences[0].tokens)

        result: renderers.Message = {"role": "assistant", "content": parsed_message["content"]}
        if "tool_calls" in parsed_message:
            result["tool_calls"] = parsed_message["tool_calls"]
        return result
