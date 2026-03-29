"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import logging
from dataclasses import dataclass, field
from typing import TypeAlias

import tinker

from tinker_cookbook import renderers

logger = logging.getLogger(__name__)

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


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

    When ``context_window`` is set, ``max_tokens`` is dynamically capped per-request
    so that ``prompt_length + effective_max_tokens <= context_window``. This prevents
    ``BadRequestError`` when long prompts would otherwise exceed the model's context.

    Args:
        sampling_client (tinker.SamplingClient): Client used to sample from
            the model.
        max_tokens (int): Maximum number of tokens to generate per call.
        temperature (float): Sampling temperature. Default: 1.0.
        context_window (int | None): If set, dynamically caps max_tokens per-request.

    Example::

        completer = TinkerTokenCompleter(sampling_client, max_tokens=512)
        result = await completer(model_input, stop=["<|endoftext|>"])
        print(result.tokens, result.logprobs)
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    context_window: int | None = field(default=None)

    # Safety buffer subtracted from context_window to account for special tokens
    _CONTEXT_BUFFER: int = field(default=64, repr=False)

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        effective_max_tokens = self.max_tokens

        if self.context_window is not None:
            prompt_length = model_input.length
            available = self.context_window - prompt_length - self._CONTEXT_BUFFER
            if available < effective_max_tokens:
                if available <= 0:
                    logger.warning(
                        "Prompt length (%d) leaves no room for generation "
                        "(context_window=%d, buffer=%d). Allowing min 1 token.",
                        prompt_length, self.context_window, self._CONTEXT_BUFFER,
                    )
                    effective_max_tokens = 1
                else:
                    logger.info(
                        "Capping max_tokens from %d to %d "
                        "(prompt_length=%d, context_window=%d)",
                        self.max_tokens, available, prompt_length, self.context_window,
                    )
                    effective_max_tokens = available

        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=effective_max_tokens,
                temperature=self.temperature,
            ),
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

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        result: renderers.Message = {"role": "assistant", "content": parsed_message["content"]}
        if "tool_calls" in parsed_message:
            result["tool_calls"] = parsed_message["tool_calls"]
        return result
