"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

from dataclasses import dataclass
from typing import TypeAlias

import tinker

from tinker_cookbook import renderers

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


class TinkerMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
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
                temperature=1.0,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        return {"role": "assistant", "content": parsed_message["content"]}
