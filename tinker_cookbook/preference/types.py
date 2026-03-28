"""
Types for preference learning and Direct Preference Optimization (DPO).

This module defines the core data structures used for preference learning,
including comparisons between model outputs and preference models.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import chz
import tinker
import torch
from tinker import SamplingClient, types

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class Comparison:
    """A pair of completions (A and B) generated from the same prompt conversation."""

    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]

    def swap(self) -> "Comparison":
        """Return a new Comparison with A and B swapped."""
        return Comparison(
            prompt_conversation=self.prompt_conversation,
            completion_A=self.completion_B,
            completion_B=self.completion_A,
        )


@dataclass
class LabeledComparison:
    """A Comparison annotated with a human preference label (A, B, or Tie)."""

    comparison: Comparison
    label: Literal["A", "B", "Tie"]

    def swap(self) -> "LabeledComparison":
        """Return a new LabeledComparison with A/B swapped and label inverted."""
        return LabeledComparison(
            comparison=self.comparison.swap(),
            label={"A": "B", "B": "A", "Tie": "Tie"}[self.label],  # pyright: ignore[reportArgumentType]
        )


class ComparisonRenderer:
    """Abstract renderer for converting Comparisons to model inputs for preference training."""

    def build_generation_prompt(self, comparison: Comparison) -> types.ModelInput:
        raise NotImplementedError

    def to_model_input_weights(
        self, labeled_comparison: LabeledComparison
    ) -> tuple[types.ModelInput, torch.Tensor]:
        """Convert a labeled comparison to model input and per-token loss weights.

        Args:
            labeled_comparison: A comparison annotated with a preference label.

        Returns:
            A tuple of (model_input, weights) for training.
        """
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError


class ComparisonRendererFromChatRenderer(ComparisonRenderer):
    """Wraps a chat Renderer to render Comparisons for preference training.

    Formats comparisons by concatenating the prompt conversation with labeled
    sections for Completion A and Completion B, separated by system markers.

    Args:
        convo_renderer: The underlying chat Renderer to delegate to.
    """

    # TODO probably shouldn't be in types.py
    def __init__(self, convo_renderer: renderers.Renderer):
        self.convo_renderer = convo_renderer

    def _comparison_to_convo(self, comparison: Comparison) -> list[renderers.Message]:
        return [
            *comparison.prompt_conversation,
            {"role": "system", "content": "==== Completion A ===="},
            *comparison.completion_A,
            {"role": "system", "content": "==== Completion B ===="},
            *comparison.completion_B,
            {"role": "system", "content": "==== Preference ===="},
        ]

    def build_generation_prompt(self, comparison: Comparison) -> types.ModelInput:
        return self.convo_renderer.build_generation_prompt(self._comparison_to_convo(comparison))

    def to_model_input_weights(
        self, labeled_comparison: LabeledComparison
    ) -> tuple[types.ModelInput, torch.Tensor]:
        convo = self._comparison_to_convo(labeled_comparison.comparison)
        convo_with_pref = convo + [{"role": "assistant", "content": labeled_comparison.label}]
        model_input, weights = self.convo_renderer.build_supervised_example(convo_with_pref)
        # TODO: support images in preference learning
        assert all(isinstance(c, tinker.types.EncodedTextChunk) for c in model_input.chunks), (
            "Preference learning currently only supports text-only content."
        )
        # Truncate at the first weight==1 position + 1
        tokens = model_input.to_ints()
        first_weight_one_index = int(torch.nonzero(weights == 1.0)[0])
        truncated_tokens = tokens[: first_weight_one_index + 1]
        truncated_weights = weights[: first_weight_one_index + 1]
        return types.ModelInput.from_ints(truncated_tokens), truncated_weights

    @property
    def tokenizer(self) -> Tokenizer:
        return self.convo_renderer.tokenizer


class PreferenceModel:
    """Abstract base class for models that score a Comparison and return a preference float."""

    async def __call__(self, comparison: Comparison) -> float:
        """Return a preference score: -1 means A is preferred, 0 is a tie, 1 means B is preferred."""
        raise NotImplementedError


class PreferenceModelBuilder:
    """Abstract builder that creates PreferenceModel instances."""

    def __call__(self) -> PreferenceModel:
        raise NotImplementedError


class PreferenceModelFromChatRenderer(PreferenceModel):
    """A PreferenceModel that uses a chat renderer and Tinker sampling client.

    Renders a Comparison into a prompt, samples a single token (A, B, or Tie),
    and maps it to a float score (-1 for A, 1 for B, 0 for Tie).

    Args:
        convo_renderer: Renderer for formatting comparisons as chat prompts.
        sampling_client: Tinker sampling client for generating preference labels.
    """

    def __init__(self, convo_renderer: renderers.Renderer, sampling_client: SamplingClient):
        self.comparison_renderer = ComparisonRendererFromChatRenderer(convo_renderer)
        self.sampling_client = sampling_client

    async def __call__(self, comparison: Comparison) -> float:
        pm_input = self.comparison_renderer.build_generation_prompt(comparison)
        response = await self.sampling_client.sample_async(
            pm_input,
            num_samples=1,
            sampling_params=types.SamplingParams(temperature=0.0, max_tokens=1),
        )
        # TODO use probabilities
        str_output = str(
            self.comparison_renderer.tokenizer.decode(response.sequences[0].tokens)
        ).strip()
        if str_output == "A":
            return -1.0
        elif str_output == "B":
            return 1.0
        elif str_output == "Tie":
            return 0.0
        else:
            logger.warning(f"Invalid output preference model output: '{str_output}'")
            return 0.0


@chz.chz
class PreferenceModelBuilderFromChatRenderer(PreferenceModelBuilder):
    """Builds a PreferenceModel that uses a chat renderer and a Tinker sampling client."""

    renderer_name: str
    model_name: str
    rm_weights_path: str
    base_url: str | None = None

    def __call__(self) -> PreferenceModel:
        convo_renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        sampling_client = tinker.ServiceClient(base_url=self.base_url).create_sampling_client(
            model_path=self.rm_weights_path,
        )
        return PreferenceModelFromChatRenderer(convo_renderer, sampling_client)
