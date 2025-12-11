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
    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]

    def swap(self) -> "Comparison":
        return Comparison(
            prompt_conversation=self.prompt_conversation,
            completion_A=self.completion_B,
            completion_B=self.completion_A,
        )


@dataclass
class LabeledComparison:
    comparison: Comparison
    label: Literal["A", "B", "Tie"]

    def swap(self) -> "LabeledComparison":
        return LabeledComparison(
            comparison=self.comparison.swap(),
            label={"A": "B", "B": "A", "Tie": "Tie"}[self.label],  # pyright: ignore[reportArgumentType]
        )


class ComparisonRenderer:
    def build_generation_prompt(self, comparison: Comparison) -> types.ModelInput:
        raise NotImplementedError

    def to_model_input_weights(
        self, labeled_comparison: LabeledComparison
    ) -> tuple[types.ModelInput, torch.Tensor]:
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError


class ComparisonRendererFromChatRenderer(ComparisonRenderer):
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
    async def __call__(self, comparison: Comparison) -> float:
        """
        1: B is strongly preferred
        0: Tie
        -1: A is strongly preferred
        """
        raise NotImplementedError


class PreferenceModelBuilder:
    def __call__(self) -> PreferenceModel:
        raise NotImplementedError


class PreferenceModelFromChatRenderer(PreferenceModel):
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
        str_output = self.comparison_renderer.tokenizer.decode(response.sequences[0].tokens).strip()
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
