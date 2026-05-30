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
    """A pair of completions (A and B) generated from the same prompt conversation.

    Used as the basic unit for preference learning: a single prompt with two
    candidate responses that can be compared by a human or a preference model.

    Attributes:
        prompt_conversation (list[renderers.Message]): The shared prompt
            messages that precede both completions.
        completion_A (list[renderers.Message]): First candidate completion.
        completion_B (list[renderers.Message]): Second candidate completion.

    Example::

        comparison = Comparison(
            prompt_conversation=[{"role": "user", "content": "Summarize X"}],
            completion_A=[{"role": "assistant", "content": "Summary A..."}],
            completion_B=[{"role": "assistant", "content": "Summary B..."}],
        )
    """

    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]

    def swap(self) -> "Comparison":
        """Return a new Comparison with A and B swapped.

        Returns:
            Comparison: A copy where ``completion_A`` and ``completion_B``
                are exchanged.
        """
        return Comparison(
            prompt_conversation=self.prompt_conversation,
            completion_A=self.completion_B,
            completion_B=self.completion_A,
        )


@dataclass
class LabeledComparison:
    """A Comparison annotated with a human preference label (A, B, or Tie).

    Attributes:
        comparison (Comparison): The underlying comparison of two completions.
        label (Literal["A", "B", "Tie"]): Which completion is preferred, or
            ``"Tie"`` if neither is.

    Example::

        labeled = LabeledComparison(comparison=comparison, label="A")
    """

    comparison: Comparison
    label: Literal["A", "B", "Tie"]

    def swap(self) -> "LabeledComparison":
        """Return a new LabeledComparison with A/B swapped and label inverted.

        The label is flipped (A becomes B and vice versa; Tie stays Tie) so
        the semantics remain consistent after the swap.

        Returns:
            LabeledComparison: A copy with completions and label swapped.
        """
        return LabeledComparison(
            comparison=self.comparison.swap(),
            label={"A": "B", "B": "A", "Tie": "Tie"}[self.label],  # pyright: ignore[reportArgumentType]
        )


class ComparisonRenderer:
    """Abstract renderer for converting Comparisons to model inputs for preference training.

    Subclasses must implement ``build_generation_prompt``,
    ``to_model_input_weights``, and the ``tokenizer`` property.
    """

    def build_generation_prompt(self, comparison: Comparison) -> types.ModelInput:
        """Build a model input for generating a preference label.

        Args:
            comparison (Comparison): The comparison to render as a prompt.

        Returns:
            types.ModelInput: Token-level model input ready for sampling.
        """
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
        """The tokenizer associated with this renderer.

        Returns:
            Tokenizer: The tokenizer instance.
        """
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
        """Build a model input prompt for generating a preference label.

        Args:
            comparison (Comparison): The comparison to render.

        Returns:
            types.ModelInput: Token-level model input ending at the
                ``"==== Preference ===="`` marker, ready for sampling.
        """
        return self.convo_renderer.build_generation_prompt(self._comparison_to_convo(comparison))

    def to_model_input_weights(
        self, labeled_comparison: LabeledComparison
    ) -> tuple[types.ModelInput, torch.Tensor]:
        convo = self._comparison_to_convo(labeled_comparison.comparison)
        convo_with_pref = convo + [{"role": "assistant", "content": labeled_comparison.label}]
        model_input, weights = self.convo_renderer.build_supervised_example(convo_with_pref)
        # Find the first position with weight==1 (start of the preference label)
        first_weight_one_index = int(torch.nonzero(weights == 1.0)[0])

        # Truncate model_input and weights at first_weight_one_index + 1
        # Handle both text chunks and image chunks
        truncated_chunks: list[types.ModelInputChunk] = []
        truncated_weights_list: list[float] = []

        current_pos = 0
        for chunk in model_input.chunks:
            chunk_start = current_pos
            chunk_end = current_pos + chunk.length

            # Check if this chunk is entirely before the truncation point
            if chunk_end <= first_weight_one_index + 1:
                truncated_chunks.append(chunk)
                truncated_weights_list.extend([weights[i].item() for i in range(chunk_start, chunk_end)])
            else:
                # Chunk overlaps with truncation point
                if isinstance(chunk, tinker.types.EncodedTextChunk):
                    # For text chunks, we can partially include them
                    tokens_before_truncation = first_weight_one_index + 1 - chunk_start
                    if tokens_before_truncation > 0:
                        truncated_chunks.append(
                            tinker.types.EncodedTextChunk(tokens=chunk.tokens[:tokens_before_truncation])
                        )
                        truncated_weights_list.extend(
                            [weights[i].item() for i in range(chunk_start, chunk_start + tokens_before_truncation)]
                        )
                # Image chunks must be entirely included or excluded (can't partially truncate)
                # If an image chunk overlaps, we exclude it entirely
                elif isinstance(chunk, tinker.types.EncodedImageChunk):
                    logger.warning(f"Image chunk dropped due to truncation at index {first_weight_one_index}")

            current_pos = chunk_end

        truncated_model_input = types.ModelInput(chunks=truncated_chunks)
        truncated_weights = torch.tensor(truncated_weights_list, dtype=torch.float32)

        return truncated_model_input, truncated_weights

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer from the underlying chat renderer.

        Returns:
            Tokenizer: The tokenizer instance.
        """
        return self.convo_renderer.tokenizer


class PreferenceModel:
    """Abstract base class for models that score a Comparison and return a preference float.

    Subclasses implement ``__call__`` to return a score in [-1, 1] indicating
    which completion is preferred.
    """

    async def __call__(self, comparison: Comparison) -> float:
        """Return a preference score for the given comparison.

        Args:
            comparison (Comparison): The comparison to evaluate.

        Returns:
            float: A score in [-1, 1].  -1 means A is strongly preferred,
                0 is a tie, and 1 means B is strongly preferred.
        """
        raise NotImplementedError


class PreferenceModelBuilder:
    """Abstract builder that creates PreferenceModel instances.

    Implementations typically hold configuration (model paths, renderer names)
    and construct a fully-initialized ``PreferenceModel`` on ``__call__``.
    """

    def __call__(self) -> PreferenceModel:
        """Build and return a PreferenceModel instance.

        Returns:
            PreferenceModel: A ready-to-use preference model.
        """
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
        """Score a comparison by sampling a preference token from the model.

        Renders the comparison into a prompt, samples a single token
        (``A``, ``B``, or ``Tie``), and maps it to a float.

        Args:
            comparison (Comparison): The comparison to evaluate.

        Returns:
            float: -1.0 if A is preferred, 1.0 if B is preferred, 0.0 for
                a tie or an unrecognized output.
        """
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
    """Builds a PreferenceModel that uses a chat renderer and a Tinker sampling client.

    This is a ``chz`` dataclass that can be serialized and passed as
    configuration to training scripts.

    Attributes:
        renderer_name (str): Name of the chat renderer (e.g. ``"llama3"``).
        model_name (str): Model name used to select the tokenizer.
        rm_weights_path (str): Path to the reward/preference model weights.
        base_url (str | None): Optional override for the Tinker service URL.

    Example::

        builder = PreferenceModelBuilderFromChatRenderer(
            renderer_name="llama3",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            rm_weights_path="path/to/rm_weights",
        )
        pref_model = builder()
    """

    renderer_name: str
    model_name: str
    rm_weights_path: str
    base_url: str | None = None

    def __call__(self) -> PreferenceModel:
        """Create and return a ``PreferenceModelFromChatRenderer``.

        Returns:
            PreferenceModel: An initialized preference model backed by a
                Tinker sampling client.
        """
        convo_renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        sampling_client = tinker.ServiceClient(base_url=self.base_url).create_sampling_client(
            model_path=self.rm_weights_path,
        )
        return PreferenceModelFromChatRenderer(convo_renderer, sampling_client)
