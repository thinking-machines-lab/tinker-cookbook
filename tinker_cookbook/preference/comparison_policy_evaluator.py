import asyncio
from dataclasses import replace
from typing import Sequence, cast

import datasets
import numpy as np
import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.preference.preference_datasets import HHHComparisonBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    PreferenceModel,
    PreferenceModelFromChatRenderer,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


class ComparisonEvaluator:
    """
    Evaluates a policy by comparing its completions to references, with a reward model
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        comparisons: Sequence[Comparison],
    ):
        self.preference_model = preference_model
        self.comparisons = comparisons

    async def __call__(self, policy: MessageCompleter) -> dict[str, float]:
        async def process_comparison(comparison: Comparison) -> float:
            new_completion_message = await policy(comparison.prompt_conversation)
            new_comparison = replace(comparison, completion_B=[new_completion_message])
            result = await self.preference_model(new_comparison)
            return result

        results = await asyncio.gather(
            *[process_comparison(comparison) for comparison in self.comparisons]
        )
        # TODO swap some of them
        results_scaled = (np.array(results) + 1.0) / 2.0
        return {
            "win_rate": np.mean(results_scaled).item(),
            "stderr": np.std(results_scaled).item() / np.sqrt(len(results)),
        }


def make_comparison_evaluator(
    renderer_name: str,
    sampling_client: tinker.SamplingClient,
    tmp_model_name_for_tokenizer: str,
    dataset_size: int = 512,
) -> ComparisonEvaluator:
    # XXX do this when it works: tokenizer = sampling_client.get_tokenizer()
    tokenizer = get_tokenizer(tmp_model_name_for_tokenizer)
    convo_renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    preference_model = PreferenceModelFromChatRenderer(
        convo_renderer=convo_renderer, sampling_client=sampling_client
    )
    dataset = (
        cast(datasets.Dataset, datasets.load_dataset("Anthropic/hh-rlhf", split="test"))
        .shuffle(seed=0)
        .select(range(dataset_size))
    )
    hhh_builder = HHHComparisonBuilder()
    comparisons = [
        labeled_comparison.comparison
        for example in dataset
        if (
            labeled_comparison := hhh_builder.example_to_labeled_comparison(
                cast(dict[str, str], example)
            )
        )
        is not None
    ]
    return ComparisonEvaluator(preference_model, comparisons)
