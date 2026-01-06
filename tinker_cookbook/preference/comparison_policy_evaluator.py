import asyncio
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.preference.types import (
    Comparison,
    PreferenceModel,
)
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


class ComparisonEvaluator(SamplingClientEvaluator):
    """
    Evaluates a policy by comparing its completions to references, with a reward model
    """

    def __init__(
        self,
        preference_model_builder: Callable[[], PreferenceModel],
        comparisons: Sequence[Comparison],
        renderer_name: str,
        model_name_for_tokenizer: str,
        both_ways: bool = True,
        max_tokens: int = 1024,
        content_preprocessor: Callable[[str], str] | None = None,
    ):
        self.preference_model_builder = preference_model_builder
        self.both_ways = both_ways
        self.comparisons = comparisons
        self.renderer = get_renderer(renderer_name, get_tokenizer(model_name_for_tokenizer))
        self.max_tokens = max_tokens
        if content_preprocessor is None:
            self.content_preprocessor = lambda x: x
        else:
            self.content_preprocessor = content_preprocessor

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        preference_model = self.preference_model_builder()
        policy = TinkerMessageCompleter(sampling_client, self.renderer, self.max_tokens)

        async def process_comparison(comparison: Comparison) -> float:
            new_completion_message = await policy(comparison.prompt_conversation)
            new_completion_content = get_text_content(new_completion_message)
            new_completion_message = {
                "role": "assistant",
                "content": self.content_preprocessor(new_completion_content),
            }
            new_comparison = replace(comparison, completion_B=[new_completion_message])
            r_0, r_1 = await asyncio.gather(
                preference_model(new_comparison), preference_model(new_comparison.swap())
            )
            # r_0, r_1 are in between -1 and 1
            # so r0-r1 is in between -2 and 2, and we normalize it to 0-1
            return (r_0 - r_1 + 2) / 4.0

        results = await asyncio.gather(
            *[process_comparison(comparison) for comparison in self.comparisons]
        )
        return {
            "win_rate": np.mean(results).item(),
            "stderr": np.std(results).item() / np.sqrt(len(results)),
        }
