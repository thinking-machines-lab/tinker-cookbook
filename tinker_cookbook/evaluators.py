from typing import Callable

import tinker_public


class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient
    """

    async def __call__(self, training_client: tinker_public.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a TokenCompleter
    """

    async def __call__(self, sampling_client: tinker_public.SamplingClient) -> dict[str, float]:
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
