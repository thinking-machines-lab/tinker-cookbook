import logging
from typing import Callable

import tinker

# Set up logger
logger = logging.getLogger(__name__)


class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient
    """

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a TokenCompleter
    """

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
