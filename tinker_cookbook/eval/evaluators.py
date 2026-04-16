"""Base evaluator interfaces for inline evaluation during training."""

import logging
from collections.abc import Callable

import tinker

# Set up logger
logger = logging.getLogger(__name__)


class TrainingClientEvaluator:
    """An evaluator that uses a TrainingClient to compute metrics (e.g., loss).

    Example::

        class NLLEval(TrainingClientEvaluator):
            async def __call__(self, training_client):
                loss = await compute_nll(training_client, self.eval_data)
                return {"eval/nll": loss}
    """

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        """Run evaluation and return a dict of metric names to values.

        Args:
            training_client: The Tinker training client to evaluate with.

        Returns:
            A dict mapping metric names to float values.
        """
        raise NotImplementedError


class SamplingClientEvaluator:
    """An evaluator that uses a SamplingClient to compute metrics (e.g., accuracy).

    Example::

        class AccuracyEval(SamplingClientEvaluator):
            async def __call__(self, sampling_client):
                acc = await measure_accuracy(sampling_client, self.test_set)
                return {"eval/accuracy": acc}
    """

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Run evaluation and return a dict of metric names to values.

        Args:
            sampling_client: The Tinker sampling client to evaluate with.

        Returns:
            A dict mapping metric names to float values.
        """
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
