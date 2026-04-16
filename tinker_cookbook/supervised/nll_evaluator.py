import itertools

import tinker

from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset


class NLLEvaluator(TrainingClientEvaluator):
    """Evaluator that computes mean negative log-likelihood on held-out data.

    Uses the training client's ``forward_async`` to compute log-probabilities
    on a fixed set of datums and returns the weighted mean NLL.

    Attributes:
        name (str): Prefix for the returned metric key (default ``"test"``).
        data (list[tinker.Datum]): Evaluation datums.
    """

    def __init__(self, data: list[tinker.Datum], name: str = "test"):
        """Initialise the evaluator.

        Args:
            data (list[tinker.Datum]): Evaluation datums to score.
            name (str): Metric key prefix.  The returned dict will contain
                ``"{name}/nll"``.  Default ``"test"``.
        """
        self.name = name
        self.data = data

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        """Run a forward pass and return the mean NLL metric.

        Args:
            training_client (tinker.TrainingClient): Client whose current
                weights are evaluated.

        Returns:
            dict[str, float]: Single-entry dict ``{"{name}/nll": <value>}``.
        """
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        key = f"{self.name}/nll"
        return {key: nll}

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset, name: str = "test") -> "NLLEvaluator":
        """Create an evaluator from all batches of a ``SupervisedDataset``.

        Materialises every batch into a flat list of datums so the evaluator
        can score them in a single forward call.

        Args:
            dataset (SupervisedDataset): Dataset to draw evaluation data from.
            name (str): Metric key prefix. Default ``"test"``.

        Returns:
            NLLEvaluator: A new evaluator instance.
        """
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name)
