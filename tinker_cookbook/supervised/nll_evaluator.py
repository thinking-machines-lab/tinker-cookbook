import itertools

import tinker

from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_bpb, compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import Tokenizer


class NLLEvaluator(TrainingClientEvaluator):
    """Evaluator that computes mean negative log-likelihood on held-out data.

    Uses the training client's ``forward_async`` to compute log-probabilities
    on a fixed set of datums and returns the weighted mean NLL.  When a
    ``tokenizer`` is supplied it additionally reports bits-per-byte, a
    tokenizer-independent NLL that is comparable across models.

    Attributes:
        name (str): Prefix for the returned metric key (default ``"test"``).
        data (list[tinker.Datum]): Evaluation datums.
        tokenizer (Tokenizer | None): Tokenizer used to compute bits-per-byte.
            When ``None``, only ``"{name}/nll"`` is reported.
    """

    def __init__(
        self,
        data: list[tinker.Datum],
        name: str = "test",
        tokenizer: Tokenizer | None = None,
    ):
        """Initialise the evaluator.

        Args:
            data (list[tinker.Datum]): Evaluation datums to score.
            name (str): Metric key prefix.  The returned dict will contain
                ``"{name}/nll"``.  Default ``"test"``.
            tokenizer (Tokenizer | None): If provided, also report
                ``"{name}/bpb"`` (bits per byte), a tokenizer-normalized NLL.
                Requires each datum to carry ``loss_fn_inputs["target_tokens"]``.
        """
        self.name = name
        self.data = data
        self.tokenizer = tokenizer

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        """Run a forward pass and return the NLL (and optionally BPB) metric.

        Args:
            training_client (tinker.TrainingClient): Client whose current
                weights are evaluated.

        Returns:
            dict[str, float]: ``{"{name}/nll": <value>}``, plus
            ``"{name}/bpb"`` when a tokenizer was provided.
        """
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        metrics = {f"{self.name}/nll": compute_mean_nll(logprobs, weights)}
        if (
            self.tokenizer is not None
            and self.data
            and "target_tokens" in self.data[0].loss_fn_inputs
        ):
            target_tokens = [datum.loss_fn_inputs["target_tokens"] for datum in self.data]
            metrics[f"{self.name}/bpb"] = compute_bpb(
                logprobs, weights, target_tokens, self.tokenizer
            )
        return metrics

    @classmethod
    def from_dataset(
        cls,
        dataset: SupervisedDataset,
        name: str = "test",
        tokenizer: Tokenizer | None = None,
    ) -> "NLLEvaluator":
        """Create an evaluator from all batches of a ``SupervisedDataset``.

        Materialises every batch into a flat list of datums so the evaluator
        can score them in a single forward call.

        Args:
            dataset (SupervisedDataset): Dataset to draw evaluation data from.
            name (str): Metric key prefix. Default ``"test"``.
            tokenizer (Tokenizer | None): If provided, also report
                ``"{name}/bpb"`` (bits per byte).

        Returns:
            NLLEvaluator: A new evaluator instance.
        """
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name, tokenizer=tokenizer)
