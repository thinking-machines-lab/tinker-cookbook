import asyncio
import itertools

import numpy as np
import tinker

from tinker_cookbook.eval import SamplingClientEvaluator
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_bpb, compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.deprecation import deprecated, warn_deprecated


class SamplerNLLEvaluator(SamplingClientEvaluator):
    """Compute mean negative log-likelihood on held-out data via a sampling client.

    Scores a fixed set of datums with ``compute_logprobs_async`` and returns the
    weighted mean NLL. When a ``tokenizer`` is supplied it also reports
    bits-per-byte, a tokenizer-independent NLL comparable across models.

    Attributes:
        data: Evaluation datums to score. Each must carry ``weights`` and
            ``target_tokens`` in ``loss_fn_inputs``, as produced by the
            ``supervised`` dataset helpers.
        name: Prefix for the returned metric keys (``"{name}/nll"``, and
            ``"{name}/bpb"`` when a tokenizer is given). Defaults to ``"test"``.
        tokenizer: If provided, also report ``"{name}/bpb"`` (bits per byte).
    """

    def __init__(
        self,
        data: list[tinker.Datum],
        name: str = "test",
        tokenizer: Tokenizer | None = None,
    ):
        self.data = data
        self.name = name
        self.tokenizer = tokenizer

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Score every datum concurrently and return the NLL (and optional BPB)."""
        logprobs: list[tinker.TensorData] = await asyncio.gather(
            *[self._compute_logprobs(sampling_client, datum) for datum in self.data]
        )
        return _compute_nll_metrics(logprobs, self.data, self.name, self.tokenizer)

    @staticmethod
    async def _compute_logprobs(
        sampling_client: tinker.SamplingClient, datum: tinker.Datum
    ) -> tinker.TensorData:
        """Return per-token logprobs for one datum, aligned with its weights.

        A datum's ``model_input`` is right-shifted (it drops the final token) and
        its weights align with ``target_tokens``. ``compute_logprobs_async``
        scores every position except the first, so we rebuild the full sequence by
        appending the last target token and discard the leading logprob. The
        result then lines up one-to-one with the datum's weights.
        """
        target_tokens = datum.loss_fn_inputs["target_tokens"].data
        full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
        logprobs = await sampling_client.compute_logprobs_async(full_sequence)
        return tinker.TensorData.from_numpy(np.array(logprobs[1:], dtype=np.float32))

    @classmethod
    def from_dataset(
        cls,
        dataset: SupervisedDataset,
        name: str = "test",
        tokenizer: Tokenizer | None = None,
    ) -> "SamplerNLLEvaluator":
        """Build an evaluator from every batch of a ``SupervisedDataset``."""
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name, tokenizer=tokenizer)


@deprecated(
    message="Use TrainingClient.save_weights_and_get_sampling_client() with SamplerNLLEvaluator instead"
)
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
        warn_deprecated(
            name="NLLEvaluator",
            message="NLLEvaluator is deprecated. Prefer to create a sampling client with TrainingClient.save_weights_and_get_sampling_client() and SamplerNLLEvaluator instead",
        )
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        return _compute_nll_metrics(logprobs, self.data, self.name, self.tokenizer)

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


def _compute_nll_metrics(
    logprobs_list: list[tinker.TensorData],
    data: list[tinker.Datum],
    name: str,
    tokenizer: Tokenizer | None,
) -> dict[str, float]:
    """Compute the NLL (and optionally BPB) metrics from per-datum logprobs.

    Shared by :class:`SamplerNLLEvaluator` and :class:`NLLEvaluator` so both
    report identical loss math; the evaluators differ only in how they obtain
    ``logprobs_list`` (sampling client vs. training client forward pass). Each
    entry in ``logprobs_list`` must be per-token log-probabilities aligned with
    the corresponding datum's ``loss_fn_inputs["weights"]``.

    Args:
        logprobs_list (list[tinker.TensorData]): Per-token log-probabilities,
            one entry per datum in ``data``, aligned with each datum's weights.
        data (list[tinker.Datum]): The scored datums, supplying weights and
            (for BPB) target tokens.
        name (str): Metric key prefix.
        tokenizer (Tokenizer | None): If provided, also report ``"{name}/bpb"``.

    Returns:
        dict[str, float]: ``{"{name}/nll": <value>}``, plus ``"{name}/bpb"``
        when a tokenizer was provided and the datums carry target tokens.
    """
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    metrics = {f"{name}/nll": compute_mean_nll(logprobs_list, weights)}
    if tokenizer is not None and data and "target_tokens" in data[0].loss_fn_inputs:
        target_tokens = [datum.loss_fn_inputs["target_tokens"] for datum in data]
        metrics[f"{name}/bpb"] = compute_bpb(logprobs_list, weights, target_tokens, tokenizer)
    return metrics
