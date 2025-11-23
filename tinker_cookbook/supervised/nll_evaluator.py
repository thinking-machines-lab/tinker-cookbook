import itertools

import tinker
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset


class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[tinker.Datum], name: str = "test"):
        self.name = name
        self.data = data

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        key = f"{self.name}/nll"
        return {key: nll}

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset, name: str = "test") -> "NLLEvaluator":
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name)
