import tinker_public
from tinker_cookbook.evaluators import Evaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_public import types


class NLLEvaluator(Evaluator):
    def __init__(self, data: list[types.Datum]):
        self.data = data

    def __call__(self, training_client: tinker_public.TrainingClient) -> dict[str, float]:
        result = training_client.forward(self.data, loss_fn="cross_entropy").result()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        return {"nll": nll}
