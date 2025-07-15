import tinker_public


class Evaluator:
    def __call__(self, training_client: tinker_public.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class EvaluatorBuilder:
    def __call__(self) -> Evaluator:
        raise NotImplementedError
