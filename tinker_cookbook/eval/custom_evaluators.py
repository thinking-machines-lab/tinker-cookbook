import asyncio
from typing import Any, Callable

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


class CustomEvaluator(SamplingClientEvaluator):
    """
    A toy SamplingClientEvaluator that runs a custom evaluation and returns its metrics.
    """

    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        """
        Initialize the CustomEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.dataset = dataset
        self.grader_fn = grader_fn

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """

        metrics = {}

        num_examples = len(self.dataset)
        num_correct = 0

        sampling_params = types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        for datum in self.dataset:
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["input"])]
            )
            # Generate response
            r: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens: list[int] = r.sequences[0].tokens
            response: renderers.Message = self.renderer.parse_response(tokens)[0]
            content = renderers.ensure_text(response["content"])
            if self.grader_fn(content, datum["output"]):
                num_correct += 1

        metrics["accuracy"] = num_correct / num_examples
        return metrics


QA_DATASET = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Germany?", "output": "Berlin"},
    {"input": "What is the capital of Italy?", "output": "Rome"},
]


def grader_fn(response: str, target: str) -> bool:
    return target.lower() in response.lower()


evaluator = CustomEvaluator(
    dataset=QA_DATASET,
    grader_fn=grader_fn,
    renderer_name="llama3",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    base_model="meta-llama/Llama-3.1-8B-Instruct"
)


async def main():
    result = await evaluator(sampling_client)
    print(result)


asyncio.run(main())
