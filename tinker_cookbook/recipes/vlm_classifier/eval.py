import asyncio
import logging
from typing import TypedDict, Any, cast
from PIL import Image

import datasets
import chz
import io

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, EvaluatorBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor, resize_image
from tinker_cookbook.renderers import Message, ImagePart, TextPart, ensure_text
from tinker_cookbook.utils.misc_utils import timed


# Set up logger
logger = logging.getLogger(__name__)


@chz.chz
class ClassifierEvaluatorConfig:
    """
    Configuration for classifier evaluation.
    """

    dataset: str
    dataset_split: str

    image_column_name: str = "image"
    label_column_name: str = "label"

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480


class ClassifierOutput(TypedDict):
    """
    Parsed output from an image classification model.
    """

    predicted_class_name: str


class ClassifierEvaluator(SamplingClientEvaluator):
    """
    Evaluator that runs image classification evaluation.
    """

    def __init__(
        self,
        config: ClassifierEvaluatorConfig,
    ):
        """
        Initialize the CustomEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """

        self.config = config

        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        image_processor = get_image_processor(self.config.model_name_for_tokenizer)

        self.renderer = renderers.get_renderer(
            name=self.config.renderer_name, tokenizer=tokenizer, image_processor=image_processor
        )

        dataset = datasets.load_dataset(self.config.dataset)
        dataset = cast(datasets.DatasetDict, dataset)
        self.dataset = dataset[self.config.dataset_split]

        self.shuffled_dataset = self.dataset.shuffle(seed=0)
        self.class_labels = self.dataset.features[self.config.label_column_name]

    def get_class_name(self, label: str) -> str:
        """
        Helper function to clean up the original class name.
        """

        return label.replace("_", " ").replace(".", " ").replace("-", " ").lower()

    def build_generation_prompt(
        self,
        example: dict[str, Any],
    ) -> tinker.ModelInput:
        """
        Generate an input to prompt the model.
        """

        image = example[self.config.image_column_name]
        pil_image: Image.Image | None = None

        if isinstance(image, dict) and "bytes" in image:
            pil_image = Image.open(io.BytesIO(image["bytes"]))

        elif isinstance(image, Image.Image):
            pil_image = cast(Image.Image, image)

        # If the dataset cannot be loaded
        if pil_image is None:
            raise AssertionError(f"Unable to interpret {image} as an image")

        pil_image = resize_image(image=pil_image, max_size=self.config.max_image_size)

        content_parts = [
            ImagePart(type="image", image=pil_image),
            TextPart(type="text", text="What is the name of the subject in this photo?"),
        ]

        messages = [
            Message(role="user", content=content_parts),
        ]

        return self.renderer.build_generation_prompt(
            messages=messages, role="assistant", prefill="The subject in this photo is:"
        )

    async def generate_output(
        self,
        model_input: tinker.ModelInput,
        sampling_client: tinker.SamplingClient,
        sampling_params: types.SamplingParams,
    ) -> ClassifierOutput:
        """
        Generate a completion and extract the class name from the model.
        """

        # Generate response
        r: types.SampleResponse = await sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )
        tokens: list[int] = r.sequences[0].tokens
        response = self.renderer.parse_response(tokens)[0]

        predicted_class_name = ensure_text(response["content"]).split(":")[-1].strip().lower()

        return ClassifierOutput(predicted_class_name=predicted_class_name)

    def get_metrics_for_output(
        self, example: dict[str, Any], classifier_output: ClassifierOutput
    ) -> dict[str, float]:
        """
        Score the class name predicted by the model.
        """

        predicted_class_name = classifier_output["predicted_class_name"]
        class_label = example[self.config.label_column_name]
        class_label_name = self.get_class_name(self.class_labels.int2str(class_label))

        return {"accuracy": float(predicted_class_name == class_label_name)}

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Evaluate a vision-language model as an image classifier.

        Args:
            sampling_client: The sampling client to evaluate

        Returns:
            Dictionary of metrics from evaluation

        """

        sampling_params = types.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stop=self.renderer.get_stop_sequences(),
        )

        num_examples = min(
            len(self.shuffled_dataset), self.config.n_eval or len(self.shuffled_dataset)
        )

        # Limit concurrent sampling tasks
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def bounded_generate_output(example: dict[str, Any]) -> ClassifierOutput:
            async with semaphore:
                return await self.generate_output(
                    self.build_generation_prompt(example), sampling_client, sampling_params
                )

        # Sample from the model in parallel
        async_tasks = []

        logger.info(
            f"Submitting {num_examples} sampling tasks (max {self.config.max_parallel_tasks} parallel)"
        )
        for example_id in range(num_examples):
            example = self.shuffled_dataset[example_id]

            # Prepare model input for sampling, generate
            async_tasks.append(asyncio.create_task(bounded_generate_output(example)))

        # Wait for the tinker API to return the sampled completions
        with timed("sample outputs", {}):
            outputs = await asyncio.gather(*async_tasks)

        # Aggregate metrics for each example
        metrics_per_example = []

        logger.info(f"Evaluating {num_examples} sampled responses")
        for example_id in range(num_examples):
            example = self.shuffled_dataset[example_id]
            output = outputs[example_id]

            # Evaluate the model response
            metrics = self.get_metrics_for_output(example, output)
            metrics_per_example.append(metrics)

        # aggregate the performance metrics
        aggregated_metrics = {
            key: np.mean([example[key] for example in metrics_per_example]).item()
            for key in metrics_per_example[0].keys()
        }

        return aggregated_metrics


@chz.chz
class Caltech101EvaluatorBuilder:
    """
    Configuration for classifier evaluation.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480

    def __call__(self) -> ClassifierEvaluator:
        config = ClassifierEvaluatorConfig(
            dataset="dpdl-benchmark/caltech101",
            dataset_split="test",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            image_column_name="image",
            label_column_name="label",
            n_eval=self.n_eval,
            max_parallel_tasks=self.max_parallel_tasks,
            max_image_size=self.max_image_size,
        )

        return ClassifierEvaluator(config)


@chz.chz
class Flowers102EvaluatorBuilder:
    """
    Configuration for classifier evaluation.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480

    def __call__(self) -> ClassifierEvaluator:
        config = ClassifierEvaluatorConfig(
            dataset="dpdl-benchmark/oxford_flowers102",
            dataset_split="test",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            image_column_name="image",
            label_column_name="label",
            n_eval=self.n_eval,
            max_parallel_tasks=self.max_parallel_tasks,
            max_image_size=self.max_image_size,
        )

        return ClassifierEvaluator(config)


@chz.chz
class OxfordPetsEvaluatorBuilder:
    """
    Configuration for classifier evaluation.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480

    def __call__(self) -> ClassifierEvaluator:
        config = ClassifierEvaluatorConfig(
            dataset="dpdl-benchmark/oxford_iiit_pet",
            dataset_split="test",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            image_column_name="image",
            label_column_name="label",
            n_eval=self.n_eval,
            max_parallel_tasks=self.max_parallel_tasks,
            max_image_size=self.max_image_size,
        )

        return ClassifierEvaluator(config)


@chz.chz
class StanfordCarsEvaluatorBuilder:
    """
    Configuration for classifier evaluation.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480

    def __call__(self) -> ClassifierEvaluator:
        config = ClassifierEvaluatorConfig(
            dataset="tanganke/stanford_cars",
            dataset_split="test",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            image_column_name="image",
            label_column_name="label",
            n_eval=self.n_eval,
            max_parallel_tasks=self.max_parallel_tasks,
            max_image_size=self.max_image_size,
        )

        return ClassifierEvaluator(config)


EVALUATORS = {
    "caltech101": Caltech101EvaluatorBuilder,
    "flowers102": Flowers102EvaluatorBuilder,
    "pets": OxfordPetsEvaluatorBuilder,
    "cars": StanfordCarsEvaluatorBuilder,
}


def get_evaluator_builder(
    dataset: str,
    model_name_for_tokenizer: str,
    renderer_name: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
    top_p: float = 1.0,
    top_k: int = -1,
    n_eval: int | None = None,
    max_parallel_tasks: int = 128,
    max_image_size: int = 480,
) -> EvaluatorBuilder:
    """
    Create a sampling based evaluator for a vlm classifier.
    """

    return EVALUATORS[dataset](
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        n_eval=n_eval,
        max_parallel_tasks=max_parallel_tasks,
        max_image_size=max_image_size,
    )


@chz.chz
class EvalConfig:
    """
    Config for launching evaluation on a model checkpoint.
    """

    dataset: str
    model_path: str

    renderer_name: str = "qwen3_vl"
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    # Infrastructure parameters
    base_url: str | None = None

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 128

    max_image_size: int = 480


def run_eval(eval_config: EvalConfig):
    """
    Launch evaluation on a model checkpoint on an image dataset.
    """

    service_client = tinker.ServiceClient(base_url=eval_config.base_url)
    sampling_client = service_client.create_sampling_client(model_path=eval_config.model_path)

    evaluator_builder = get_evaluator_builder(
        dataset=eval_config.dataset,
        model_name_for_tokenizer=eval_config.model_name,
        renderer_name=eval_config.renderer_name,
        temperature=eval_config.temperature,
        max_tokens=eval_config.max_tokens,
        top_p=eval_config.top_p,
        top_k=eval_config.top_k,
        n_eval=eval_config.n_eval,
        max_parallel_tasks=eval_config.max_parallel_tasks,
        max_image_size=eval_config.max_image_size,
    )

    evaluator = evaluator_builder()

    async def main():
        result = await evaluator(sampling_client)  # type: ignore[arg-type]
        print(f"Metrics = {result}")

    asyncio.run(main())


if __name__ == "__main__":
    chz.nested_entrypoint(run_eval)
