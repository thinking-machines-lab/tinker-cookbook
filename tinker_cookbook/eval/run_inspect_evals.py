import asyncio
import logging

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluator, InspectEvaluatorBuilder

logger = logging.getLogger(__name__)


@chz.chz
class Config(InspectEvaluatorBuilder):
    model_path: str | None = None


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    # Create a sampling client from the model path
    service_client = tinker.ServiceClient()
    model_path = config.model_path
    model_name = config.model_name
    renderer_name = config.renderer_name

    # Resolve model name from checkpoint when needed, and validate explicit model_name if provided.
    if model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if model_name is not None and model_name != training_run.base_model:
            raise ValueError(
                f"Model name {model_name} does not match training run base model {training_run.base_model}"
            )
        model_name = model_name or training_run.base_model

    if model_name is None:
        raise ValueError("model_path or model_name must be provided")

    # Resolve renderer with precedence: explicit config > checkpoint metadata > model default.
    if renderer_name is None and model_path is not None:
        renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
            service_client, model_path
        )
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)

    config = chz.replace(config, model_name=model_name, renderer_name=renderer_name)

    logger.info(f"Using base model: {config.model_name}")
    logger.info(f"Using renderer: {config.renderer_name}")

    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.model_name
    )

    # Run the evaluation
    logger.info(f"Running inspect evaluation for tasks: {config.tasks}")

    # Create the inspect evaluator
    evaluator = InspectEvaluator(config)
    metrics = await evaluator(sampling_client)

    # Print results
    logger.info("Inspect evaluation completed!")
    logger.info("Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
