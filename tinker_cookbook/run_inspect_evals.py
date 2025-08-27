"""
Example:
    python -m tinker_cookbook.run_inspect_evals model_path=tinker://fabfd72a-1451-4da4-8edf-7f32ae31bcfc/sampler_weights/checkpoint_final tasks=paws renderer_name=role_colon model_name=meta-llama/Llama-3.1-8B
"""

import asyncio
import logging

import chz
import tinker

from tinker_cookbook.inspect_evaluators import InspectEvaluator, InspectEvaluatorBuilder

logger = logging.getLogger(__name__)


@chz.chz
class Config(InspectEvaluatorBuilder):
    model_path: str | None = None


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    # Create the inspect evaluator
    evaluator = InspectEvaluator(config)

    # Create a sampling client from the model path
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.model_name
    )

    # Run the evaluation
    logger.info(f"Running inspect evaluation for tasks: {config.tasks}")
    metrics = await evaluator(sampling_client)

    # Print results
    logger.info("Inspect evaluation completed!")
    logger.info("Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
